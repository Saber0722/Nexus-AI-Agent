import ast
import hashlib
import json
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from rich.console import Console

from config import MEMORY_DB, INDEX_DIR

console = Console()
MEMORY_INDEX = INDEX_DIR / "memory.faiss"
MEMORY_META  = INDEX_DIR / "memory.meta.json"


@dataclass
class FailureRecord:
    error_type: str
    error_message: str
    traceback: str
    failed_code: str
    fix_applied: str
    fix_worked: bool
    steps_taken: int
    ast_fingerprint: str
    timestamp: float = field(default_factory=time.time)
    record_id: Optional[int] = None


def _ast_fingerprint(code: str) -> str:
    """
    Hash the structural shape of code — not its content.
    Two functions with different variable names but same structure
    get the same fingerprint.
    """
    try:
        tree = ast.parse(code)
        # Walk and collect node types only (ignore identifiers/values)
        node_types = [type(n).__name__ for n in ast.walk(tree)]
        structure = ",".join(node_types)
        return hashlib.md5(structure.encode()).hexdigest()[:16]
    except SyntaxError:
        # Unparseable code — fingerprint the error message structure instead
        return hashlib.md5(code[:200].encode()).hexdigest()[:16]


class FailureMemory:
    def __init__(self):
        self.db_path = MEMORY_DB
        self.model = SentenceTransformer(
            "nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True, device='cpu'
        )
        self.index: Optional[faiss.Index] = None
        self.meta: list[dict] = []
        self._init_db()
        self._load_index()

    # ── SQLite (persistent storage) ───────────────────────────────────────────

    def _init_db(self):
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS failures (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                error_type      TEXT,
                error_message   TEXT,
                traceback       TEXT,
                failed_code     TEXT,
                fix_applied     TEXT,
                fix_worked      INTEGER,
                steps_taken     INTEGER,
                ast_fingerprint TEXT,
                timestamp       REAL
            )
        """)
        conn.commit()
        conn.close()

    def store(self, record: FailureRecord) -> int:
        conn = sqlite3.connect(self.db_path)
        cur = conn.execute("""
            INSERT INTO failures
              (error_type, error_message, traceback, failed_code,
               fix_applied, fix_worked, steps_taken, ast_fingerprint, timestamp)
            VALUES (?,?,?,?,?,?,?,?,?)
        """, (
            record.error_type, record.error_message, record.traceback,
            record.failed_code, str(record.fix_applied), int(record.fix_worked),
            record.steps_taken, record.ast_fingerprint, record.timestamp,
        ))
        record_id = cur.lastrowid
        conn.commit()
        conn.close()

        # Add to FAISS index
        self._add_to_index(record, record_id)
        console.print(f"[dim]Memory stored: {record.error_type} (id={record_id})[/dim]")
        return record_id

    def mark_fix_failed(self, record_id: int):
        """Call this if a stored fix turned out to be wrong."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("UPDATE failures SET fix_worked=0 WHERE id=?", (record_id,))
        conn.commit()
        conn.close()

    # ── FAISS (semantic retrieval) ────────────────────────────────────────────

    def _load_index(self):
        if MEMORY_INDEX.exists() and MEMORY_META.exists():
            self.index = faiss.read_index(str(MEMORY_INDEX))
            self.meta = json.loads(MEMORY_META.read_text())
        else:
            self.index = faiss.IndexFlatIP(768)
            self.meta = []
            # Rebuild from SQLite if records exist but index is missing
            conn = sqlite3.connect(self.db_path)
            rows = conn.execute("SELECT * FROM failures").fetchall()
            conn.close()
            if rows:
                for row in rows:
                    rec = FailureRecord(
                        error_type=row[1], error_message=row[2],
                        traceback=row[3], failed_code=row[4],
                        fix_applied=row[5], fix_worked=bool(row[6]),
                        steps_taken=row[7], ast_fingerprint=row[8],
                        timestamp=row[9], record_id=row[0],
                    )
                    emb = self._embed_record(rec)
                    self.index.add(emb)
                    self.meta.append({
                        "record_id": row[0],
                        "error_type": row[1],
                        "ast_fingerprint": row[8],
                        "fix_worked": bool(row[6]),
                    })
                self._save_index()

    def _save_index(self):
        faiss.write_index(self.index, str(MEMORY_INDEX))
        MEMORY_META.write_text(json.dumps(self.meta, indent=2))

    def _embed_record(self, record: FailureRecord) -> np.ndarray:
        # Combine error message + fingerprint for embedding
        text = f"search_document: {record.error_type} {record.error_message} {record.ast_fingerprint}"
        emb = self.model.encode([text])
        emb = np.array(emb).astype("float32")
        faiss.normalize_L2(emb)
        return emb

    def _add_to_index(self, record: FailureRecord, record_id: int):
        emb = self._embed_record(record)
        self.index.add(emb)
        self.meta.append({
            "record_id": record_id,
            "error_type": record.error_type,
            "ast_fingerprint": record.ast_fingerprint,
            "fix_worked": record.fix_worked,
        })
        self._save_index()

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def retrieve(
        self,
        error_message: str,
        failed_code: str = "",
        top_k: int = 3,
        worked_only: bool = True,
    ) -> list[dict]:
        if self.index.ntotal == 0:
            return []

        # Hybrid score: semantic similarity + AST structural match bonus
        current_fp = _ast_fingerprint(failed_code) if failed_code else ""
        query_text = f"search_query: {error_message}"
        emb = self.model.encode([query_text])
        emb = np.array(emb).astype("float32")
        faiss.normalize_L2(emb)

        fetch_k = min(top_k * 4, self.index.ntotal)
        scores, indices = self.index.search(emb, fetch_k)

        conn = sqlite3.connect(self.db_path)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            m = self.meta[idx]
            if worked_only and not m["fix_worked"]:
                continue

            # Bonus: same AST fingerprint = structurally similar code
            structural_bonus = 0.15 if (current_fp and m["ast_fingerprint"] == current_fp) else 0.0
            final_score = float(score) + structural_bonus

            row = conn.execute(
                "SELECT * FROM failures WHERE id=?", (m["record_id"],)
            ).fetchone()
            if row:
                results.append({
                    "record_id": row[0],
                    "error_type": row[1],
                    "error_message": row[2],
                    "fix_applied": row[5],
                    "fix_worked": bool(row[6]),
                    "steps_taken": row[7],
                    "ast_fingerprint": row[8],
                    "score": final_score,
                    "structural_match": structural_bonus > 0,
                })

        conn.close()
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def stats(self) -> dict:
        conn = sqlite3.connect(self.db_path)
        total      = conn.execute("SELECT COUNT(*) FROM failures").fetchone()[0]
        worked     = conn.execute("SELECT COUNT(*) FROM failures WHERE fix_worked=1").fetchone()[0]
        error_types = conn.execute(
            "SELECT error_type, COUNT(*) FROM failures GROUP BY error_type"
        ).fetchall()
        conn.close()
        return {
            "total_records": total,
            "successful_fixes": worked,
            "hit_rate": f"{worked/total:.0%}" if total else "n/a",
            "error_types": dict(error_types),
            "index_size": self.index.ntotal,
        }