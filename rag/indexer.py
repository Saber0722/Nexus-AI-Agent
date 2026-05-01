import ast
import hashlib
import json
import os
import pickle
import time
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field, asdict

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

from config import INDEX_DIR, CHUNK_SIZE, CHUNK_OVERLAP

console = Console()

SUPPORTED_EXTENSIONS = {".py", ".js", ".ts", ".md", ".json", ".yaml", ".yml", ".txt"}
IGNORE_DIRS = {"__pycache__", ".git", "node_modules", ".venv", "venv", "dist", "build", ".mypy_cache"}


@dataclass
class Chunk:
    chunk_id: str
    file_path: str
    content: str
    chunk_type: str        # "function" | "class" | "module_block" | "text_block"
    name: str              # function/class name, or "" for blocks
    start_line: int
    end_line: int
    language: str
    file_mtime: float

    def to_dict(self) -> dict:
        return asdict(self)


def _make_chunk_id(file_path: str, start_line: int, name: str) -> str:
    raw = f"{file_path}:{start_line}:{name}"
    return hashlib.md5(raw.encode()).hexdigest()[:12]


# ── Python AST chunker ────────────────────────────────────────────────────────

def _extract_python_chunks(file_path: Path, source: str, mtime: float) -> list[Chunk]:
    chunks = []
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        console.print(f"[yellow]  SyntaxError in {file_path}: {e}, falling back to block chunker[/yellow]")
        return _extract_block_chunks(file_path, source, "python", mtime)

    lines = source.splitlines()
    covered_lines: set[int] = set()

    # Extract top-level functions and classes (and nested ones inside classes)
    nodes_to_extract = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            nodes_to_extract.append(node)

    # Sort by start line
    nodes_to_extract.sort(key=lambda n: n.lineno)

    for node in nodes_to_extract:
        start = node.lineno - 1          # 0-indexed
        end = node.end_lineno            # exclusive

        # Skip if already covered by a parent (avoid duplicating nested funcs)
        if start in covered_lines:
            continue

        chunk_lines = lines[start:end]
        content = "\n".join(chunk_lines)

        if not content.strip():
            continue

        chunk_type = "function" if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) else "class"
        chunk = Chunk(
            chunk_id=_make_chunk_id(str(file_path), start + 1, node.name),
            file_path=str(file_path),
            content=content,
            chunk_type=chunk_type,
            name=node.name,
            start_line=start + 1,
            end_line=end,
            language="python",
            file_mtime=mtime,
        )
        chunks.append(chunk)
        covered_lines.update(range(start, end))

    # Catch module-level code not inside any function/class
    module_lines = []
    module_start = None
    for i, line in enumerate(lines):
        if i not in covered_lines:
            if module_start is None:
                module_start = i
            module_lines.append(line)
        else:
            if module_lines and len("\n".join(module_lines).strip()) > 30:
                content = "\n".join(module_lines)
                chunks.append(Chunk(
                    chunk_id=_make_chunk_id(str(file_path), module_start + 1, "__module__"),
                    file_path=str(file_path),
                    content=content,
                    chunk_type="module_block",
                    name="",
                    start_line=module_start + 1,
                    end_line=i,
                    language="python",
                    file_mtime=mtime,
                ))
            module_lines = []
            module_start = None

    return chunks


# ── Generic sliding window chunker ───────────────────────────────────────────

def _extract_block_chunks(file_path: Path, content: str, language: str, mtime: float) -> list[Chunk]:
    chunks = []
    words = content.split()
    step = CHUNK_SIZE - CHUNK_OVERLAP
    lines = content.splitlines()

    for i in range(0, len(words), step):
        snippet = " ".join(words[i: i + CHUNK_SIZE])
        if not snippet.strip():
            continue

        # Approximate line number
        char_offset = len(" ".join(words[:i]))
        approx_line = content[:char_offset].count("\n") + 1

        chunks.append(Chunk(
            chunk_id=_make_chunk_id(str(file_path), approx_line, f"block_{i}"),
            file_path=str(file_path),
            content=snippet,
            chunk_type="text_block",
            name="",
            start_line=approx_line,
            end_line=min(approx_line + 30, len(lines)),
            language=language,
            file_mtime=mtime,
        ))

    return chunks


# ── File dispatcher ───────────────────────────────────────────────────────────

def chunk_file(file_path: Path) -> list[Chunk]:
    mtime = file_path.stat().st_mtime
    try:
        source = file_path.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        console.print(f"[red]  Cannot read {file_path}: {e}[/red]")
        return []

    ext = file_path.suffix.lower()

    if ext == ".py":
        return _extract_python_chunks(file_path, source, mtime)
    elif ext in {".js", ".ts"}:
        return _extract_block_chunks(file_path, source, "javascript", mtime)
    elif ext == ".md":
        return _extract_block_chunks(file_path, source, "markdown", mtime)
    elif ext in {".json", ".yaml", ".yml"}:
        return _extract_block_chunks(file_path, source, "config", mtime)
    else:
        return _extract_block_chunks(file_path, source, "text", mtime)


# ── Index builder ─────────────────────────────────────────────────────────────

class CodebaseIndexer:
    def __init__(self, index_name: str = "default"):
        self.index_name = index_name
        self.index_path = INDEX_DIR / f"{index_name}.faiss"
        self.meta_path = INDEX_DIR / f"{index_name}.meta.pkl"
        self.mtime_path = INDEX_DIR / f"{index_name}.mtimes.json"

        self.model = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True, device='cpu')
        self.embedding_dim = 768

        self.index: Optional[faiss.Index] = None
        self.chunks: list[Chunk] = []
        self.file_mtimes: dict[str, float] = {}

    def _load_existing(self) -> bool:
        if self.index_path.exists() and self.meta_path.exists():
            self.index = faiss.read_index(str(self.index_path))
            with open(self.meta_path, "rb") as f:
                self.chunks = pickle.load(f)
            if self.mtime_path.exists():
                self.file_mtimes = json.loads(self.mtime_path.read_text())
            console.print(f"[dim]Loaded existing index: {len(self.chunks)} chunks[/dim]")
            return True
        return False

    def _save(self):
        faiss.write_index(self.index, str(self.index_path))
        with open(self.meta_path, "wb") as f:
            pickle.dump(self.chunks, f)
        self.mtime_path.write_text(json.dumps(self.file_mtimes, indent=2))

    def _collect_files(self, root: Path) -> list[Path]:
        files = []
        for path in root.rglob("*"):
            if any(ignored in path.parts for ignored in IGNORE_DIRS):
                continue
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
                files.append(path)
        return sorted(files)

    def _files_needing_reindex(self, files: list[Path]) -> list[Path]:
        changed = []
        for f in files:
            mtime = f.stat().st_mtime
            if str(f) not in self.file_mtimes or self.file_mtimes[str(f)] != mtime:
                changed.append(f)
        return changed

    def build(self, root: Path, force: bool = False):
        root = root.resolve()
        console.print(f"\n[cyan]Indexing:[/cyan] {root}")

        existing = self._load_existing() if not force else False

        all_files = self._collect_files(root)
        console.print(f"Found [bold]{len(all_files)}[/bold] files")

        if existing:
            to_index = self._files_needing_reindex(all_files)
            console.print(f"[dim]{len(all_files) - len(to_index)} files unchanged, {len(to_index)} need reindex[/dim]")

            # Remove stale chunks for changed files
            changed_paths = {str(f) for f in to_index}
            self.chunks = [c for c in self.chunks if c.file_path not in changed_paths]
        else:
            to_index = all_files
            self.chunks = []

        if not to_index:
            console.print("[green]Index is up to date.[/green]")
            return

        # Chunk all files that need indexing
        new_chunks: list[Chunk] = []
        with Progress(SpinnerColumn(), TextColumn("{task.description}"), BarColumn(), console=console) as progress:
            task = progress.add_task("Chunking files...", total=len(to_index))
            for f in to_index:
                new_chunks.extend(chunk_file(f))
                self.file_mtimes[str(f)] = f.stat().st_mtime
                progress.advance(task)

        console.print(f"Generated [bold]{len(new_chunks)}[/bold] new chunks")

        if not new_chunks:
            console.print("[yellow]No chunks produced.[/yellow]")
            return

        # Embed
        texts = [f"search_document: {c.content}" for c in new_chunks]
        with Progress(SpinnerColumn(), TextColumn("{task.description}"), console=console) as progress:
            task = progress.add_task("Embedding chunks...", total=None)
            embeddings = self.model.encode(texts, batch_size=64, show_progress_bar=False)
            progress.update(task, completed=True)

        embeddings = np.array(embeddings).astype("float32")
        faiss.normalize_L2(embeddings)   # cosine similarity via normalized L2

        # Build or extend FAISS index
        self.chunks.extend(new_chunks)

        all_texts = [f"search_document: {c.content}" for c in self.chunks]
        all_embeddings = self.model.encode(all_texts, batch_size=64, show_progress_bar=False)
        all_embeddings = np.array(all_embeddings).astype("float32")
        faiss.normalize_L2(all_embeddings)

        self.index = faiss.IndexFlatIP(self.embedding_dim)   # Inner product = cosine on normalized vecs
        self.index.add(all_embeddings)

        self._save()
        console.print(f"[green]Index saved:[/green] {len(self.chunks)} total chunks, {self.index.ntotal} vectors")

    def stats(self):
        if not self.chunks:
            self._load_existing()
        if not self.chunks:
            console.print("[yellow]No index found. Run build() first.[/yellow]")
            return

        from collections import Counter
        langs = Counter(c.language for c in self.chunks)
        types = Counter(c.chunk_type for c in self.chunks)
        files = len(set(c.file_path for c in self.chunks))

        console.print(f"\n[bold]Index Stats[/bold]")
        console.print(f"  Total chunks : {len(self.chunks)}")
        console.print(f"  Unique files : {files}")
        console.print(f"  By language  : {dict(langs)}")
        console.print(f"  By type      : {dict(types)}")
