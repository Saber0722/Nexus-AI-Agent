import pickle
from pathlib import Path
from dataclasses import dataclass

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from rich.console import Console

from config import INDEX_DIR, TOP_K_RESULTS

console = Console()


@dataclass
class RetrievalResult:
    chunk_id: str
    file_path: str
    content: str
    chunk_type: str
    name: str
    start_line: int
    end_line: int
    language: str
    score: float

    def format_for_prompt(self) -> str:
        header = f"# [{self.language}] {self.file_path}"
        if self.name:
            header += f" → {self.chunk_type}: {self.name}"
        header += f" (lines {self.start_line}–{self.end_line}, score: {self.score:.3f})"
        return f"{header}\n{self.content}"


class CodebaseRetriever:
    def __init__(self, index_name: str = "default"):
        self.index_name = index_name
        self.index_path = INDEX_DIR / f"{index_name}.faiss"
        self.meta_path = INDEX_DIR / f"{index_name}.meta.pkl"

        self.model = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True, device='cpu')
        self.index = None
        self.chunks = []
        self._loaded = False

    def _ensure_loaded(self):
        if self._loaded:
            return
        if not self.index_path.exists():
            raise FileNotFoundError(
                f"No index found at {self.index_path}. Run 'nexus index <path>' first."
            )
        self.index = faiss.read_index(str(self.index_path))
        with open(self.meta_path, "rb") as f:
            self.chunks = pickle.load(f)
        self._loaded = True

    def retrieve(self, query: str, top_k: int = TOP_K_RESULTS, lang_filter: str = None) -> list[RetrievalResult]:
        self._ensure_loaded()

        embedding = self.model.encode([f"search_query: {query}"])
        embedding = np.array(embedding).astype("float32")
        faiss.normalize_L2(embedding)

        # Fetch more than needed if filtering by language
        fetch_k = top_k * 3 if lang_filter else top_k
        fetch_k = min(fetch_k, len(self.chunks))

        scores, indices = self.index.search(embedding, fetch_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            chunk = self.chunks[idx]
            if lang_filter and chunk.language != lang_filter:
                continue
            results.append(RetrievalResult(
                chunk_id=chunk.chunk_id,
                file_path=chunk.file_path,
                content=chunk.content,
                chunk_type=chunk.chunk_type,
                name=chunk.name,
                start_line=chunk.start_line,
                end_line=chunk.end_line,
                language=chunk.language,
                score=float(score),
            ))
            if len(results) >= top_k:
                break

        return results

    def retrieve_for_prompt(self, query: str, top_k: int = TOP_K_RESULTS, lang_filter: str = None) -> str:
        results = self.retrieve(query, top_k, lang_filter)
        if not results:
            return "No relevant code found in index."
        return "\n\n---\n\n".join(r.format_for_prompt() for r in results)
