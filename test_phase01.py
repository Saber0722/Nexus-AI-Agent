from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rag.indexer import CodebaseIndexer
from rag.retriever import CodebaseRetriever

console = Console()

# 1. Build index on the nexus project itself
indexer = CodebaseIndexer(index_name="nexus_test")
indexer.build(Path("."))
indexer.stats()

# 2. Run 3 test queries
retriever = CodebaseRetriever(index_name="nexus_test")

queries = [
    "how are files chunked",
    "FAISS index creation",
    "what file extensions are supported",
]

console.print("\n[bold cyan]Retrieval Tests[/bold cyan]")
for q in queries:
    results = retriever.retrieve(q, top_k=2)
    console.print(Panel(
        "\n".join(f"[dim]{r.file_path}[/dim] → [bold]{r.name or r.chunk_type}[/bold] (score: {r.score:.3f})" for r in results),
        title=f"[yellow]Query:[/yellow] {q}",
        expand=False
    ))
