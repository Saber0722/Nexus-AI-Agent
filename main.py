import typer
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

app = typer.Typer(help="nexus — local AI coding agent")
console = Console()


@app.command()
def run(
    task: str = typer.Argument(..., help="Task to perform"),
    path: Path = typer.Option(Path("."), "--path", "-p", help="Project path to index"),
    mode: str = typer.Option("balanced", "--mode", "-m", help="fast | balanced | quality"),
    index: str = typer.Option("default", "--index", "-i", help="Index name"),
):
    """Run the agent on a task."""
    from rag.indexer import CodebaseIndexer
    from orchestrator.agent_loop import AgentLoop

    # Auto-reindex if path given
    indexer = CodebaseIndexer(index_name=index)
    indexer.build(path)

    loop = AgentLoop(index_name=index, mode=mode)
    result = loop.run(task)

    if result.files_modified:
        console.print(f"\n[green]Files modified:[/green] {', '.join(result.files_modified)}")
    console.print(f"[dim]Total time: {result.total_duration_ms/1000:.1f}s | Steps: {len(result.steps)}[/dim]")


@app.command()
def index(
    path: Path = typer.Argument(Path("."), help="Path to index"),
    name: str = typer.Option("default", "--name", "-n"),
    force: bool = typer.Option(False, "--force", "-f", help="Force full reindex"),
):
    """Index a codebase."""
    from rag.indexer import CodebaseIndexer
    indexer = CodebaseIndexer(index_name=name)
    indexer.build(path, force=force)
    indexer.stats()


@app.command()
def memory():
    """Show memory store stats."""
    from memory.memory_store import FailureMemory
    m = FailureMemory()
    stats = m.stats()
    table = Table(title="Failure Memory")
    table.add_column("Metric")
    table.add_column("Value")
    for k, v in stats.items():
        table.add_row(str(k), str(v))
    console.print(table)


@app.command()
def ask(
    question: str = typer.Argument(..., help="Question about the codebase"),
    index: str = typer.Option("default", "--index", "-i"),
    top_k: int = typer.Option(5, "--top-k", "-k"),
):
    """Ask a question about the indexed codebase (RAG only, no LLM)."""
    from rag.retriever import CodebaseRetriever
    retriever = CodebaseRetriever(index_name=index)
    results = retriever.retrieve(question, top_k=top_k)
    for r in results:
        console.print(Panel(
            r.content[:300],
            title=f"[cyan]{r.file_path}[/cyan] → {r.name or r.chunk_type} (score: {r.score:.3f})",
            expand=False,
        ))


if __name__ == "__main__":
    app()
