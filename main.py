import os, typer, re
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.syntax import Syntax

app = typer.Typer(help="nexus — local AI coding agent")
console = Console()

def _resolve(path_arg=None):
    cwd = Path(os.environ["NEXUS_CWD"]).resolve() if "NEXUS_CWD" in os.environ else Path.cwd()
    if not path_arg:
        return cwd
    p = Path(path_arg)
    return p.resolve() if p.is_absolute() else (cwd / p).resolve()

def _extract_code(text: str) -> str:
    match = re.search(r"```(?:\w+)?\n(.*?)```", text, re.DOTALL)
    return match.group(1).strip() if match else text.strip()

@app.command()
def run(
    task: str = typer.Argument(...),
    path: str = typer.Option(None, "--path", "-p"),
    mode: str = typer.Option("balanced", "--mode", "-m"),
    index: str = typer.Option(None, "--index", "-i"),
):
    from rag.indexer import CodebaseIndexer
    from orchestrator.agent_loop import AgentLoop
    from llm.client import get_client, Message, ModelTier
    p = _resolve(path)
    n = index or p.name
    console.print(f"[dim]Project: {p} | Index: {n}[/dim]")
    CodebaseIndexer(index_name=n).build(p)
    loop = AgentLoop(index_name=n, mode=mode)
    result = loop.run(task)

    # Only show write prompt for single-agent results (planner writes its own files)
    if not result.files_modified:
        code = _extract_code(result.final_output)
        if code and ("def " in code or "class " in code or "import " in code):
            console.print(Syntax(code, "python", theme="monokai", line_numbers=True))
            # Ask LLM to name the file
            from llm.client import get_client, Message, ModelTier
            client = get_client()
            fname_resp = client.chat(
                messages=[Message(role="user",
                    content=f"Task: \"{task}\"\nReply with ONE snake_case .py filename only. Nothing else.")],
                tier=ModelTier.FAST, temperature=0.1, max_tokens=20,
            )
            suggested = fname_resp.content.strip().split()[0]
            if not suggested.endswith(".py"):
                suggested += ".py"
            suggested = suggested.replace("/", "_").replace("-", "_")
            target_path = p / suggested
            target_path.write_text(code)
            console.print(f"[green]Written:[/green] {target_path}")

    if result.files_modified:
        console.print(f"[green]Files modified:[/green] {chr(44).join(result.files_modified)}")
    console.print(f"[dim]{result.total_duration_ms/1000:.1f}s | {len(result.steps)} steps[/dim]")


@app.command()
def index(
    path: str = typer.Argument(None),
    name: str = typer.Option(None, "--name", "-n"),
    force: bool = typer.Option(False, "--force", "-f"),
):
    from rag.indexer import CodebaseIndexer
    p = _resolve(path)
    n = name or p.name
    console.print(f"[dim]Project: {p} | Index: {n}[/dim]")
    CodebaseIndexer(index_name=n).build(p, force=force)


@app.command()
def ask(
    question: str = typer.Argument(...),
    path: str = typer.Option(None, "--path", "-p"),
    index: str = typer.Option(None, "--index", "-i"),
    top_k: int = typer.Option(5, "--top-k", "-k"),
    raw: bool = typer.Option(False, "--raw"),
):
    from rag.retriever import CodebaseRetriever
    from llm.client import get_client, Message, ModelTier
    p = _resolve(path)
    n = index or p.name
    retriever = CodebaseRetriever(index_name=n)
    try:
        results = retriever.retrieve(question, top_k=top_k)
    except FileNotFoundError:
        console.print("[yellow]No index found. Run: nexus index[/yellow]")
        return

    if not results:
        console.print("[yellow]No relevant code found.[/yellow]")
        return

    if raw:
        for r in results:
            console.print(Panel(r.content[:300],
                title=f"[cyan]{r.file_path}[/cyan] (score: {r.score:.3f})", expand=False))
        return

    context = "\n\n---\n\n".join(
        f"File: {r.file_path}\n{r.content}" for r in results
    )
    messages = [
        Message(role="system", content="Answer based ONLY on the provided code context. Be concise."),
        Message(role="user", content=f"Context:\n{context}\n\nQuestion: {question}"),
    ]
    client = get_client()
    console.print(f"[dim]Synthesizing from {len(results)} chunks...[/dim]\n")
    for token in client.stream(messages, tier=ModelTier.FAST):
        print(token, end="", flush=True)
    print()


@app.command()
def memory():
    from memory.memory_store import FailureMemory
    stats = FailureMemory().stats()
    t = Table(title="Failure Memory")
    t.add_column("Metric"); t.add_column("Value")
    for k, v in stats.items(): t.add_row(str(k), str(v))
    console.print(t)


if __name__ == "__main__":
    app()
