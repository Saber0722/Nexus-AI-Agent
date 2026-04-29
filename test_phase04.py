from rich.console import Console
from rich.panel import Panel
from rag.retriever import CodebaseRetriever
from llm.client import ModelTier
import agents.planner as planner
import agents.coder as coder
import agents.debugger as debugger

console = Console()
retriever = CodebaseRetriever(index_name="nexus_test")

# ── Planner ───────────────────────────────────────────────────────────────────
console.rule("[cyan]Planner Agent[/cyan]")
result = planner.run(
    task="Add a function to retriever.py that returns the top-5 most common file types in the index",
    retriever=retriever,
)
console.print(Panel(result.output or result.error, title=f"Planner — success:{result.success}"))

# ── Coder ─────────────────────────────────────────────────────────────────────
console.rule("[cyan]Coder Agent[/cyan]")
result = coder.run(
    task="Write a standalone Python function called `chunk_stats(chunks)` that takes a list of Chunk objects and returns a dict with keys: total, by_language, by_type",
    retriever=retriever,
)
console.print(Panel(result.output[:800] or result.error, title=f"Coder — success:{result.success}"))

# ── Debugger ──────────────────────────────────────────────────────────────────
console.rule("[cyan]Debugger Agent[/cyan]")
result = debugger.run(
    error="AttributeError: 'NoneType' object has no attribute 'search'",
    traceback="File 'rag/retriever.py', line 48, in retrieve\n    scores, indices = self.index.search(embedding, fetch_k)\nAttributeError: 'NoneType' object has no attribute 'search'",
    retriever=retriever,
)
console.print(Panel(result.output or result.error, title=f"Debugger — success:{result.success}"))

console.print("\n[bold green]Phase 04 complete.[/bold green]" if all([
    result.success
]) else "\n[red]Some agents failed — check output above.[/red]")