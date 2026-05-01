import time
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from rich.console import Console
from rich.table import Table

from orchestrator.agent_loop import AgentLoop
from rag.indexer import CodebaseIndexer

console = Console()

# ── Eval tasks ────────────────────────────────────────────────────────────────
TASKS = [
    # Bug fix
    {"id": "bf01", "category": "bug_fix",   "expected_agent": "debugger",
     "task": "AttributeError: 'NoneType' object has no attribute 'search' in retriever.py line 48"},
    {"id": "bf02", "category": "bug_fix",   "expected_agent": "debugger",
     "task": "KeyError: 'content' when iterating chunks in indexer.py"},
    {"id": "bf03", "category": "bug_fix",   "expected_agent": "debugger",
     "task": "TypeError: expected str not NoneType in embed call"},
    {"id": "bf04", "category": "bug_fix",   "expected_agent": "debugger",
     "task": "IndexError: list index out of range in retriever retrieve method"},
    {"id": "bf05", "category": "bug_fix",   "expected_agent": "debugger",
     "task": "ValueError: invalid JSON returned from LLM in chat_json method"},

    # Feature add
    {"id": "fa01", "category": "feature",   "expected_agent": "coder",
     "task": "Write a function called deduplicate_chunks that removes duplicate chunks by chunk_id"},
    {"id": "fa02", "category": "feature",   "expected_agent": "coder",
     "task": "Write a function called format_duration that converts milliseconds to a human readable string"},
    {"id": "fa03", "category": "feature",   "expected_agent": "coder",
     "task": "Write a function called truncate_content that shortens chunk content to N tokens"},
    {"id": "fa04", "category": "feature",   "expected_agent": "planner",
     "task": "Add retry logic with exponential backoff to the OllamaClient chat method"},
    {"id": "fa05", "category": "feature",   "expected_agent": "planner",
     "task": "Add a cache layer to the retriever so identical queries skip re-embedding"},

    # Refactor
    {"id": "rf01", "category": "refactor",  "expected_agent": "coder",
     "task": "Refactor the _extract_block_chunks function to use a generator instead of building a list"},
    {"id": "rf02", "category": "refactor",  "expected_agent": "coder",
     "task": "Rewrite the git_checkpoint function to return a ToolResult dataclass"},

    # RAG questions
    {"id": "rq01", "category": "rag_query", "expected_agent": "rag_only",
     "task": "Where is the FAISS index saved to disk?"},
    {"id": "rq02", "category": "rag_query", "expected_agent": "rag_only",
     "task": "What embedding model is used and why?"},
    {"id": "rq03", "category": "rag_query", "expected_agent": "rag_only",
     "task": "How does the router decide between fast and quality model?"},
    {"id": "rq04", "category": "rag_query", "expected_agent": "rag_only",
     "task": "What file extensions does the indexer support?"},
    {"id": "rq05", "category": "rag_query", "expected_agent": "rag_only",
     "task": "How is the AST fingerprint computed?"},

    # Multi-step
    {"id": "ms01", "category": "multi_step", "expected_agent": "planner",
     "task": "Add input validation to the route function in router.py to reject empty task strings"},
    {"id": "ms02", "category": "multi_step", "expected_agent": "planner",
     "task": "Add logging to the agent loop so each step is written to a log file"},
    {"id": "ms03", "category": "multi_step", "expected_agent": "planner",
     "task": "Add a timeout parameter to the CodebaseIndexer build method"},
]


@dataclass
class EvalResult:
    task_id: str
    category: str
    expected_agent: str
    actual_agent: str
    routed_correctly: bool
    success: bool
    steps_taken: int
    duration_ms: int
    memory_hit: bool = False
    error: str = ""


def run_eval(index_name: str = "nexus_eval", mode: str = "balanced") -> list[EvalResult]:
    # Index the project once
    console.print("[cyan]Indexing project for eval...[/cyan]")
    indexer = CodebaseIndexer(index_name=index_name)
    indexer.build(Path("."))

    results = []
    loop = AgentLoop(index_name=index_name, mode=mode)

    for i, task_def in enumerate(TASKS):
        console.rule(f"[dim]{task_def['id']} ({i+1}/{len(TASKS)})[/dim]")

        loop.steps = []
        loop.files_modified = []

        t0 = time.time()
        try:
            result = loop.run(task_def["task"])
            duration_ms = int((time.time() - t0) * 1000)

            actual_agent = loop.steps[0].agent if loop.steps else "unknown"
            memory_hit = "Memory hit" in (result.final_output or "")

            results.append(EvalResult(
                task_id=task_def["id"],
                category=task_def["category"],
                expected_agent=task_def["expected_agent"],
                actual_agent=actual_agent,
                routed_correctly=actual_agent == task_def["expected_agent"],
                success=result.success,
                steps_taken=len(result.steps),
                duration_ms=duration_ms,
                memory_hit=memory_hit,
            ))
        except Exception as e:
            results.append(EvalResult(
                task_id=task_def["id"],
                category=task_def["category"],
                expected_agent=task_def["expected_agent"],
                actual_agent="error",
                routed_correctly=False,
                success=False,
                steps_taken=0,
                duration_ms=int((time.time() - t0) * 1000),
                error=str(e)[:100],
            ))

    return results


def print_results(results: list[EvalResult]):
    # Per-task table
    table = Table(title="Eval Results — Per Task")
    table.add_column("ID", style="dim")
    table.add_column("Category")
    table.add_column("Expected")
    table.add_column("Got")
    table.add_column("Routed")
    table.add_column("Success")
    table.add_column("Steps")
    table.add_column("Time")

    for r in results:
        table.add_row(
            r.task_id,
            r.category,
            r.expected_agent,
            r.actual_agent,
            "[green]✓[/green]" if r.routed_correctly else "[red]✗[/red]",
            "[green]✓[/green]" if r.success else "[red]✗[/red]",
            str(r.steps_taken),
            f"{r.duration_ms/1000:.1f}s",
        )
    console.print(table)

    # Aggregate stats
    total        = len(results)
    routing_acc  = sum(r.routed_correctly for r in results) / total
    success_rate = sum(r.success for r in results) / total
    avg_steps    = sum(r.steps_taken for r in results) / total
    avg_time     = sum(r.duration_ms for r in results) / total / 1000

    by_category = {}
    for r in results:
        by_category.setdefault(r.category, []).append(r)

    summary = Table(title="Aggregate Metrics")
    summary.add_column("Metric")
    summary.add_column("Value")
    summary.add_row("Total tasks",       str(total))
    summary.add_row("Routing accuracy",  f"{routing_acc:.0%}")
    summary.add_row("Task success rate", f"{success_rate:.0%}")
    summary.add_row("Avg steps/task",    f"{avg_steps:.1f}")
    summary.add_row("Avg time/task",     f"{avg_time:.1f}s")

    for cat, cat_results in by_category.items():
        acc = sum(r.routed_correctly for r in cat_results) / len(cat_results)
        summary.add_row(f"  routing/{cat}", f"{acc:.0%} ({len(cat_results)} tasks)")

    console.print(summary)

    # Save raw results
    out = Path("evals/results.json")
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps([asdict(r) for r in results], indent=2))
    console.print(f"\n[dim]Raw results saved to {out}[/dim]")


if __name__ == "__main__":
    results = run_eval(mode="balanced")
    print_results(results)
