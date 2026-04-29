from rich.console import Console
from rich.table import Table
from orchestrator.agent_loop import AgentLoop

console = Console()

loop = AgentLoop(index_name="nexus_test", mode="balanced")

tasks = [
    ("single coder",    "Write a Python function called `flatten(lst)` that flattens one level of nesting"),
    ("single debugger", "AttributeError: 'NoneType' object has no attribute 'search' in retriever.py line 48"),
    ("rag only",        "Where is the FAISS index saved to disk?"),
    ("planner",         "Add error handling to the run_command tool so it logs failed commands to a file"),
]

summary = []
for label, task in tasks:
    console.rule(f"[cyan]{label}[/cyan]")
    loop.steps = []
    loop.files_modified = []
    result = loop.run(task)
    summary.append((label, result.success, len(result.steps), result.total_duration_ms))
    console.print(f"  Output: {result.final_output[:150]}\n")

# Summary table
table = Table(title="Phase 05 — Orchestrator Summary")
table.add_column("Task")
table.add_column("Success")
table.add_column("Steps")
table.add_column("Time")

for label, ok, steps, ms in summary:
    table.add_row(
        label,
        "[green]✓[/green]" if ok else "[red]✗[/red]",
        str(steps),
        f"{ms/1000:.1f}s"
    )

console.print(table)