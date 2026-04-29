import time
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import print as rprint

from llm.client import get_client, ModelTier, Message
from llm.router import route

console = Console()

client = get_client()

# ── Test 1: Basic chat (non-streaming) ───────────────────────────────────────
console.rule("[bold cyan]Test 1: Basic Chat[/bold cyan]")

response = client.chat(
    messages=[Message(role="user", content="Write a Python one-liner to flatten a nested list.")],
    tier=ModelTier.FAST,
)
console.print(Panel(
    f"{response.content}\n\n[dim]model: {response.model} | "
    f"{response.duration_ms}ms | "
    f"{response.prompt_tokens}→{response.response_tokens} tokens[/dim]",
    title="1.5B Response"
))

# ── Test 2: Streaming ────────────────────────────────────────────────────────
console.rule("[bold cyan]Test 2: Streaming[/bold cyan]")
console.print("[dim]Streaming from 1.5B:[/dim] ", end="")

tokens_received = 0
for token in client.stream(
    messages=[Message(role="user", content="In one sentence, what is a FAISS index?")],
    tier=ModelTier.FAST,
):
    print(token, end="", flush=True)
    tokens_received += 1
print()
console.print(f"[dim]({tokens_received} chunks received)[/dim]")

# ── Test 3: JSON structured output ───────────────────────────────────────────
console.rule("[bold cyan]Test 3: JSON Mode[/bold cyan]")

result = client.chat_json(
    messages=[
        Message(role="system", content='Return only valid JSON with keys: "name", "version", "ready"'),
        Message(role="user", content="Give me a status object for a system called nexus at version 0.1"),
    ],
    tier=ModelTier.FAST,
)
rprint(result)
assert isinstance(result, dict), "JSON mode should return a dict"
console.print("[green]JSON mode: PASS[/green]")

# ── Test 4: Router decisions ─────────────────────────────────────────────────
console.rule("[bold cyan]Test 4: Router[/bold cyan]")

test_tasks = [
    ("Fix the KeyError on line 42 in auth.py", "debugger"),
    ("Add JWT authentication to the login endpoint", "planner"),
    ("Write a function to parse CSV files", "coder"),
    ("Where is the database connection handled?", "rag_only"),
]

table = Table(title="Routing Decisions")
table.add_column("Task", max_width=45)
table.add_column("Expected")
table.add_column("Got")
table.add_column("Confidence")
table.add_column("Match")

for task, expected in test_tasks:
    decision = route(task, mode="balanced")
    match = "✓" if decision.agent == expected else "✗"
    color = "green" if match == "✓" else "yellow"
    table.add_row(
        task[:45],
        expected,
        decision.agent,
        f"{decision.confidence:.2f}",
        f"[{color}]{match}[/{color}]"
    )

console.print(table)

# ── Test 5: Model tier speed comparison ──────────────────────────────────────
console.rule("[bold cyan]Test 5: Speed — 1.5B vs 7B[/bold cyan]")

prompt = [Message(role="user", content="Write a Python function to binary search a sorted list.")]

t0 = time.time()
r_fast = client.chat(prompt, tier=ModelTier.FAST)
fast_ms = int((time.time() - t0) * 1000)

t0 = time.time()
r_quality = client.chat(prompt, tier=ModelTier.QUALITY)
quality_ms = int((time.time() - t0) * 1000)

speed_table = Table(title="Speed Comparison")
speed_table.add_column("Model")
speed_table.add_column("Time")
speed_table.add_column("Tokens out")
speed_table.add_row("qwen2.5:1.5b", f"{fast_ms}ms", str(r_fast.response_tokens))
speed_table.add_row("qwen2.5:7b",   f"{quality_ms}ms", str(r_quality.response_tokens))
speed_table.add_row("[dim]speedup[/dim]", f"[green]{quality_ms/fast_ms:.1f}x faster[/green]", "")

console.print(speed_table)
console.print("\n[bold green]Phase 02 complete.[/bold green]")
