from rich.console import Console
from rich.table import Table
from memory.memory_store import FailureMemory, FailureRecord, _ast_fingerprint

console = Console()
memory = FailureMemory()

# ── Store 3 failure records ───────────────────────────────────────────────────
errors = [
    FailureRecord(
        error_type="AttributeError",
        error_message="'NoneType' object has no attribute 'search'",
        traceback="File retriever.py line 48 in retrieve\n  scores, indices = self.index.search()",
        failed_code="scores, indices = self.index.search(embedding, k)",
        fix_applied="Call _ensure_loaded() before search: if not self._loaded: self._ensure_loaded()",
        fix_worked=True, steps_taken=2,
        ast_fingerprint=_ast_fingerprint("scores, indices = self.index.search(embedding, k)"),
    ),
    FailureRecord(
        error_type="KeyError",
        error_message="KeyError: 'content' in chunk processing",
        traceback="File indexer.py line 92\n  content = chunk['content']",
        failed_code="content = chunk['content']",
        fix_applied="Use .get() with default: content = chunk.get('content', '')",
        fix_worked=True, steps_taken=1,
        ast_fingerprint=_ast_fingerprint("content = chunk['content']"),
    ),
    FailureRecord(
        error_type="AttributeError",
        error_message="'NoneType' object has no attribute 'encode'",
        traceback="File retriever.py line 31\n  emb = self.model.encode(text)",
        failed_code="emb = self.model.encode(text)",
        fix_applied="Check model is loaded: assert self.model is not None, 'Model not initialized'",
        fix_worked=False,  # this fix didn't work
        steps_taken=3,
        ast_fingerprint=_ast_fingerprint("emb = self.model.encode(text)"),
    ),
]

for e in errors:
    memory.store(e)

# ── Retrieval tests ───────────────────────────────────────────────────────────
queries = [
    ("'NoneType' has no attribute 'search'",
     "self.index.search(emb, k)",
     "AttributeError — should hit record 1 with structural bonus"),

    ("KeyError when accessing dictionary key",
     "val = data['key']",
     "KeyError — should hit record 2"),

    ("NoneType attribute error on model",
     "result = model.predict(x)",
     "AttributeError — worked_only should exclude record 3"),
]

table = Table(title="Memory Retrieval Tests")
table.add_column("Query", max_width=38)
table.add_column("Top Hit")
table.add_column("Score")
table.add_column("Struct Match")
table.add_column("Fix Worked")

for query, code, note in queries:
    results = memory.retrieve(query, failed_code=code, top_k=1, worked_only=True)
    if results:
        r = results[0]
        table.add_row(
            query[:38],
            r["error_type"],
            f"{r['score']:.3f}",
            "✓" if r["structural_match"] else "—",
            "[green]✓[/green]" if r["fix_worked"] else "[red]✗[/red]",
        )
    else:
        table.add_row(query[:38], "[dim]no hit[/dim]", "—", "—", "—")

console.print(table)

# ── Stats ─────────────────────────────────────────────────────────────────────
stats = memory.stats()
console.print(f"\nMemory stats: {stats}")
