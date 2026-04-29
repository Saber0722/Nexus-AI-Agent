from rich.console import Console
from rich.table import Table
from pathlib import Path
import tempfile, os

from tools.file_tool import read_file, write_file, edit_file, list_files
from tools.terminal_tool import run_command
from tools.python_exec import execute_python
from tools.git_tool import git_status, git_checkpoint

console = Console()
results = []

def check(name, result, expect_success=True):
    ok = result.success == expect_success
    results.append((name, ok, result.error or result.output[:60]))
    return result

# ── File tools ────────────────────────────────────────────────────────────────
r = check("write_file", write_file("data/test_output.txt", "hello from nexus\n"))
r = check("read_file",  read_file("data/test_output.txt"))
assert "hello" in r.output

r = check("edit_file",  edit_file("data/test_output.txt", "hello from nexus", "hello EDITED"))
r = check("read_file after edit", read_file("data/test_output.txt"))
assert "EDITED" in r.output

r = check("list_files", list_files("tools"))

# ── Security guard ────────────────────────────────────────────────────────────
r = check("path traversal blocked", write_file("/tmp/evil.txt", "bad"), expect_success=False)

# ── Terminal tool ─────────────────────────────────────────────────────────────
r = check("run echo",      run_command("echo 'nexus works'"))
r = check("run python -V", run_command("python --version"))
r = check("blocked rm",    run_command("sudo ls"), expect_success=False)

# ── Python exec ───────────────────────────────────────────────────────────────
r = check("exec print",    execute_python("print(2 + 2)"))
assert r.output == "4"

r = check("exec error",    execute_python("raise ValueError('test error')"), expect_success=False)
assert "ValueError" in r.error

r = check("exec multiline", execute_python("""
xs = list(range(5))
print(sum(xs))
"""))
assert r.output == "10"

# ── Git tools ─────────────────────────────────────────────────────────────────
r = check("git_status",     git_status())
r = check("git_checkpoint", git_checkpoint("phase03-test"))

# ── Results table ─────────────────────────────────────────────────────────────
table = Table(title="Phase 03 — Tool Layer")
table.add_column("Test")
table.add_column("Status")
table.add_column("Detail", max_width=55)

for name, ok, detail in results:
    table.add_row(name, "[green]PASS[/green]" if ok else "[red]FAIL[/red]", detail)

console.print(table)
passed = sum(1 for _, ok, _ in results if ok)
console.print(f"\n[bold]{'All' if passed == len(results) else passed}/{len(results)} passed[/bold]")