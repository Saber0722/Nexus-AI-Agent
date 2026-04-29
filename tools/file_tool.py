import difflib
from pathlib import Path
from tools.base import ToolResult, tool_result
from config import BASE_DIR


def _guard(path: Path) -> Path:
    """Prevent writes outside the project directory."""
    resolved = path.resolve()
    if not str(resolved).startswith(str(BASE_DIR.resolve())):
        raise PermissionError(f"Access denied outside project: {resolved}")
    return resolved


@tool_result
def read_file(path: str) -> str:
    p = Path(path)
    return p.read_text(encoding="utf-8")


@tool_result
def write_file(path: str, content: str) -> str:
    p = _guard(Path(path))
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return f"Written: {p} ({len(content)} chars)"


@tool_result
def edit_file(path: str, old_content: str, new_content: str) -> str:
    """Replace old_content with new_content in file. Shows unified diff."""
    p = _guard(Path(path))
    original = p.read_text(encoding="utf-8")

    if old_content not in original:
        raise ValueError(f"Target content not found in {path}. No changes made.")

    updated = original.replace(old_content, new_content, 1)
    p.write_text(updated, encoding="utf-8")

    diff = difflib.unified_diff(
        original.splitlines(keepends=True),
        updated.splitlines(keepends=True),
        fromfile=f"a/{path}",
        tofile=f"b/{path}",
        n=2,
    )
    return "".join(diff) or "No diff (content identical)"


@tool_result
def list_files(path: str = ".", pattern: str = "**/*") -> str:
    p = Path(path)
    files = [str(f.relative_to(p)) for f in p.glob(pattern) if f.is_file()]
    return "\n".join(sorted(files))