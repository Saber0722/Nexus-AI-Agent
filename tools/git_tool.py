from pathlib import Path
from tools.base import ToolResult, tool_result
from config import BASE_DIR
import git


def _repo():
    return git.Repo(BASE_DIR)


@tool_result
def git_status() -> str:
    repo = _repo()
    changed = [item.a_path for item in repo.index.diff(None)]
    untracked = repo.untracked_files
    staged = [item.a_path for item in repo.index.diff("HEAD")] if repo.head.is_valid() else []
    lines = []
    if staged:    lines.append(f"Staged:    {', '.join(staged)}")
    if changed:   lines.append(f"Modified:  {', '.join(changed)}")
    if untracked: lines.append(f"Untracked: {', '.join(untracked)}")
    return "\n".join(lines) or "Clean working tree"


@tool_result
def git_diff(path: str = None) -> str:
    repo = _repo()
    return repo.git.diff(path) if path else repo.git.diff()


@tool_result
def git_commit(message: str) -> str:
    repo = _repo()
    repo.git.add(A=True)
    if not repo.index.diff("HEAD") and repo.head.is_valid():
        return "Nothing to commit"
    commit = repo.index.commit(message)
    return f"Committed: {commit.hexsha[:8]} — {message}"


@tool_result
def git_checkpoint(label: str = "agent-checkpoint") -> str:
    """Auto-commit before any destructive agent action."""
    repo = _repo()
    repo.git.add(A=True)
    try:
        changed = repo.index.diff("HEAD")
    except Exception:
        changed = True  # first commit
    if not changed:
        return "No changes to checkpoint"
    commit = repo.index.commit(f"[nexus checkpoint] {label}")
    return f"Checkpoint: {commit.hexsha[:8]}"
