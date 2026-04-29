import subprocess
from tools.base import ToolResult, tool_result
from config import BASE_DIR

# Commands that are never allowed regardless of context
BLOCKED = {
    "rm -rf /", "rm -rf ~", ":(){ :|:& };:",   # destructive
    "sudo", "su ",                               # privilege escalation
    "curl | bash", "wget | bash",               # remote execution
    "dd if=", "mkfs",                           # disk operations
}


def _check_blocked(command: str):
    for blocked in BLOCKED:
        if blocked in command:
            raise PermissionError(f"Blocked command: '{blocked}'")


@tool_result
def run_command(
    command: str,
    cwd: str = None,
    timeout: int = 30,
) -> str:
    _check_blocked(command)
    work_dir = cwd or str(BASE_DIR)

    result = subprocess.run(
        command,
        shell=True,
        cwd=work_dir,
        capture_output=True,
        text=True,
        timeout=timeout,
    )

    output = result.stdout.strip()
    stderr = result.stderr.strip()

    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed (exit {result.returncode})\n"
            f"stdout: {output}\nstderr: {stderr}"
        )

    return output or stderr or "(no output)"
