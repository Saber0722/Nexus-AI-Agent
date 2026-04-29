import sys
import io
import traceback
import contextlib
from tools.base import ToolResult
import time


def execute_python(code: str, timeout: int = 15) -> ToolResult:
    """
    Execute Python code in an isolated namespace.
    Returns stdout + return value. Never crashes the agent process.
    """
    start = time.time()
    stdout_capture = io.StringIO()
    namespace = {"__builtins__": __builtins__}

    try:
        with contextlib.redirect_stdout(stdout_capture):
            exec(compile(code, "<agent_exec>", "exec"), namespace)

        output = stdout_capture.getvalue().strip()
        duration_ms = int((time.time() - start) * 1000)
        return ToolResult(output=output or "(executed, no output)", error="", success=True, duration_ms=duration_ms)

    except Exception:
        error = traceback.format_exc()
        duration_ms = int((time.time() - start) * 1000)
        return ToolResult(output=stdout_capture.getvalue().strip(), error=error, success=False, duration_ms=duration_ms)
