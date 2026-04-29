import time
from dataclasses import dataclass, asdict


@dataclass
class ToolResult:
    output: str
    error: str
    success: bool
    duration_ms: int

    def to_dict(self) -> dict:
        return asdict(self)

    def __bool__(self):
        return self.success


def tool_result(fn):
    """Decorator: wraps any tool function, catches exceptions, returns ToolResult."""
    def wrapper(*args, **kwargs):
        start = time.time()
        try:
            output = fn(*args, **kwargs)
            return ToolResult(
                output=str(output) if output is not None else "",
                error="",
                success=True,
                duration_ms=int((time.time() - start) * 1000),
            )
        except Exception as e:
            return ToolResult(
                output="",
                error=f"{type(e).__name__}: {e}",
                success=False,
                duration_ms=int((time.time() - start) * 1000),
            )
    wrapper.__name__ = fn.__name__
    return wrapper
