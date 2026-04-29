from dataclasses import dataclass, field
from llm.client import ModelTier


@dataclass
class AgentResult:
    success: bool
    output: str
    files_modified: list[str] = field(default_factory=list)
    steps_taken: int = 0
    error: str = ""

    def __bool__(self):
        return self.success