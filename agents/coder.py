from llm.client import get_client, Message, ModelTier
from llm.prompts import CODER_PROMPT
from rag.retriever import CodebaseRetriever
from tools.file_tool import write_file, edit_file
from tools.git_tool import git_checkpoint
from agents.base import AgentResult
import re


def _extract_code(text: str) -> str:
    """Strip markdown fences if present."""
    match = re.search(r"```(?:\w+)?\n(.*?)```", text, re.DOTALL)
    return match.group(1).strip() if match else text.strip()


def run(
    task: str,
    target_file: str = None,
    tier: ModelTier = ModelTier.QUALITY,
    retriever: CodebaseRetriever = None,
) -> AgentResult:
    client = get_client()
    context = ""
    if retriever:
        context = retriever.retrieve_for_prompt(task, top_k=5)

    messages = CODER_PROMPT.build(context=context, user=task)
    files_modified = []

    try:
        response = client.chat(messages, tier=tier, temperature=0.2)
        code = _extract_code(response.content)

        if target_file:
            git_checkpoint(f"before-coder-{target_file}")
            result = write_file(target_file, code)
            if not result.success:
                return AgentResult(success=False, output=code, error=result.error)
            files_modified.append(target_file)

        return AgentResult(
            success=True,
            output=code,
            files_modified=files_modified,
            steps_taken=1,
        )
    except Exception as e:
        return AgentResult(success=False, output="", error=str(e))