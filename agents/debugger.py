from llm.client import get_client, ModelTier
from llm.prompts import DEBUGGER_PROMPT
from rag.retriever import CodebaseRetriever
from tools.file_tool import edit_file
from tools.git_tool import git_checkpoint
from agents.base import AgentResult


def run(
    error: str,
    traceback: str = "",
    target_file: str = None,
    tier: ModelTier = ModelTier.QUALITY,
    retriever: CodebaseRetriever = None,
) -> AgentResult:
    client = get_client()

    query = f"{error} {traceback}"
    context = retriever.retrieve_for_prompt(query, top_k=5) if retriever else ""

    user_msg = f"Error: {error}\n\nTraceback:\n{traceback}"
    messages = DEBUGGER_PROMPT.build(context=context, user=user_msg)

    try:
        diagnosis = client.chat_json(messages, tier=tier, temperature=0.1)
        files_modified = []

        fixed_code = diagnosis.get("fixed_code", "")
        if fixed_code and target_file:
            git_checkpoint(f"before-debug-{target_file}")
            # Read current content to find what to replace
            from tools.file_tool import read_file
            current = read_file(target_file)
            if current.success:
                # Ask coder to apply the fix properly
                result = edit_file(target_file, current.output, fixed_code)
                if result.success:
                    files_modified.append(target_file)

        summary = (
            f"Root cause: {diagnosis.get('root_cause', 'unknown')}\n"
            f"Fix: {diagnosis.get('fix_description', '')}\n"
            f"Confidence: {diagnosis.get('confidence', 0):.0%}"
        )
        return AgentResult(
            success=True,
            output=summary,
            files_modified=files_modified,
            steps_taken=1,
        )
    except Exception as e:
        return AgentResult(success=False, output="", error=str(e))