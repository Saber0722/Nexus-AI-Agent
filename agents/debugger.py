from llm.client import get_client, ModelTier
from llm.prompts import DEBUGGER_PROMPT
from rag.retriever import CodebaseRetriever
from tools.file_tool import edit_file
from tools.git_tool import git_checkpoint
from agents.base import AgentResult
from memory.memory_store import FailureMemory, FailureRecord, _ast_fingerprint
from rich.console import Console

console = Console()


_memory = FailureMemory()


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
        # Check memory first — do we have a past fix for this?
        past_fixes = _memory.retrieve(error, failed_code=traceback, top_k=2)
        memory_hint = ""
        if past_fixes:
            best = past_fixes[0]
            memory_hint = (
                f"\n\nPAST FIX (score:{best['score']:.2f}, "
                f"structural_match:{best['structural_match']}):\n{best['fix_applied']}"
            )
            console.print(f"  [dim magenta]Memory hit: {best['error_type']} "
                        f"(score:{best['score']:.2f})[/dim magenta]")

        user_msg = f"Error: {error}\n\nTraceback:\n{traceback}{memory_hint}"
        messages = DEBUGGER_PROMPT.build(context=context, user=user_msg)
        diagnosis = client.chat_json(messages, tier=tier, temperature=0.1)

        files_modified = []
        fixed_code = diagnosis.get("fixed_code", "")
        if fixed_code and target_file:
            git_checkpoint(f"before-debug-{target_file}")
            from tools.file_tool import read_file
            current = read_file(target_file)
            if current.success:
                result = edit_file(target_file, current.output, fixed_code)
                if result.success:
                    files_modified.append(target_file)

        # Store this interaction in memory
        _memory.store(FailureRecord(
            error_type=diagnosis.get("error_type", "Unknown"),
            error_message=error,
            traceback=traceback,
            failed_code=traceback,
            fix_applied=diagnosis.get("fixed_code", diagnosis.get("fix_description", "")),
            fix_worked=diagnosis.get("confidence", 0) > 0.6,
            steps_taken=1,
            ast_fingerprint=_ast_fingerprint(traceback),
        ))

        summary = (
            f"Root cause: {diagnosis.get('root_cause', 'unknown')}\n"
            f"Fix: {diagnosis.get('fix_description', '')}\n"
            f"Confidence: {diagnosis.get('confidence', 0.0):.0%}"
            + (f"\n[Memory hit: {past_fixes[0]['score']:.2f}]" if past_fixes else "")
        )
        return AgentResult(success=True, output=summary,
                        files_modified=files_modified, steps_taken=1)
    except Exception as e:
        return AgentResult(success=False, output="", error=str(e))