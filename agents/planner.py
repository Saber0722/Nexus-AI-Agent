from llm.client import get_client, Message, ModelTier
from llm.prompts import PLANNER_PROMPT
from rag.retriever import CodebaseRetriever
from agents.base import AgentResult


def run(task: str, tier: ModelTier = ModelTier.QUALITY, retriever: CodebaseRetriever = None) -> AgentResult:
    client = get_client()
    context = ""
    if retriever:
        context = retriever.retrieve_for_prompt(task, top_k=5)

    messages = PLANNER_PROMPT.build(context=context, user=task)

    try:
        plan = client.chat_json(messages, tier=tier, temperature=0.2)
        steps = plan.get("steps", [])
        formatted = "\n".join(
            f"  {s['step_number']}. [{s['action_type']}] {s['description']}"
            + (f" → {s['target_file']}" if s.get('target_file') else "")
            for s in steps
        )
        return AgentResult(
            success=True,
            output=f"Plan ({len(steps)} steps):\n{formatted}\n\nNotes: {plan.get('notes', '')}",
            steps_taken=1,
        )
    except Exception as e:
        return AgentResult(success=False, output="", error=str(e))
