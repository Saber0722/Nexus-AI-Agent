from dataclasses import dataclass
from rich.console import Console

from llm.client import get_client, ModelTier, Message
from llm.prompts import ROUTER_PROMPT

console = Console()

CONFIDENCE_THRESHOLD = 0.65  # below this, escalate to 7B for routing


@dataclass
class RoutingDecision:
    agent: str           # "planner" | "coder" | "debugger" | "rag_only"
    confidence: float
    reasoning: str
    complexity: str      # "low" | "medium" | "high"
    model_tier: ModelTier
    routed_by: str       # which model made the routing decision


def route(task: str, mode: str = "balanced") -> RoutingDecision:
    """
    mode: "fast"     → always use 1.5B for routing, 1.5B for execution
          "balanced" → 1.5B routes, 7B executes (default)
          "quality"  → 7B routes and executes
    """
    client = get_client()

    # Step 1: try routing with fast model first
    messages = ROUTER_PROMPT.build(user=task)

    if mode == "quality":
        routing_tier = ModelTier.QUALITY
    else:
        routing_tier = ModelTier.FAST

    try:
        decision_raw = client.chat_json(messages, tier=routing_tier)
    except Exception as e:
        console.print(f"[yellow]Router failed, defaulting to planner: {e}[/yellow]")
        return RoutingDecision(
            agent="planner",
            confidence=0.0,
            reasoning="Router failed, safe default",
            complexity="medium",
            model_tier=ModelTier.QUALITY,
            routed_by="fallback",
        )

    agent = decision_raw.get("agent", "planner")
    confidence = float(decision_raw.get("confidence", 0.5))
    reasoning = decision_raw.get("reasoning", "")
    complexity = decision_raw.get("complexity", "medium")

    # Step 2: if confidence is low and we used the fast model, re-route with 7B
    routed_by = routing_tier.value
    if confidence < CONFIDENCE_THRESHOLD and routing_tier == ModelTier.FAST:
        console.print(
            f"[dim]Low confidence ({confidence:.2f}) from 1.5B router, "
            f"escalating to 7B...[/dim]"
        )
        try:
            decision_raw = client.chat_json(messages, tier=ModelTier.QUALITY)
            agent = decision_raw.get("agent", agent)
            confidence = float(decision_raw.get("confidence", confidence))
            reasoning = decision_raw.get("reasoning", reasoning)
            complexity = decision_raw.get("complexity", complexity)
            routed_by = "quality_escalation"
        except Exception:
            pass  # keep original fast-model decision

    # Step 3: decide execution tier based on mode + complexity
    if mode == "fast":
        exec_tier = ModelTier.FAST
    elif mode == "quality":
        exec_tier = ModelTier.QUALITY
    else:
        # balanced: use complexity to decide
        exec_tier = ModelTier.QUALITY if complexity in ("medium", "high") else ModelTier.FAST

    console.print(
        f"[dim]→ routed to [bold]{agent}[/bold] "
        f"(confidence: {confidence:.2f}, complexity: {complexity}, "
        f"exec: {exec_tier.value}, router: {routed_by})[/dim]"
    )

    return RoutingDecision(
        agent=agent,
        confidence=confidence,
        reasoning=reasoning,
        complexity=complexity,
        model_tier=exec_tier,
        routed_by=routed_by,
    )
