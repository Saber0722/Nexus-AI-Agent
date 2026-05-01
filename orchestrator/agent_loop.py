import time
from dataclasses import dataclass, field
from rich.console import Console
from rich.panel import Panel

from llm.router import route, RoutingDecision
from llm.client import ModelTier
from rag.retriever import CodebaseRetriever
from tools.terminal_tool import run_command
from tools.python_exec import execute_python
from agents.base import AgentResult
import agents.planner as planner
import agents.coder as coder
import agents.debugger as debugger

console = Console()

MAX_STEPS = 15
MAX_RETRIES = 2  # per agent call


@dataclass
class StepLog:
    step: int
    agent: str
    task: str
    success: bool
    output: str
    duration_ms: int


@dataclass
class LoopResult:
    success: bool
    final_output: str
    steps: list[StepLog] = field(default_factory=list)
    files_modified: list[str] = field(default_factory=list)
    total_duration_ms: int = 0
    error: str = ""


class AgentLoop:
    def __init__(self, index_name: str = "default", mode: str = "balanced"):
        self.retriever = CodebaseRetriever(index_name=index_name)
        self.mode = mode
        self.steps: list[StepLog] = []
        self.files_modified: list[str] = []

    def _log_step(self, agent: str, task: str, result: AgentResult, duration_ms: int):
        step_n = len(self.steps) + 1
        log = StepLog(
            step=step_n,
            agent=agent,
            task=task[:80],
            success=result.success,
            output=result.output[:200],
            duration_ms=duration_ms,
        )
        self.steps.append(log)
        self.files_modified.extend(result.files_modified)

        status = "[green]✓[/green]" if result.success else "[red]✗[/red]"
        console.print(f"  {status} Step {step_n} [{agent}] {task[:60]}… ({duration_ms}ms)")
        if not result.success:
            console.print(f"    [red]Error: {result.error}[/red]")

        return log

    def _run_agent(self, decision: RoutingDecision, task: str) -> AgentResult:
        tier = decision.model_tier
        kwargs = dict(retriever=self.retriever, tier=tier)

        for attempt in range(1, MAX_RETRIES + 1):
            if attempt > 1:
                console.print(f"  [yellow]Retry {attempt}/{MAX_RETRIES}…[/yellow]")

            if decision.agent == "planner":
                result = planner.run(task=task, **kwargs)
            elif decision.agent == "coder":
                result = coder.run(task=task, **kwargs)
            elif decision.agent == "debugger":
                result = debugger.run(error=task, traceback="", **kwargs)
            elif decision.agent == "rag_only":
                # RAG-only: just retrieve and return context, no LLM generation
                context = self.retriever.retrieve_for_prompt(task, top_k=5)
                result = AgentResult(success=True, output=context, steps_taken=1)
            else:
                result = AgentResult(success=False, output="", error=f"Unknown agent: {decision.agent}")

            if result.success:
                return result

        return result  # return last failed result after retries

    def _reflect(self, result: AgentResult, task: str) -> bool:
        """
        Simple reflection: if coder produced code, try to execute it.
        Returns True if we should continue, False if we should stop.
        """
        # In _reflect(), change the first line to:
        if not result.success or "score:" in result.output:  # skip RAG retrieval output
            return False if not result.success else True

        output = result.output
        # If output looks like Python code, validate it compiles
        if "def " in output or "class " in output:
            exec_result = execute_python(f"compile({repr(output)}, '<check>', 'exec')")
            if not exec_result.success:
                console.print(f"  [yellow]⚠ Code has syntax errors: {exec_result.error[:100]}[/yellow]")
                # Don't stop — debugger will catch it in next iteration if needed

        return True

    def run(self, task: str) -> LoopResult:
        start = time.time()
        console.print(Panel(f"[bold]{task}[/bold]", title="[cyan]nexus[/cyan]", expand=False))

        # Step 1: Route the task
        decision = route(task, mode=self.mode)

        # Step 2: If planner, execute the plan steps sequentially
        if decision.agent == "planner":
            plan_result = self._run_agent(decision, task)
            t0 = time.time()
            self._log_step("planner", task, plan_result, int((time.time() - t0) * 1000))

            if not plan_result.success:
                return LoopResult(
                    success=False,
                    final_output="",
                    steps=self.steps,
                    error=plan_result.error,
                    total_duration_ms=int((time.time() - start) * 1000),
                )

            # Execute each plan step
            for line in plan_result.output.split("\n"):
                if len(self.steps) >= MAX_STEPS:
                    console.print("[yellow]Max steps reached.[/yellow]")
                    break

                line = line.strip()
                if not line or not line[0].isdigit():
                    continue

                # Route each sub-step
                sub_decision = route(line, mode=self.mode)
                t0 = time.time()
                sub_result = self._run_agent(sub_decision, line)
                self._log_step(sub_decision.agent, line, sub_result, int((time.time() - t0) * 1000))
                self._reflect(sub_result, line)

        else:
            # Single agent call
            t0 = time.time()
            result = self._run_agent(decision, task)
            self._log_step(decision.agent, task, result, int((time.time() - t0) * 1000))
            self._reflect(result, task)

        total_ms = int((time.time() - start) * 1000)
        last = self.steps[-1] if self.steps else None
        success = last.success if last else False
        output = last.output if last else ""

        return LoopResult(
            success=success,
            final_output=output,
            steps=self.steps,
            files_modified=list(set(self.files_modified)),
            total_duration_ms=total_ms,
        )