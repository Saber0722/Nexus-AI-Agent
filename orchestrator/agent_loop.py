import time
from dataclasses import dataclass, field
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.prompt import Confirm
from pathlib import Path
import os

from llm.router import route, RoutingDecision
from llm.client import get_client, Message, ModelTier
from rag.retriever import CodebaseRetriever
from tools.terminal_tool import run_command
from tools.python_exec import execute_python
from agents.base import AgentResult
import agents.planner as planner
import agents.coder as coder
import agents.debugger as debugger

console = Console()

MAX_STEPS = 15
MAX_RETRIES = 2


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
        self._last_full_output = ""
        # Resolve caller project path from env
        self.project_path = Path(os.environ.get("NEXUS_CWD", Path.cwd())).resolve()

    def _log_step(self, agent: str, task: str, result: AgentResult, duration_ms: int):
        step_n = len(self.steps) + 1
        log = StepLog(
            step=step_n,
            agent=agent,
            task=task[:80],
            success=result.success,
            output=result.output,
            duration_ms=duration_ms,
        )
        self.steps.append(log)
        self.files_modified.extend([f for f in result.files_modified if f not in self.files_modified])
        status = "[green]✓[/green]" if result.success else "[red]✗[/red]"
        console.print(f"  {status} Step {step_n} [{agent}] {task[:60]}… ({duration_ms}ms)")
        if not result.success:
            console.print(f"    [red]Error: {result.error}[/red]")
        return log

    def _run_agent(self, decision: RoutingDecision, task: str, target_file: str = None) -> AgentResult:
        tier = decision.model_tier
        kwargs = dict(retriever=self.retriever, tier=tier)

        for attempt in range(1, MAX_RETRIES + 1):
            if attempt > 1:
                console.print(f"  [yellow]Retry {attempt}/{MAX_RETRIES}…[/yellow]")

            if decision.agent == "planner":
                result = planner.run(task=task, **kwargs)
            elif decision.agent == "coder":
                result = coder.run(task=task, target_file=target_file, **kwargs)
            elif decision.agent == "debugger":
                result = debugger.run(error=task, traceback="", **kwargs)
            elif decision.agent == "rag_only":
                try:
                    context = self.retriever.retrieve_for_prompt(task, top_k=5)
                except FileNotFoundError:
                    context = "No index available."
                result = AgentResult(success=True, output=context, steps_taken=1)
            else:
                result = AgentResult(success=False, output="", error=f"Unknown agent: {decision.agent}")

            if result.success:
                self._last_full_output = result.output
                return result

        self._last_full_output = result.output
        return result

    def _reflect(self, result: AgentResult) -> bool:
        if not result.success:
            return False
        output = result.output
        if "score:" in output:
            return True
        if "def " in output or "class " in output:
            exec_result = execute_python(f"compile({repr(output)}, '<check>', 'exec')")
            if not exec_result.success:
                console.print(f"  [yellow]⚠ Syntax issue: {exec_result.error[:80]}[/yellow]")
        return True

    def _write_step_output(self, code: str, target_file: str = None, step_desc: str = ""):
        """Show code and write to target_file immediately (no prompt for planner steps)."""
        if not code or not ("def " in code or "class " in code or "import " in code):
            return

        if target_file:
            # Planner sub-step with known target — write directly
            target_path = self.project_path / target_file.lstrip("./")
            target_path.parent.mkdir(parents=True, exist_ok=True)
            target_path.write_text(code)
            console.print(f"  [green]Written:[/green] {target_path}")
            if str(target_path) not in self.files_modified:
                self.files_modified.append(str(target_path))

    def run(self, task: str) -> LoopResult:
        start = time.time()
        console.print(Panel(f"[bold]{task}[/bold]", title="[cyan]nexus[/cyan]", expand=False))

        decision = route(task, mode=self.mode)

        if decision.agent == "planner":
            # Get structured plan first
            t0 = time.time()
            plan_result = self._run_agent(decision, task)
            self._log_step("planner", task, plan_result, int((time.time() - t0) * 1000))

            if not plan_result.success:
                return LoopResult(
                    success=False, final_output="",
                    steps=self.steps, error=plan_result.error,
                    total_duration_ms=int((time.time() - start) * 1000),
                )

            # Parse plan — get raw JSON steps from planner
            import json
            raw_steps = []
            try:
                # planner stores the raw plan in _last_full_output via chat_json
                # re-run to get structured data
                from llm.client import get_client
                from llm.prompts import PLANNER_PROMPT
                client = get_client()
                context = ""
                try:
                    context = self.retriever.retrieve_for_prompt(task, top_k=5)
                except FileNotFoundError:
                    pass
                messages = PLANNER_PROMPT.build(context=context, user=task)
                plan_json = client.chat_json(messages, tier=decision.model_tier, temperature=0.2)
                raw_steps = plan_json.get("steps", [])
            except Exception:
                # Fallback: parse text output line by line
                for line in plan_result.output.split("\n"):
                    line = line.strip()
                    if line and line[0].isdigit():
                        raw_steps.append({"description": line, "target_file": None, "action_type": "code_edit"})

            # Execute each step
            for step_def in raw_steps:
                if len(self.steps) >= MAX_STEPS:
                    console.print("[yellow]Max steps reached.[/yellow]")
                    break

                desc = step_def.get("description", "")
                target = step_def.get("target_file")
                action = step_def.get("action_type", "")

                if not desc:
                    continue

                # Force sub-steps: never re-plan, always code or terminal
                sub_decision = route(desc, mode=self.mode)
                if sub_decision.agent == "planner":
                    from dataclasses import replace
                    sub_decision = replace(sub_decision, agent="coder")

                # Force shell commands to terminal
                # Skip vague "run/test" steps — not actionable
                if any(desc.strip().lower().startswith(w) for w in ("run ", "test ", "start ", "launch ")):
                    console.print(f"  [dim]Skipping non-actionable step: {desc[:50]}[/dim]")
                    continue

                if action == "shell_command":
                    t0 = time.time()
                    cmd_result = run_command(desc)
                    agent_result = AgentResult(
                        success=cmd_result.success,
                        output=cmd_result.output,
                        error=cmd_result.error,
                    )
                    self._log_step("terminal", desc, agent_result, int((time.time() - t0) * 1000))
                    continue

                t0 = time.time()
                sub_result = self._run_agent(sub_decision, desc, target_file=target)
                self._log_step(sub_decision.agent, desc, sub_result, int((time.time() - t0) * 1000))

                # Auto-write if target file specified in plan
                if sub_result.success and target:
                    from agents.coder import _extract_code
                    import os
                    code = _extract_code(sub_result.output)
                    # Sanitize: keep only the filename, no nested dirs
                    flat_target = os.path.basename(target.strip("./"))
                    self._write_step_output(code, flat_target, desc)

                self._reflect(sub_result)

        else:
            # Single agent — offer write confirmation after
            t0 = time.time()
            result = self._run_agent(decision, task)
            self._log_step(decision.agent, task, result, int((time.time() - t0) * 1000))
            self._reflect(result)

        total_ms = int((time.time() - start) * 1000)
        last = self.steps[-1] if self.steps else None
        success = last.success if last else False

        return LoopResult(
            success=success,
            final_output=self._last_full_output,
            steps=self.steps,
            files_modified=list(set(self.files_modified)),
            total_duration_ms=total_ms,
        )
