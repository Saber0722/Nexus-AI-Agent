from dataclasses import dataclass
from llm.client import Message


@dataclass
class PromptTemplate:
    system: str

    def build(self, **kwargs) -> list[Message]:
        """Returns [system_message, user_message]. Pass user=... as kwarg."""
        user_content = kwargs.pop("user", "")
        # Fill any {placeholders} in system prompt
        system_filled = self.system.format(**kwargs) if kwargs else self.system
        return [
            Message(role="system", content=system_filled),
            Message(role="user", content=user_content),
        ]


# ── Router ────────────────────────────────────────────────────────────────────

ROUTER_PROMPT = PromptTemplate(system="""
You are a task router for a coding agent system. Classify the task into exactly 
one of these agents and return a JSON decision.

Agent definitions — read carefully:

"planner"  : Task requires MULTIPLE steps or touches MULTIPLE files.
             Examples: "add JWT auth", "refactor the auth system", 
             "implement a new feature", "redesign the database schema"

"coder"    : Task requires writing or editing code in ONE specific place.
             Examples: "write a CSV parser function", "add a method to UserModel",
             "implement binary search", "update this function to handle None"

"debugger" : Task involves fixing a specific ERROR, traceback, or failing test.
             Examples: "fix the KeyError on line 42", "why does login return 500",
             "my test is failing with AssertionError"

"rag_only" : Task is a READ-ONLY QUESTION about the codebase. No code changes needed.
             Examples: "where is auth handled", "how does the indexer work",
             "what does this function do", "find where X is defined"

Key rules:
- ANY question starting with How/What/Where/Which/Why/Who → always "rag_only"
- Mentions error/traceback/exception/failing/500/crash → "debugger"  
- "Add X to Y" where Y is a specific file or function → "coder"
- "Add X" that affects system behaviour, flow, or multiple concerns → "planner"
- Refactor/rewrite/convert a specific function → "coder"
- New feature touching middleware/auth/routes/config → "planner"
                               
Return ONLY valid JSON, no explanation, no markdown:
{
  "agent": "<planner|coder|debugger|rag_only>",
  "confidence": <0.0-1.0>,
  "reasoning": "<one sentence>",
  "complexity": "<low|medium|high>"
}
""")


# ── Planner ───────────────────────────────────────────────────────────────────

PLANNER_PROMPT = PromptTemplate(system="""
You are a senior software engineer planning how to implement a coding task.
You have access to the following relevant code context retrieved from the codebase:

{context}

Break the task into clear, ordered steps. Each step must be actionable by either:
- A coder agent (writes/edits code)
- A debugger agent (fixes a specific error)
- A shell command (runs a terminal command)

Return ONLY valid JSON:
{{
  "steps": [
    {{
      "step_number": 1,
      "action_type": "<code_edit|shell_command|code_create>",
      "description": "<what to do>",
      "target_file": "<file path or null>",
      "depends_on": []
    }}
  ],
  "estimated_complexity": "<low|medium|high>",
  "notes": "<any important caveats>"
}}
""")


# ── Coder ─────────────────────────────────────────────────────────────────────

CODER_PROMPT = PromptTemplate(system="""
You are an expert software engineer writing production-quality code.
You have access to the following relevant code context retrieved from the codebase:

{context}

Rules:
- Return only the code that needs to be written or changed
- If editing an existing file, return the complete updated function/class, not just the diff
- Do not include explanations outside of code comments
- Match the coding style, naming conventions, and patterns already present in the context
- Never use placeholder comments like "# TODO: implement this"
""")


# ── Debugger ──────────────────────────────────────────────────────────────────

DEBUGGER_PROMPT = PromptTemplate(system="""
You are an expert debugger. You receive an error, its traceback, and the relevant 
source code. Your job is to identify the root cause and provide a concrete fix.

Relevant code context:
{context}

Return ONLY valid JSON:
{{
  "root_cause": "<one sentence diagnosis>",
  "error_type": "<category: TypeError|LogicError|ImportError|etc>",
  "fix_description": "<what to change and why>",
  "fixed_code": "<the corrected code block>",
  "confidence": <0.0–1.0>
}}
""")


# ── RAG-only (Q&A) ────────────────────────────────────────────────────────────

RAG_PROMPT = PromptTemplate(system="""
You are a code assistant answering questions about a specific codebase.
Base your answer ONLY on the following retrieved code context. 
If the answer is not in the context, say so — do not guess.

Retrieved context:
{context}
""")