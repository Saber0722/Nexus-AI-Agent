<div align="center">

# NEXUS
### Local Multi-Agent Coding System

*A resource-constrained, local-first coding agent with multi-agent routing, RAG-based codebase understanding, and failure-indexed memory*

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Ollama](https://img.shields.io/badge/Ollama-Local_LLM-green)
![FAISS](https://img.shields.io/badge/FAISS-Vector_Search-orange)
![License](https://img.shields.io/badge/License-MIT-purple)

</div>

---

## What is NEXUS?

NEXUS is a **locally-running, multi-agent coding assistant** built to work on consumer hardware without any API keys, cloud subscriptions, or internet connection. You point it at any codebase, and it can answer questions about it, write new code, fix bugs, and plan multi-step features — all running on your own machine.

It is not a wrapper around an existing tool. Every component — the router, the agents, the RAG pipeline, the memory system — is built from scratch with deliberate design decisions aimed at **resource efficiency on constrained hardware**.

---

## Motivation

Most AI coding agents assume API access and unlimited compute. They are designed around models with billions more parameters than what fits on a consumer GPU. NEXUS was built to answer a different question:

> *How capable can a multi-agent coding system be when constrained to 8GB of VRAM, running entirely locally, with no cloud dependency?*

This constraint drives every architectural choice in the system.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    CLI (nexus command)                  │
│         run | ask | index | memory                      │
└─────────────────────────┬───────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│                  CodebaseIndexer                        │
│   AST-aware chunking → nomic embeddings → FAISS index   │
└─────────────────────────┬───────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│                    AgentLoop                            │
│            plan → act → observe → reflect               │
└──────┬──────────────────┬────────────────────┬──────────┘
       │                  │                    │
       ▼                  ▼                    ▼
┌─────────────┐  ┌──────────────────────────────────────┐
│   Router    │  │           Agent Dispatch             │
│ Qwen 1.5B   │  │  Planner | Coder | Debugger | RAG    │
│  ~300ms     │  │         Qwen 7B executor             │
└─────────────┘  └──────────────────────────────────────┘
                          │
              ┌───────────┴────────────┐
              ▼                        ▼
┌─────────────────────┐   ┌───────────────────────────┐
│     Tool Layer      │   │      Memory Layer         │
│  file | terminal |  │   │  SQLite + FAISS hybrid    │
│  python_exec | git  │   │  failure-indexed store    │
└─────────────────────┘   └───────────────────────────┘
```

### Component Breakdown

#### 1. RAG Layer (`rag/`)

The RAG layer is responsible for giving agents **context awareness** about the codebase being worked on.

**Indexer (`rag/indexer.py`):**
- Walks the project directory and collects supported file types (`.py`, `.js`, `.ts`, `.md`, `.json`, `.yaml`)
- For Python files: uses `ast.parse()` to extract every `FunctionDef`, `AsyncFunctionDef`, and `ClassDef` as a standalone chunk — each chunk is always a complete, syntactically valid unit
- For other files: falls back to sliding window chunking (512 tokens, 64-token overlap)
- Each chunk stores: `file_path`, `chunk_type`, `name`, `start_line`, `end_line`, `language`, `mtime`
- Incremental reindexing: compares file modification times, only re-embeds changed files
- Embeddings: `nomic-ai/nomic-embed-text-v1.5` (768-dim, CPU-only to preserve VRAM for LLM)
- Storage: FAISS `IndexFlatIP` (exact cosine similarity search on L2-normalized vectors)

**Why AST chunking over sliding window?**
A sliding window that cuts at arbitrary token boundaries often produces incomplete chunks — half a function, imports without their context, a class definition split across two chunks. AST chunking guarantees every Python chunk is a complete semantic unit. Retrieval precision improves significantly: when you ask "how does authentication work", you retrieve complete functions rather than fragments.

**Why nomic-embed-text-v1.5 over all-MiniLM-L6-v2?**
`all-MiniLM-L6-v2` was trained on natural language. It understands semantic similarity between sentences but doesn't understand that `SUPPORTED_EXTENSIONS` relates to "file extensions" — that's a code identifier, not a sentence. `nomic-embed-text-v1.5` was trained on both text and code. Retrieval scores improved from 0.20 → 0.62 on code-specific queries after switching. The model requires `search_document:` and `search_query:` prefixes to distinguish document embeddings from query embeddings.

---

#### 2. LLM Layer (`llm/`)

**Two-model architecture:**

| Model | Role | Latency | VRAM |
|---|---|---|---|
| `qwen2.5:1.5b` (Q4) | Router, simple classifications | ~300ms | ~1GB |
| `qwen2.5:7b` (Q4) | All code generation and reasoning | ~8-15s | ~4.5GB |

**Why Qwen 2.5?**
Qwen 2.5 has strong code generation performance relative to its parameter count. The 7B quantized model fits in 4.5GB VRAM, leaving headroom for the embedding model on CPU and system memory. The 1.5B model is fast enough for routing decisions where a wrong answer just costs a retry, not a full task failure.

**Why two models instead of one?**
Running the 7B model for every operation — including simple routing classifications — wastes inference time. The router's job (classify a task into 4 categories) is a low-complexity decision that a 1.5B model handles well. Measured speedup: **3.9x faster** for routing decisions. Over a full session with dozens of operations, this compounds significantly.

**Router escalation:**
If the 1.5B router returns confidence < 0.65, it automatically escalates to the 7B model for the routing decision. This preserves accuracy on ambiguous tasks while keeping latency low for clear ones.

**OllamaClient (`llm/client.py`):**
- `chat()` — blocking call, returns `LLMResponse` with token counts and duration
- `stream()` — yields tokens as they arrive (used for `nexus ask` synthesis)
- `chat_json()` — forces JSON output mode, parses and returns dict (used for structured agent outputs)

---

#### 3. Agent Layer (`agents/`)

All agents share the same interface: `run(task, tier, retriever) → AgentResult`

**Planner (`agents/planner.py`):**
Takes a complex task and decomposes it into ordered, actionable steps using `PLANNER_PROMPT`. Returns structured JSON with `step_number`, `action_type`, `description`, and `target_file` for each step. The orchestrator then executes each step individually.

**Coder (`agents/coder.py`):**
Takes a specific coding task (usually a planner sub-step) along with RAG context from the codebase, generates code using `CODER_PROMPT`, strips markdown fences, and optionally writes to a target file. Uses `git_checkpoint()` before any file write.

**Debugger (`agents/debugger.py`):**
Takes an error message and traceback, queries the failure memory for similar past fixes, builds a diagnosis prompt with both RAG context and memory hints, returns structured JSON with `root_cause`, `fix_description`, `fixed_code`, and `confidence`. All interactions are stored in failure memory for future retrieval.

**RAG-only:**
Not a separate agent file — handled inline in the orchestrator. Pure retrieval with no LLM call. Returns in ~25ms. Used for questions about the codebase that don't require code generation.

---

#### 4. Orchestrator (`orchestrator/agent_loop.py`)

The orchestrator implements the core **Plan → Act → Observe → Reflect** loop.

**Single-agent path** (coder, debugger, rag_only):
```
route(task) → _run_agent() → _reflect() → LoopResult
```

**Planner path:**
```
route(task) → planner.run() → [for each step: route(step) → coder.run() → write file]
```

Key design decisions:
- Sub-steps from the planner are **always forced to coder** — never re-planned. This prevents infinite planning loops where the planner generates steps that are vague enough to get re-routed back to the planner.
- Steps starting with "Run", "Test", "Install", "Load" are skipped — these are English instructions the planner sometimes generates, not actionable code tasks.
- `MAX_STEPS = 15` hard limit prevents runaway execution.
- `MAX_RETRIES = 2` per agent call before moving on.
- Every agent result goes through `_reflect()`, which validates Python code compiles before declaring success.

---

#### 5. Tool Layer (`tools/`)

Every tool returns a uniform `ToolResult(output, error, success, duration_ms)`. The `@tool_result` decorator wraps any function and catches all exceptions, ensuring agents always receive structured results rather than crashing on tool failures.

**file_tool.py:** `read_file`, `write_file`, `edit_file` (unified diff), `list_files`. Path guard (`_guard()`) prevents writes outside the project directory or the nexus installation directory.

**terminal_tool.py:** `run_command` with timeout, cwd, stdout/stderr capture. Blocklist prevents destructive commands (`rm -rf /`, `sudo`, `curl | bash`, `dd`, `mkfs`).

**python_exec.py:** Sandboxed `exec()` with stdout capture and full traceback on failure. Used by the reflection step to validate generated code compiles.

**git_tool.py:** `git_status`, `git_diff`, `git_commit`, `git_checkpoint`. The `git_checkpoint()` function auto-commits before any file modification — if the agent produces bad code, `git checkout <hash> -- file.py` restores the last known-good state.

---

#### 6. Memory Layer (`memory/memory_store.py`)

The failure-indexed memory is the novel research contribution of this project.

**Storage:** SQLite (`data/memory.db`) for persistence across sessions.
**Retrieval:** FAISS (`data/indexes/memory.faiss`) for semantic search over error embeddings.

**What gets stored:**
Every time the debugger agent resolves an error, it stores:
- `error_type` (AttributeError, KeyError, etc.)
- `error_message` and `traceback`
- `failed_code` — the code that caused the error
- `fix_applied` — the fix that was used
- `fix_worked` — boolean, updated if the fix later fails
- `ast_fingerprint` — MD5 hash of the AST node type sequence of the failed code

**AST fingerprinting:**
Two functions with different variable names but identical structure produce the same AST fingerprint. This means structurally similar bugs — the same logical error in different contexts — get matched even when their surface text is completely different.

**Hybrid retrieval scoring:**
```
final_score = semantic_similarity(error_message) + (0.15 if ast_fingerprint_matches else 0)
```

When the debugger encounters an error, it retrieves the top-K past fixes filtered to `fix_worked=True`, and injects the best match into the prompt as a hint. The agent improves with every debugging session.

---

## Models

| Model | Size on Disk | VRAM Usage | Purpose |
|---|---|---|---|
| `qwen2.5:1.5b` (Q4_K_M) | ~1.0 GB | ~1.2 GB | Routing, simple classifications |
| `qwen2.5:7b` (Q4_K_M) | ~4.5 GB | ~5.0 GB | Code generation, debugging, planning |
| `nomic-embed-text-v1.5` | ~547 MB | CPU only | Code embeddings for RAG and memory |

**Why Q4 quantization?**
Q4_K_M quantization reduces model size by ~4x vs full precision with minimal quality degradation on code tasks. The 7B model at Q4 fits in 5GB VRAM, leaving 3GB for system processes, the embedding model on CPU, and FAISS indexes in RAM.

---

## Installation

### Prerequisites
- Arch Linux (or any Linux distro)
- Python 3.11+
- NVIDIA GPU with CUDA (8GB VRAM minimum)
- `uv` package manager
- `git`

### Step 1 — Install Ollama with CUDA support
```bash
# Arch Linux (AUR)
yay -S ollama-cuda

# Other distros
curl -fsSL https://ollama.com/install.sh | sh

# Start the service
sudo systemctl enable ollama --now
```

### Step 2 — Pull the models
```bash
ollama pull qwen2.5:7b
ollama pull qwen2.5:1.5b
```

### Step 3 — Clone and install dependencies
```bash
git clone https://github.com/YOUR_USERNAME/nexus
cd nexus
uv venv
source .venv/bin/activate
uv pip install langchain==0.3.25 langchain-community==0.3.23 langchain-ollama \
  langgraph==0.4.1 faiss-cpu==1.10.0 sentence-transformers==4.1.0 \
  fastapi==0.115.12 uvicorn==0.34.2 rich==14.0.0 typer==0.15.2 \
  gitpython==3.1.44 watchdog==6.0.0 sqlalchemy==2.0.40 einops
```

### Step 4 — Add shell function
```bash
echo 'function nexus() { local _cwd="$PWD"; (cd ~/path/to/nexus && NEXUS_CWD="$_cwd" uv run python ~/path/to/nexus/main.py "$@"); }' >> ~/.bashrc
source ~/.bashrc
```

### Step 5 — Verify installation
```bash
cd nexus
uv run python test_phase00.py
```

All 4 checks should show PASS.

---

## Usage

### Index a project
```bash
cd ~/your-project
nexus index
```
Run once. Subsequent runs are incremental — only changed files are re-embedded.

### Ask questions about the codebase
```bash
nexus ask "how does authentication work"
nexus ask "where is the database connection configured"
nexus ask "what does the UserModel class do"

# Show raw retrieved chunks instead of LLM synthesis
nexus ask "what file extensions are supported" --raw
```

### Run a coding task
```bash
# Write a new file
nexus run "write a function to validate email addresses"

# Fix a bug — paste the actual error
nexus run "AttributeError: NoneType has no attribute search in retriever.py line 48"

# Add a feature
nexus run "add rate limiting to the API endpoints"

# Build from scratch in an empty directory
mkdir my-new-project && cd my-new-project
nexus run "Create a Flask REST API with JWT authentication and SQLite database"
```

### Control inference quality
```bash
# Fast mode — always uses 1.5B (~3s per task)
nexus run --mode fast "what does the Config class do"

# Balanced — 1.5B routes, 7B executes (default)
nexus run "add error handling to the fetch function"

# Quality — always uses 7B for routing and execution
nexus run --mode quality "refactor the authentication module"
```

### Check memory store
```bash
nexus memory
```
Shows total records, successful fixes, hit rate, and error type distribution.

---

## Current Capabilities

| Task Type | Capability | Notes |
|---|---|---|
| Code generation | ✅ Strong | Single-file scripts, functions, classes |
| Multi-file project creation | ✅ Working | Planner decomposes, coder writes each file |
| Bug diagnosis | ✅ Working | With memory hints for recurring errors |
| Codebase Q&A | ✅ Strong | RAG retrieval + LLM synthesis |
| Incremental indexing | ✅ Strong | Only re-embeds changed files |
| Git safety checkpoints | ✅ Working | Auto-commits before every file write |
| Memory accumulation | ✅ Working | Improves with every debugging session |
| Python AST chunking | ✅ Strong | Complete function/class chunks |
| Multi-language support | ⚠️ Partial | JS/TS/MD via sliding window |
| Test generation | ⚠️ Partial | Works but sometimes generates wrong framework |
| Cross-file refactoring | ⚠️ Partial | Planner handles it, quality varies |
| Running/executing code | ❌ Limited | Sandboxed exec only, no interactive sessions |

---

## Evaluation Results

Benchmarked on 20 tasks across 5 categories:

| Category | Tasks | Routing Accuracy | Task Success |
|---|---|---|---|
| Bug fix | 5 | 100% | 80% |
| Feature (single file) | 5 | 100% | 100% |
| Refactor | 2 | 100% | 100% |
| RAG query | 5 | 20% | 100% |
| Multi-step feature | 3 | 0% | 100% |
| **Overall** | **20** | **70%** | **90%** |

**Speed benchmark:**
- `qwen2.5:1.5b`: 2.7s average, ~465 tokens
- `qwen2.5:7b`: 10.4s average, ~526 tokens
- Speedup: **3.9x faster** with 1.5B router

**Memory retrieval scores:**
- Seen error pattern (AttributeError): 0.875
- Seen error pattern (KeyError): 0.841
- Unseen error variant: 0.764

---

## Known Limitations

**Routing accuracy on questions and multi-step tasks:**
The 1.5B router conflates "how/where/what" questions with code tasks, routing them to the debugger instead of rag_only. Tasks still succeed (the debugger handles questions reasonably) but at higher latency than necessary. This is a fundamental limitation of the 1.5B model's reasoning capacity, not a prompt issue.

**RAG context pollution:**
When the project index contains unrelated files (e.g., a PyTorch CNN file in the same directory as a Flask API project), the coder agent may retrieve irrelevant context and generate code mixing the two frameworks. Mitigated by using explicit, detailed prompts. A relevance threshold filter (score > 0.55) is a planned fix.

**No streaming agent output:**
The coder agent generates the complete response before displaying it. For complex tasks, this means 15-30 seconds of silence before output appears. Streaming coder output is on the roadmap.

**Code quality on complex tasks:**
The 7B model occasionally produces incorrect code on complex tasks — wrong imports, mixed frameworks, incomplete implementations. The debugger memory system catches recurring patterns over time, but first-occurrence errors require human review.

**Single-file output per step:**
Each planner sub-step writes one file. Tasks requiring coordinated changes across multiple files in a single step (e.g., updating an import in one file while adding a function in another) require manual coordination.

**No execution environment:**
NEXUS can write and validate Python syntax but cannot run the generated code and observe runtime behavior. There is no feedback loop from execution output back to the agent.

---

## Hardware Requirements

| Component | Minimum | Tested On |
|---|---|---|
| GPU VRAM | 8GB | RTX 4060 8GB |
| RAM | 16GB | 16GB DDR5 |
| Storage | 20GB free | 1TB SSD |
| OS | Linux | Arch Linux |
| CUDA | 11.8+ | CUDA 13.2 |

**VRAM allocation during operation:**
- `qwen2.5:7b` loaded: ~5.0 GB
- System (Wayland, VS Code, terminal): ~0.3 GB
- Available headroom: ~2.7 GB
- Embedding model: CPU (intentional — preserves VRAM for LLM)

**Note:** 6GB VRAM GPUs (RTX 3060, 4060 non-Ti) may work with `qwen2.5:7b` at Q4 but will have very little headroom. The 1.5B model runs comfortably on 4GB VRAM.

---

## Project Structure

```
nexus/
├── main.py                    ← CLI entry point
├── config.py                  ← constants, model names, paths
│
├── llm/
│   ├── client.py              ← OllamaClient: chat/stream/json modes
│   ├── prompts.py             ← PromptTemplate for each agent role
│   └── router.py              ← route() → RoutingDecision
│
├── rag/
│   ├── indexer.py             ← AST chunker + FAISS builder
│   └── retriever.py           ← semantic search interface
│
├── agents/
│   ├── base.py                ← AgentResult dataclass
│   ├── planner.py             ← task decomposition
│   ├── coder.py               ← code generation
│   └── debugger.py            ← error diagnosis + memory store
│
├── tools/
│   ├── base.py                ← ToolResult + @tool_result decorator
│   ├── file_tool.py           ← read/write/edit with path guard
│   ├── terminal_tool.py       ← run_command with blocklist
│   ├── python_exec.py         ← sandboxed execution
│   └── git_tool.py            ← checkpoint/commit/diff
│
├── orchestrator/
│   └── agent_loop.py          ← main execution loop
│
├── memory/
│   └── memory_store.py        ← FailureMemory: SQLite + FAISS
│
├── evals/
│   ├── eval_suite.py          ← 20-task benchmark
│   └── results.json           ← raw results
│
└── data/
    ├── indexes/               ← per-project FAISS indexes
    └── memory.db              ← SQLite failure store
```

---

## How NEXUS Learns

NEXUS does not update model weights — the Qwen models are frozen. Learning happens at the **retrieval layer**:

Every time the debugger fixes a bug:
1. The error type, message, traceback, failed code, and fix are stored in SQLite
2. An embedding of the error is added to the FAISS memory index
3. An AST fingerprint of the failed code is computed and stored

Next time a similar error occurs:
1. The error message is embedded and searched against the memory index
2. Results are ranked by semantic similarity + AST structural match bonus
3. The top match (if `fix_worked=True`) is injected into the debugger's prompt as a hint
4. The agent uses this prior knowledge to fix the error faster and more accurately

After months of daily use, NEXUS will have seen your common error patterns — your usual import mistakes, your recurring API misuse, your project-specific bugs — and will retrieve relevant past fixes automatically.

---

## Roadmap

**Near-term:**
- [ ] Relevance threshold filter for RAG (score > 0.55 only)
- [ ] Streaming coder output to terminal in real time
- [ ] `nexus diff` — show what changed vs git HEAD after an agent run
- [ ] Better filename inference (regex over LLM for speed)

**Research extensions:**
- [ ] Quantization benchmarks: Q4 vs Q5 vs Q8 on agent task quality
- [ ] Structured attention head pruning on the coder agent
- [ ] Fine-tuned router classifier (replace prompt-based routing with a small trained classifier)
- [ ] Multi-turn conversation: maintain context across multiple `nexus run` calls in a session

**Scaling:**
- [ ] Support for `qwen2.5:14b` on systems with 16GB VRAM
- [ ] FAISS `IndexIVFFlat` for codebases with 100k+ chunks
- [ ] Web UI dashboard for session monitoring

---

## Research Framing

If you're using this project for academic work, the key claims are:

1. A 1.5B router model achieves **3.9x latency reduction** vs always using 7B, with 70% routing accuracy on a 20-task benchmark
2. **AST-aware chunking** produces higher retrieval precision than sliding-window for Python codebases (verified by comparing retrieval scores before/after the switch)
3. **Failure-indexed memory with AST fingerprinting** retrieves relevant past fixes with semantic similarity scores of 0.74–0.875 on a 3-record test store
4. The system achieves **90% task success rate** on a 20-task benchmark running entirely on consumer hardware (RTX 4060, 8GB VRAM)

For ablation studies, compare:
- With vs without router (always 7B): measure latency increase
- With vs without memory (no hints to debugger): measure steps-to-fix on repeated errors
- AST chunking vs sliding window only: measure retrieval score distribution

---

## Contributing

This is a personal research project but contributions are welcome. The most useful areas:

- Additional language support (better JS/TS/Go chunking)
- Router accuracy improvements (a trained classifier vs prompt-based)
- Eval suite expansion (more task categories, automated correctness checking)
- Documentation and usage examples

---

## License

MIT License. Use it, fork it, build on it.