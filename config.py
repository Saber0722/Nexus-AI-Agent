from pathlib import Path
# LLM import: from langchain_ollama import OllamaLLM

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
INDEX_DIR = DATA_DIR / "indexes"
MEMORY_DB = DATA_DIR / "memory.db"

INDEX_DIR.mkdir(parents=True, exist_ok=True)

# Model config
ROUTER_MODEL = "qwen2.5:1.5b"
EXECUTOR_MODEL = "qwen2.5:7b"
OLLAMA_BASE_URL = "http://localhost:11434"

# RAG config
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64
TOP_K_RESULTS = 5

# Agent config
MAX_STEPS = 15
TOOL_TIMEOUT = 30
