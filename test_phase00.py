import requests
from rich.console import Console
from rich.table import Table

console = Console()

def test_ollama():
    try:
        r = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": "qwen2.5:1.5b", "prompt": "reply with the single word: ready", "stream": False},
            timeout=30
        )
        resp = r.json()["response"].strip().lower()
        return "ready" in resp, resp
    except Exception as e:
        return False, str(e)

def test_faiss():
    try:
        import faiss
        import numpy as np
        idx = faiss.IndexFlatL2(4)
        idx.add(np.random.rand(5, 4).astype("float32"))
        _, I = idx.search(np.random.rand(1, 4).astype("float32"), 2)
        return True, f"index size: {idx.ntotal}"
    except Exception as e:
        return False, str(e)

def test_sentence_transformers():
    try:
        from sentence_transformers import SentenceTransformer
        m = SentenceTransformer("all-MiniLM-L6-v2")
        emb = m.encode(["hello world"])
        return True, f"embedding dim: {emb.shape[1]}"
    except Exception as e:
        return False, str(e)

def test_langchain():
    try:
        from langchain_ollama import OllamaLLM
        llm = OllamaLLM(model="qwen2.5:1.5b")
        return True, "langchain-ollama ok"
    except Exception as e:
        return False, str(e)

table = Table(title="Phase 00 — Stack Verification")
table.add_column("Component", style="cyan")
table.add_column("Status")
table.add_column("Detail", style="dim")

tests = [
    ("Ollama + qwen2.5:1.5b", test_ollama),
    ("FAISS", test_faiss),
    ("SentenceTransformers", test_sentence_transformers),
    ("LangChain", test_langchain),
]

all_pass = True
for name, fn in tests:
    ok, detail = fn()
    status = "[green]PASS[/green]" if ok else "[red]FAIL[/red]"
    if not ok:
        all_pass = False
    table.add_row(name, status, detail)

console.print(table)
if all_pass:
    console.print("\n[green]Phase 00 complete. All systems go.[/green]")
else:
    console.print("\n[red]Fix the failures above before proceeding to Phase 01.[/red]")
