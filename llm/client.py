import json
import time
from enum import Enum
from dataclasses import dataclass
from typing import Iterator, Optional

import requests
from rich.console import Console

from config import OLLAMA_BASE_URL, ROUTER_MODEL, EXECUTOR_MODEL

console = Console()


class ModelTier(Enum):
    FAST = "fast"        # qwen2.5:1.5b — routing, simple tasks
    QUALITY = "quality"  # qwen2.5:7b   — coding, debugging, planning


@dataclass
class LLMResponse:
    content: str
    model: str
    tier: ModelTier
    duration_ms: int
    prompt_tokens: int
    response_tokens: int

    def __str__(self):
        return self.content


@dataclass
class Message:
    role: str     # "system" | "user" | "assistant"
    content: str

    def to_dict(self) -> dict:
        return {"role": self.role, "content": self.content}


class OllamaClient:
    def __init__(self, base_url: str = OLLAMA_BASE_URL):
        self.base_url = base_url
        self._verify_connection()

    def _verify_connection(self):
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=5)
            r.raise_for_status()
        except Exception as e:
            raise ConnectionError(
                f"Cannot reach Ollama at {self.base_url}. "
                f"Is 'systemctl status ollama' running? Error: {e}"
            )

    def _model_for_tier(self, tier: ModelTier) -> str:
        return ROUTER_MODEL if tier == ModelTier.FAST else EXECUTOR_MODEL

    # ── Non-streaming call ────────────────────────────────────────────────────

    def chat(
        self,
        messages: list[Message],
        tier: ModelTier = ModelTier.QUALITY,
        temperature: float = 0.2,
        max_tokens: int = 2048,
        json_mode: bool = False,
    ) -> LLMResponse:
        model = self._model_for_tier(tier)
        start = time.time()

        payload = {
            "model": model,
            "messages": [m.to_dict() for m in messages],
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }

        if json_mode:
            payload["format"] = "json"

        try:
            r = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=120,
            )
            r.raise_for_status()
        except requests.exceptions.Timeout:
            raise TimeoutError(f"LLM call timed out after 120s (model: {model})")
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Ollama request failed: {e}")

        data = r.json()
        duration_ms = int((time.time() - start) * 1000)

        return LLMResponse(
            content=data["message"]["content"].strip(),
            model=model,
            tier=tier,
            duration_ms=duration_ms,
            prompt_tokens=data.get("prompt_eval_count", 0),
            response_tokens=data.get("eval_count", 0),
        )

    # ── Streaming call ────────────────────────────────────────────────────────

    def stream(
        self,
        messages: list[Message],
        tier: ModelTier = ModelTier.QUALITY,
        temperature: float = 0.2,
        max_tokens: int = 2048,
    ) -> Iterator[str]:
        """
        Yields tokens as they arrive.
        Usage:
            for token in client.stream(messages):
                print(token, end="", flush=True)
        """
        model = self._model_for_tier(tier)

        payload = {
            "model": model,
            "messages": [m.to_dict() for m in messages],
            "stream": True,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }

        try:
            with requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                stream=True,
                timeout=120,
            ) as r:
                r.raise_for_status()
                for line in r.iter_lines():
                    if not line:
                        continue
                    chunk = json.loads(line)
                    token = chunk.get("message", {}).get("content", "")
                    if token:
                        yield token
                    if chunk.get("done"):
                        break
        except Exception as e:
            raise RuntimeError(f"Streaming failed: {e}")

    # ── JSON structured output ────────────────────────────────────────────────

    def chat_json(
        self,
        messages: list[Message],
        tier: ModelTier = ModelTier.FAST,
        temperature: float = 0.1,
    ) -> dict:
        """
        Forces JSON output mode. Returns parsed dict.
        Use for router decisions, structured plans, etc.
        Low temperature (0.1) for deterministic structured outputs.
        """
        response = self.chat(
            messages=messages,
            tier=tier,
            temperature=temperature,
            json_mode=True,
        )
        try:
            return json.loads(response.content)
        except json.JSONDecodeError:
            # Fallback: try to extract JSON from markdown code block
            content = response.content
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            try:
                return json.loads(content)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"LLM returned invalid JSON even in json_mode.\n"
                    f"Raw response: {response.content[:300]}\n"
                    f"Error: {e}"
                )


# Singleton — import this everywhere instead of instantiating per call
_client: Optional[OllamaClient] = None

def get_client() -> OllamaClient:
    global _client
    if _client is None:
        _client = OllamaClient()
    return _client
