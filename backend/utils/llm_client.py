"""Provider-agnostic LLM client.

Switch providers by setting LLM_PROVIDER in .env.
All providers expose the same .generate() and .embed() interface so the rest of
the code never has to know which backend is in use.
"""
from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from typing import Optional

from backend.config import get_settings

logger = logging.getLogger(__name__)


class BaseLLMClient(ABC):
    @abstractmethod
    def generate(self, prompt: str, system: str = "", json_mode: bool = False,
                 temperature: float = 0.4, max_tokens: int = 1024) -> str: ...

    @abstractmethod
    def embed(self, text: str) -> list[float]: ...


class OpenAIClient(BaseLLMClient):
    def __init__(self):
        from openai import OpenAI
        s = get_settings()
        if not s.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY missing in .env")
        self.client = OpenAI(api_key=s.openai_api_key)
        self.model = s.openai_model
        self.embed_model = s.embedding_model

    def generate(self, prompt, system="", json_mode=False, temperature=0.4, max_tokens=1024):
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        kwargs = dict(model=self.model, messages=messages,
                      temperature=temperature, max_tokens=max_tokens)
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        resp = self.client.chat.completions.create(**kwargs)
        return resp.choices[0].message.content or ""

    def embed(self, text: str) -> list[float]:
        text = text.replace("\n", " ")
        resp = self.client.embeddings.create(input=[text], model=self.embed_model)
        return resp.data[0].embedding


class AnthropicClient(BaseLLMClient):
    def __init__(self):
        import anthropic
        from sentence_transformers import SentenceTransformer
        s = get_settings()
        if not s.anthropic_api_key:
            raise RuntimeError("ANTHROPIC_API_KEY missing in .env")
        self.client = anthropic.Anthropic(api_key=s.anthropic_api_key)
        self.model = s.anthropic_model
        # Anthropic doesn't ship embeddings; use a local sentence-transformer.
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")

    def generate(self, prompt, system="", json_mode=False, temperature=0.4, max_tokens=1024):
        if json_mode:
            prompt += "\n\nRespond with valid JSON only — no markdown fences, no preamble."
        msg = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system or "You are a helpful assistant.",
            messages=[{"role": "user", "content": prompt}],
        )
        return msg.content[0].text

    def embed(self, text: str) -> list[float]:
        return self.embedder.encode(text).tolist()


class OllamaClient(BaseLLMClient):
    def __init__(self):
        import httpx
        from sentence_transformers import SentenceTransformer
        s = get_settings()
        self.host = s.ollama_host
        self.model = s.ollama_model
        self.client = httpx.Client(timeout=120)
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")

    def generate(self, prompt, system="", json_mode=False, temperature=0.4, max_tokens=1024):
        if json_mode:
            prompt += "\n\nRespond with valid JSON only."
        full = f"{system}\n\n{prompt}" if system else prompt
        r = self.client.post(f"{self.host}/api/generate", json={
            "model": self.model, "prompt": full, "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        })
        r.raise_for_status()
        return r.json().get("response", "")

    def embed(self, text: str) -> list[float]:
        return self.embedder.encode(text).tolist()


def build_llm_client() -> BaseLLMClient:
    s = get_settings()
    provider = s.llm_provider.lower()
    if provider == "openai":
        return OpenAIClient()
    if provider == "anthropic":
        return AnthropicClient()
    if provider == "ollama":
        return OllamaClient()
    raise ValueError(f"Unknown LLM_PROVIDER: {provider}")


_client: Optional[BaseLLMClient] = None


def get_llm() -> BaseLLMClient:
    """Module-level singleton."""
    global _client
    if _client is None:
        _client = build_llm_client()
        logger.info(f"LLM client initialised: {get_settings().llm_provider}")
    return _client


def safe_json_parse(text: str) -> dict | list | None:
    """Tolerant JSON parser — strips markdown fences if present."""
    if not text:
        return None
    t = text.strip()
    if t.startswith("```"):
        t = t.strip("`")
        if t.startswith("json"):
            t = t[4:]
        t = t.strip()
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        # Try to recover the first {...} or [...] block
        for opener, closer in [("{", "}"), ("[", "]")]:
            i, j = t.find(opener), t.rfind(closer)
            if i != -1 and j != -1 and j > i:
                try:
                    return json.loads(t[i:j+1])
                except json.JSONDecodeError:
                    continue
        logger.warning("Failed to parse JSON: %s", text[:200])
        return None
