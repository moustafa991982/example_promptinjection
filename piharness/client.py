"""
Backends for local LLaMA serving stacks.

openai   -> llama.cpp llama-server, vLLM, LM Studio, TGI, text-generation-webui,
            Ollama's compat shim (http://localhost:11434/v1)
ollama   -> Ollama native /api/chat, which exposes num_predict / num_ctx /
            keep_alive and reports token counts more reliably than the shim

Both return a normalized Completion so the runner does not care which is used.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import httpx


@dataclass
class Completion:
    text: str
    latency_s: float
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    finish_reason: Optional[str] = None
    error: Optional[str] = None
    attempts: int = 1
    raw: Dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.error is None


class LLMClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        backend: str = "openai",
        api_key: str = "not-needed",
        temperature: float = 0.0,
        max_tokens: int = 512,
        top_p: float = 1.0,
        seed: Optional[int] = None,
        timeout: float = 180.0,
        max_retries: int = 3,
        num_ctx: Optional[int] = None,
        keep_alive: str = "10m",
        extra_body: Optional[Dict[str, Any]] = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.backend = backend
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.seed = seed
        self.timeout = timeout
        self.max_retries = max_retries
        self.num_ctx = num_ctx
        self.keep_alive = keep_alive
        self.extra_body = extra_body or {}
        self._client: Optional[httpx.AsyncClient] = None

    async def __aenter__(self):
        # Local inference is slow and serialized on the GPU. A generous read
        # timeout beats spurious retries that just queue more work.
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(self.timeout, connect=10.0),
            limits=httpx.Limits(max_connections=64, max_keepalive_connections=32),
        )
        return self

    async def __aexit__(self, *exc):
        if self._client:
            await self._client.aclose()

    # ------------------------------------------------------------------
    def _endpoint(self) -> str:
        if self.backend == "ollama":
            return f"{self.base_url}/api/chat"
        return f"{self.base_url}/chat/completions"

    def _payload(self, messages: List[Dict[str, str]], temperature: Optional[float], seed: Optional[int]) -> Dict[str, Any]:
        temp = self.temperature if temperature is None else temperature
        sd = self.seed if seed is None else seed

        if self.backend == "ollama":
            options: Dict[str, Any] = {
                "temperature": temp,
                "num_predict": self.max_tokens,
                "top_p": self.top_p,
            }
            if sd is not None:
                options["seed"] = sd
            if self.num_ctx:
                options["num_ctx"] = self.num_ctx
            body = {
                "model": self.model,
                "messages": messages,
                "stream": False,
                "keep_alive": self.keep_alive,
                "options": options,
            }
        else:
            body = {
                "model": self.model,
                "messages": messages,
                "temperature": temp,
                "max_tokens": self.max_tokens,
                "top_p": self.top_p,
                "stream": False,
            }
            if sd is not None:
                body["seed"] = sd
        body.update(self.extra_body)
        return body

    def _headers(self) -> Dict[str, str]:
        h = {"Content-Type": "application/json"}
        if self.backend != "ollama":
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    @staticmethod
    def _parse(backend: str, data: Dict[str, Any]) -> Dict[str, Any]:
        if backend == "ollama":
            return {
                "text": (data.get("message") or {}).get("content", "") or "",
                "prompt_tokens": data.get("prompt_eval_count"),
                "completion_tokens": data.get("eval_count"),
                "finish_reason": data.get("done_reason"),
            }
        choice = (data.get("choices") or [{}])[0]
        msg = choice.get("message") or {}
        usage = data.get("usage") or {}
        text = msg.get("content")
        if text is None:
            text = choice.get("text", "") or ""
        return {
            "text": text,
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "finish_reason": choice.get("finish_reason"),
        }

    # ------------------------------------------------------------------
    async def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> Completion:
        assert self._client is not None, "use LLMClient as an async context manager"

        payload = self._payload(messages, temperature, seed)
        url = self._endpoint()
        last_err = None
        started = time.perf_counter()

        for attempt in range(1, self.max_retries + 1):
            try:
                r = await self._client.post(url, json=payload, headers=self._headers())
                if r.status_code >= 500 or r.status_code == 429:
                    last_err = f"HTTP {r.status_code}: {r.text[:200]}"
                    await asyncio.sleep(min(2 ** attempt, 20))
                    continue
                if r.status_code >= 400:
                    # 4xx other than rate limiting is a config bug (bad model
                    # name, wrong path). Retrying just wastes wall clock.
                    return Completion(
                        text="", latency_s=time.perf_counter() - started,
                        error=f"HTTP {r.status_code}: {r.text[:300]}", attempts=attempt,
                    )
                data = r.json()
                parsed = self._parse(self.backend, data)
                return Completion(
                    text=parsed["text"],
                    latency_s=time.perf_counter() - started,
                    prompt_tokens=parsed["prompt_tokens"],
                    completion_tokens=parsed["completion_tokens"],
                    finish_reason=parsed["finish_reason"],
                    attempts=attempt,
                )
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last_err = f"{type(exc).__name__}: {exc}"
                await asyncio.sleep(min(2 ** attempt, 20))
            except Exception as exc:  # noqa: BLE001
                last_err = f"{type(exc).__name__}: {exc}"
                break

        return Completion(
            text="", latency_s=time.perf_counter() - started,
            error=last_err or "unknown error", attempts=self.max_retries,
        )

    async def healthcheck(self) -> Completion:
        return await self.chat(
            [{"role": "user", "content": "Reply with the single word: pong"}],
            temperature=0.0,
        )
