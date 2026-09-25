"""LLM adapters: Anthropic API, OpenAI / any OpenAI-compatible server, and local Ollama (native API)."""
from __future__ import annotations

import json
import logging
import os
import time
import uuid

import httpx

from .base import LLMClient, LLMResponse, Message, Tool, ToolCall, strip_thinking

log = logging.getLogger(__name__)


# =============================================================================== Anthropic
class AnthropicClient(LLMClient):
    provider = "anthropic"

    def __init__(self, model: str | None, temperature: float = 0.2, max_tokens: int = 2000, api_key: str | None = None):
        import anthropic
        self.temperature, self.max_tokens = temperature, max_tokens
        self._client = anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))
        self.model = model or self._latest_model()

    def _latest_model(self, prefer: str = "sonnet") -> str:
        """No model configured: ask the Models API (newest first) and prefer the mid-size tier."""
        ids = [m.id for m in self._client.models.list(limit=50).data]
        if not ids:
            raise RuntimeError("Anthropic Models API returned no models; set ANTHROPIC_MODEL in .env")
        return next((i for i in ids if prefer in i), ids[0])

    @staticmethod
    def _convert(messages: list[Message]) -> list[dict]:
        out: list[dict] = []

        def push(role: str, blocks: list[dict]):
            if out and out[-1]["role"] == role:          # Anthropic requires alternating roles
                out[-1]["content"].extend(blocks)
            else:
                out.append({"role": role, "content": blocks})

        for m in messages:
            if m.role == "user":
                push("user", [{"type": "text", "text": m.content}])
            elif m.role == "assistant":
                blocks = [{"type": "text", "text": m.content}] if m.content else []
                blocks += [{"type": "tool_use", "id": tc.id, "name": tc.name, "input": tc.arguments} for tc in m.tool_calls]
                push("assistant", blocks or [{"type": "text", "text": "(no content)"}])
            elif m.role == "tool":
                push("user", [{"type": "tool_result", "tool_use_id": m.tool_call_id, "content": m.content}])
        return out

    def chat(self, messages, system=None, tools=None, temperature=None, max_tokens=None,
             json_mode=False) -> LLMResponse:
        kw = {"model": self.model, "max_tokens": max_tokens or self.max_tokens, "messages": self._convert(messages)}
        if system:
            kw["system"] = system
        if tools:
            kw["tools"] = [{"name": t.name, "description": t.description, "input_schema": t.parameters} for t in tools]
        temp = self.temperature if temperature is None else temperature
        t0 = time.time()
        try:
            resp = self._client.messages.create(temperature=temp, **kw)
        except Exception as e:
            if "temperature" in str(e).lower():          # some models reject sampling params
                resp = self._client.messages.create(**kw)
            else:
                raise
        text, calls = [], []
        for b in resp.content:
            if b.type == "text":
                text.append(b.text)
            elif b.type == "tool_use":
                calls.append(ToolCall(b.id, b.name, dict(b.input or {})))
        usage = {"input_tokens": getattr(resp.usage, "input_tokens", None),
                 "output_tokens": getattr(resp.usage, "output_tokens", None)}
        return LLMResponse("\n".join(text), calls, usage, resp.stop_reason, time.time() - t0)


# =============================================================================== OpenAI-compatible
class OpenAICompatClient(LLMClient):
    provider = "openai"

    def __init__(self, model: str, temperature: float = 0.2, max_tokens: int = 2000, base_url: str | None = None,
                 api_key: str | None = None):
        from openai import OpenAI
        self.model, self.temperature, self.max_tokens = model, temperature, max_tokens
        self._client = OpenAI(base_url=base_url or os.environ.get("OPENAI_BASE_URL") or None,
                              api_key=api_key or os.environ.get("OPENAI_API_KEY") or "not-needed")

    @staticmethod
    def _convert(messages: list[Message], system: str | None) -> list[dict]:
        out: list[dict] = [{"role": "system", "content": system}] if system else []
        for m in messages:
            if m.role == "user":
                out.append({"role": "user", "content": m.content})
            elif m.role == "assistant":
                d: dict = {"role": "assistant", "content": m.content or None}
                if m.tool_calls:
                    d["tool_calls"] = [{"id": tc.id, "type": "function",
                                        "function": {"name": tc.name, "arguments": json.dumps(tc.arguments, ensure_ascii=False)}}
                                       for tc in m.tool_calls]
                out.append(d)
            elif m.role == "tool":
                out.append({"role": "tool", "tool_call_id": m.tool_call_id, "content": m.content})
        return out

    def chat(self, messages, system=None, tools=None, temperature=None, max_tokens=None,
             json_mode=False) -> LLMResponse:
        kw: dict = {"model": self.model, "messages": self._convert(messages, system),
                    "temperature": self.temperature if temperature is None else temperature}
        if tools:
            kw["tools"] = [{"type": "function", "function": {"name": t.name, "description": t.description,
                                                              "parameters": t.parameters}} for t in tools]
        if json_mode and not tools:
            kw["response_format"] = {"type": "json_object"}
        mt = max_tokens or self.max_tokens
        t0 = time.time()
        try:
            resp = self._client.chat.completions.create(max_tokens=mt, **kw)
        except Exception as e:  # newer OpenAI models: max_completion_tokens, fixed temperature
            msg = str(e).lower()
            if "max_tokens" in msg or "max_completion_tokens" in msg or "temperature" in msg:
                kw.pop("temperature", None) if "temperature" in msg else None
                resp = self._client.chat.completions.create(max_completion_tokens=mt, **kw)
            else:
                raise
        msg = resp.choices[0].message
        calls = []
        for tc in msg.tool_calls or []:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            calls.append(ToolCall(tc.id or uuid.uuid4().hex[:12], tc.function.name, args))
        usage = {"input_tokens": getattr(resp.usage, "prompt_tokens", None),
                 "output_tokens": getattr(resp.usage, "completion_tokens", None)} if resp.usage else {}
        return LLMResponse(strip_thinking(msg.content or ""), calls, usage, resp.choices[0].finish_reason,
                           time.time() - t0)


# =============================================================================== Ollama
class OllamaClient(LLMClient):
    """Local models on the M2 via Ollama's native /api/chat (tool calling + `think` switch).

    Thinking models (qwen3, qwen3-vl, deepseek-r1 ...) spend thousands of tokens reasoning before answering;
    `think=False` makes them answer directly, which is ~5-10x faster and keeps the JSON outputs intact.
    """
    provider = "ollama"

    def __init__(self, model: str, temperature: float = 0.2, max_tokens: int = 2000, base_url: str | None = None,
                 think: bool = False, num_ctx: int = 16384, timeout: float = 600):
        self.model, self.temperature, self.max_tokens = model, temperature, max_tokens
        self.host = (base_url or os.environ.get("OLLAMA_BASE_URL") or "http://localhost:11434").rstrip("/")
        self.think, self.num_ctx, self.timeout = think, num_ctx, timeout

    def installed_models(self) -> list[str]:
        try:
            r = httpx.get(f"{self.host}/api/tags", timeout=3)
            return [m["name"] for m in r.json().get("models", [])]
        except Exception:
            return []

    @property
    def enabled(self) -> bool:
        names = self.installed_models()
        return any(n == self.model or n.split(":")[0] == self.model for n in names)

    @staticmethod
    def _convert(messages: list[Message], system: str | None) -> list[dict]:
        out: list[dict] = [{"role": "system", "content": system}] if system else []
        for m in messages:
            if m.role == "assistant":
                d: dict = {"role": "assistant", "content": m.content or ""}
                if m.tool_calls:
                    d["tool_calls"] = [{"function": {"name": tc.name, "arguments": tc.arguments}} for tc in m.tool_calls]
                out.append(d)
            elif m.role == "tool":
                out.append({"role": "tool", "content": m.content, "tool_name": m.name or ""})
            else:
                out.append({"role": "user", "content": m.content})
        return out

    def chat(self, messages, system=None, tools=None, temperature=None, max_tokens=None,
             json_mode=False) -> LLMResponse:
        body: dict = {"model": self.model, "messages": self._convert(messages, system), "stream": False,
                      "think": self.think,
                      "options": {"temperature": self.temperature if temperature is None else temperature,
                                  "num_predict": max_tokens or self.max_tokens, "num_ctx": self.num_ctx}}
        if tools:
            body["tools"] = [{"type": "function", "function": {"name": t.name, "description": t.description,
                                                                "parameters": t.parameters}} for t in tools]
        if json_mode and not tools:
            body["format"] = "json"          # grammar-constrained decoding: always valid JSON
        t0 = time.time()
        r = httpx.post(f"{self.host}/api/chat", json=body, timeout=self.timeout)
        if r.status_code == 400 and "think" in r.text.lower():      # model without a thinking switch
            body.pop("think", None)
            r = httpx.post(f"{self.host}/api/chat", json=body, timeout=self.timeout)
        r.raise_for_status()
        data = r.json()
        msg = data.get("message") or {}
        calls = []
        for tc in msg.get("tool_calls") or []:
            fn = tc.get("function") or {}
            args = fn.get("arguments") or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            calls.append(ToolCall(tc.get("id") or uuid.uuid4().hex[:12], fn.get("name", ""), args))
        usage = {"input_tokens": data.get("prompt_eval_count"), "output_tokens": data.get("eval_count")}
        text = msg.get("content") or ""
        if not text.strip() and not calls and not self.think:
            # some thinking-capable models (e.g. qwen3-vl) return the whole answer in `thinking`
            # when thinking is switched off — it is the answer, not a reasoning trace
            text = msg.get("thinking") or ""
        return LLMResponse(strip_thinking(text), calls, usage, data.get("done_reason"), time.time() - t0)
