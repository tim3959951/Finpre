"""Provider-neutral chat + tool-calling interface.

Agents only talk to `LLMClient`; Anthropic / OpenAI(-compatible) / Ollama adapters translate the neutral
message format. `NullLLM` lets the whole system run in rule-only mode without any model.
"""
from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable

log = logging.getLogger(__name__)


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict


@dataclass
class Message:
    role: str                                  # user | assistant | tool
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None
    name: str | None = None                    # tool name for role=tool


@dataclass
class LLMResponse:
    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: dict = field(default_factory=dict)
    stop_reason: str | None = None
    latency_s: float = 0.0


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict                           # JSON schema (type=object)
    fn: Callable[..., Any]


class LLMClient:
    provider: str = "base"
    model: str = ""
    temperature: float = 0.2
    max_tokens: int = 2000

    def chat(self, messages: list[Message], system: str | None = None, tools: list[Tool] | None = None,
             temperature: float | None = None, max_tokens: int | None = None,
             json_mode: bool = False) -> LLMResponse:  # pragma: no cover
        """json_mode=True asks the backend to constrain the reply to a single JSON object (when supported)."""
        raise NotImplementedError

    @property
    def enabled(self) -> bool:
        return True

    def complete(self, prompt: str, system: str | None = None, **kw) -> str:
        return self.chat([Message("user", prompt)], system=system, **kw).text

    def __repr__(self) -> str:  # pragma: no cover
        return f"<{self.provider}:{self.model}>"


class NullLLM(LLMClient):
    """No model: agents fall back to deterministic template narratives."""
    provider, model = "none", "rule-only"

    def __init__(self, reason: str = "no LLM configured"):
        self.reason = reason

    @property
    def enabled(self) -> bool:
        return False

    def chat(self, messages, system=None, tools=None, temperature=None, max_tokens=None, json_mode=False) -> LLMResponse:
        return LLMResponse(text="", stop_reason="null")


_THINK = re.compile(r"<think>.*?</think>", re.S)


def strip_thinking(text: str) -> str:
    return _THINK.sub("", text or "").strip()


def extract_json(text: str) -> dict | None:
    """Pull the first JSON object out of a model reply (```json fences or bare braces)."""
    if not text:
        return None
    text = strip_thinking(text)
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    candidates = [m.group(1)] if m else []
    start = text.find("{")
    while start != -1:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                esc = (ch == "\\") and not esc
                if ch == '"' and not esc:
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidates.append(text[start:i + 1])
                    break
        start = text.find("{", start + 1) if not candidates else -1
    for c in candidates:
        try:
            return json.loads(c)
        except json.JSONDecodeError:
            try:
                return json.loads(re.sub(r",\s*([}\]])", r"\1", c))   # trailing commas
            except json.JSONDecodeError:
                continue
    return None


def run_tool_loop(client: LLMClient, messages: list[Message], system: str, tools: list[Tool],
                  max_steps: int = 6, on_event: Callable[[str, dict], None] | None = None) -> tuple[str, list[dict]]:
    """Let the model call tools until it produces a final answer. Returns (answer, trace)."""
    by_name = {t.name: t for t in tools}
    trace: list[dict] = []
    for _ in range(max_steps):
        resp = client.chat(messages, system=system, tools=tools)
        messages.append(Message("assistant", resp.text, resp.tool_calls))
        if not resp.tool_calls:
            return strip_thinking(resp.text), trace
        for tc in resp.tool_calls:
            if on_event:
                on_event("tool_call", {"name": tc.name, "arguments": tc.arguments})
            t0 = time.time()
            try:
                tool = by_name[tc.name]
                result = tool.fn(**(tc.arguments or {}))
                ok = True
            except Exception as e:  # tool errors go back to the model, never crash the chat
                log.exception("tool %s failed", tc.name)
                result, ok = {"error": f"{type(e).__name__}: {e}"}, False
            payload = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False, default=str)
            trace.append({"tool": tc.name, "arguments": tc.arguments, "ok": ok, "seconds": round(time.time() - t0, 2)})
            if on_event:
                on_event("tool_result", {"name": tc.name, "ok": ok})
            messages.append(Message("tool", payload[:15000], tool_call_id=tc.id, name=tc.name))
    resp = client.chat(messages + [Message("user", "請根據以上工具結果直接給出最終回答。")], system=system)
    messages.append(Message("assistant", resp.text))
    return strip_thinking(resp.text), trace
