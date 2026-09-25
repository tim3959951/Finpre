import json

import pytest

from fintech_agent.agents import ClientProfile, InvestmentAdvisor
from fintech_agent.data import SyntheticProvider
from fintech_agent.llm import LLMClient, LLMResponse, Message, NullLLM, Tool, ToolCall, extract_json, run_tool_loop
from fintech_agent.llm.providers import AnthropicClient, OpenAICompatClient

PANEL = ["naive", "drift"]


class ScriptedLLM(LLMClient):
    """Returns canned replies; the first chat turn issues a tool call."""
    provider, model = "mock", "scripted"

    def __init__(self):
        self.calls = 0

    def chat(self, messages, system=None, tools=None, temperature=None, max_tokens=None):
        self.calls += 1
        last = messages[-1]
        if tools and last.role == "user":
            return LLMResponse("", [ToolCall("t1", "get_quote", {"ticker": "2330"})])
        if tools and last.role == "tool":
            return LLMResponse(f"報價結果：{last.content[:40]}")
        if "最終決策" in last.content or "只輸出 JSON" in last.content and "action" in last.content:
            return LLMResponse(json.dumps({"action": "觀望", "conviction": 55, "position_pct": 99, "thesis": "測試",
                                           "client_message": "hello"}, ensure_ascii=False))
        return LLMResponse('```json\n{"summary": "LLM 摘要", "key_points": ["a"], "risks": ["r"], '
                           '"score_adjustment": 0.9, "adjustment_reason": "x"}\n```')


def test_extract_json_variants():
    assert extract_json('blah ```json\n{"a": 1}\n``` tail') == {"a": 1}
    assert extract_json('<think>{"no": 1}</think> answer {"b": [1, 2,]}') == {"b": [1, 2]}
    assert extract_json("no json here") is None


def test_rule_only_pipeline():
    adv = InvestmentAdvisor(SyntheticProvider(), llm=NullLLM(), quant_panel=PANEL)
    res = adv.analyze("2330", 5, ClientProfile(risk="保守"))
    assert set(res.reports) == {"technical", "fundamental", "quant"}
    d = res.decision
    assert d["action"] in {"買進", "分批買進", "持有", "觀望", "減碼", "賣出", "避開"}
    assert 0 <= d["position_pct"] <= 5.0
    assert d["stop_loss"] < res.ctx.close
    json.dumps(res.brief(), ensure_ascii=False, default=str)


def test_llm_adjustment_is_clipped_and_position_capped():
    llm = ScriptedLLM()
    adv = InvestmentAdvisor(SyntheticProvider(), llm=llm, quant_panel=PANEL)
    res = adv.analyze("AAPL", 5, ClientProfile(risk="穩健"))
    for r in res.reports.values():
        assert r.summary == "LLM 摘要"
        assert abs(r.score - r.base_score) <= 0.5 + 1e-9
    assert res.decision["position_pct"] <= 10.0          # LLM asked for 99%, capped by risk profile
    assert res.decision["action"] == "觀望"


def test_tool_loop_and_chat():
    adv = InvestmentAdvisor(SyntheticProvider(), llm=ScriptedLLM(), quant_panel=PANEL)
    history: list[Message] = []
    answer, trace = adv.chat(history, "2330 報價?", ClientProfile())
    assert trace and trace[0]["tool"] == "get_quote" and trace[0]["ok"]
    assert "報價結果" in answer
    roles = [m.role for m in history]
    assert roles == ["user", "assistant", "tool", "assistant"]


def test_rule_chat_parses_ticker():
    adv = InvestmentAdvisor(SyntheticProvider(), llm=NullLLM(), quant_panel=PANEL)
    out, _ = adv.chat([], "幫我分析2330 20日", ClientProfile())
    assert "2330" in out and "建議" in out


def test_anthropic_message_conversion_merges_tool_results():
    msgs = [Message("user", "hi"), Message("assistant", "", [ToolCall("a", "f", {"x": 1}), ToolCall("b", "g", {})]),
            Message("tool", "r1", tool_call_id="a"), Message("tool", "r2", tool_call_id="b"), Message("user", "go")]
    out = AnthropicClient._convert(msgs)
    assert [m["role"] for m in out] == ["user", "assistant", "user"]
    assert [b["type"] for b in out[2]["content"]] == ["tool_result", "tool_result", "text"]


def test_openai_message_conversion():
    msgs = [Message("user", "hi"), Message("assistant", "", [ToolCall("a", "f", {"x": 1})]),
            Message("tool", "r1", tool_call_id="a")]
    out = OpenAICompatClient._convert(msgs, "sys")
    assert out[0]["role"] == "system" and out[2]["tool_calls"][0]["function"]["arguments"] == '{"x": 1}'
    assert out[3] == {"role": "tool", "tool_call_id": "a", "content": "r1"}


def test_tool_errors_are_returned_not_raised():
    class Caller(LLMClient):
        provider, model = "mock", "m"
        n = 0

        def chat(self, messages, system=None, tools=None, **kw):
            self.n += 1
            if self.n == 1:
                return LLMResponse("", [ToolCall("1", "boom", {})])
            return LLMResponse("done")

    def boom():
        raise RuntimeError("bad")

    ans, trace = run_tool_loop(Caller(), [Message("user", "x")], "", [Tool("boom", "", {"type": "object"}, boom)])
    assert ans == "done" and trace[0]["ok"] is False


def test_ollama_native_message_conversion():
    from fintech_agent.llm.providers import OllamaClient
    msgs = [Message("user", "hi"), Message("assistant", "", [ToolCall("a", "f", {"x": 1})]),
            Message("tool", "r1", tool_call_id="a", name="f")]
    out = OllamaClient._convert(msgs, "sys")
    assert out[0] == {"role": "system", "content": "sys"}
    assert out[2]["tool_calls"][0]["function"] == {"name": "f", "arguments": {"x": 1}}
    assert out[3] == {"role": "tool", "content": "r1", "tool_name": "f"}


def test_unparseable_llm_output_is_labelled_rule_only():
    class Garbage(LLMClient):
        provider, model = "mock", "garbage"

        def chat(self, messages, system=None, tools=None, **kw):
            return LLMResponse("I cannot comply")

    adv = InvestmentAdvisor(SyntheticProvider(), llm=Garbage(), quant_panel=PANEL)
    res = adv.analyze("2330", 5, ClientProfile())
    assert all(r.llm.startswith("rule-only") for r in res.reports.values())
    assert res.decision["advisor_llm"].startswith("rule-only")
