import json
import sqlite3

import pytest

from fintech_agent.agents import ClientProfile, InvestmentAdvisor
from fintech_agent.data import SyntheticProvider
from fintech_agent.llm import LLMClient, LLMResponse, NullLLM, ToolCall
from fintech_agent.product.audit import AuditLog
from fintech_agent.product.compliance import ComplianceGuard, find_violations, scrub_text

PANEL = ["naive", "drift"]


class AdviceLLM(LLMClient):
    """An LLM that keeps giving advice — research mode must still never show it."""
    provider, model = "mock", "advice"

    def chat(self, messages, system=None, tools=None, **kw):
        last = messages[-1]
        if tools and last.role == "user":
            return LLMResponse("", [ToolCall("t1", "analyze_stock", {"ticker": "2330"})])
        if tools:
            return LLMResponse("台積電訊號偏多。建議買進，停損 950 元，目標價 1200。外資連 3 日買進。")
        if "client_message" in last.content:
            return LLMResponse(json.dumps({"thesis": "技術面轉強，建議分批布局。基本面穩健。",
                                           "risks": ["跌破支撐 980 要停損", "匯率風險"],
                                           "client_message": "訊號偏多，信心中等。可逢低加碼。"}, ensure_ascii=False))
        return LLMResponse(json.dumps({"summary": "均線多頭排列。壓力 1100、支撐 980。建議進場。",
                                       "key_points": ["KD 黃金交叉", "目標價 1200"], "risks": ["停損 950"],
                                       "score_adjustment": 0.0}, ensure_ascii=False))


def test_violation_detection_allows_institutional_facts():
    assert find_violations("外資連 3 日買進，投信賣出 2,000 張") == []
    assert find_violations("內部人近 6 月買進次數多於賣出") == []
    assert find_violations("建議買進，停損 950") and find_violations("停損 950", [1000])
    assert find_violations("The model says buy.") == ["buy"]
    clean, removed = scrub_text("訊號偏多。建議買進。波動偏高。")
    assert clean == "訊號偏多。波動偏高。" and removed == ["建議買進。"]


def test_research_mode_never_shows_advice_even_if_llm_gives_it():
    adv = InvestmentAdvisor(SyntheticProvider(), llm=AdviceLLM(), quant_panel=PANEL, mode="research")
    res = adv.analyze("2330", 5, ClientProfile())
    out = res.brief()
    assert ComplianceGuard("research").check(out) == []
    d = res.decision
    assert d["mode"] == "research" and d["signal"] in {"偏多（強）", "偏多", "中性", "偏空", "偏空（強）"}
    assert not {"action", "stop_loss", "take_profit", "position_pct", "entry_zone"} & set(d)
    assert "匯率風險" in d["risks"]                       # clean items survive
    assert res.compliance["removed_sentences"] > 0          # the user sees how much was filtered …
    assert "removed_text" not in res.compliance             # … never the removed advice itself
    assert "evidence" not in out["agents"]["technical"]
    assert "support" not in res.reports["technical"].evidence


class SneakyLLM(LLMClient):
    """Advice without trigger words: bare price levels, simplified Chinese, structured values, fake headlines."""
    provider, model = "mock", "sneaky"

    def __init__(self, close):
        self.c = close

    def chat(self, messages, system=None, tools=None, **kw):
        c = self.c
        if "client_message" in messages[-1].content:
            return LLMResponse(json.dumps({
                "thesis": {"headlines": [f"強烈建議買進，目標 {c * 1.2:.0f}"]},
                "risks": {f"跌破 {c * 0.95:.0f} 宜出清": 1, "news": ["加仓布局"]},
                "client_message": f"下檔關注 {c * 0.95:,.0f} 元，上看 {c * 1.2:,.0f}。現在**买**進。訊號偏多。"}, ensure_ascii=False))
        return LLMResponse(json.dumps({"summary": f"關鍵價位 {c * 1.1:,.0f}。均線多頭排列。",
                                       "key_points": [f"股價上看 {c * 1.15:,.1f}", "量能放大"], "risks": ["建议止损"],
                                       "score_adjustment": 0.0}, ensure_ascii=False))


def test_research_mode_blocks_price_levels_and_structured_bypass():
    prov = SyntheticProvider()
    close = float(prov.prices(prov.symbol("2330"))["close"].iloc[-1])
    res = InvestmentAdvisor(prov, llm=SneakyLLM(close), quant_panel=PANEL, mode="research").analyze("2330", 5)
    out = res.brief()
    g = ComplianceGuard("research", [close], ["2330"])
    assert g.check(out) == [], g.check(out)
    assert isinstance(res.decision["thesis"], str) and all(isinstance(x, str) for x in res.decision["risks"])
    assert "訊號偏多" in res.decision["client_message"] and "量能放大" in res.reports["technical"].key_points


def test_research_chat_scrubs_and_falls_back():
    adv = InvestmentAdvisor(SyntheticProvider(), llm=AdviceLLM(), quant_panel=PANEL, mode="research")
    answer, trace = adv.chat([], "2330 可以買嗎？", ClientProfile())
    assert trace and trace[0]["tool"] == "analyze_stock"
    body = answer.split("\n\n>")[0]
    assert find_violations(body) == []
    assert "外資連 3 日買進" in body and "訊號偏多" in body


def test_advisor_mode_keeps_recommendations():
    adv = InvestmentAdvisor(SyntheticProvider(), llm=NullLLM(), quant_panel=PANEL, mode="advisor")
    d = adv.analyze("2330", 5, ClientProfile()).decision
    assert {"action", "stop_loss", "take_profit"} <= set(d)


def test_every_analysis_is_audited_and_chain_is_tamper_evident(tmp_path):
    log = AuditLog(path=tmp_path / "a.sqlite")
    adv = InvestmentAdvisor(SyntheticProvider(), llm=NullLLM(), quant_panel=PANEL, mode="research",
                            actor="tenant:demo", audit=log)
    adv.analyze("2330", 5, ClientProfile())
    adv.chat([], "分析 2317", ClientProfile())
    rows = log.rows()
    assert [r["kind"] for r in rows] == ["chat", "analysis", "analysis"] and rows[0]["actor"] == "tenant:demo"
    assert log.verify() == (True, None)
    with sqlite3.connect(log.path) as c:
        c.execute("UPDATE audit SET response = '{}' WHERE id = 2")
    ok, bad = log.verify()
    assert not ok and bad == 2
    assert len(list(log.export())) == 3


def test_unknown_mode_rejected():
    with pytest.raises(ValueError):
        ComplianceGuard("guru")


def test_portfolio_risk_properties():
    from fintech_agent.product.portfolio import portfolio_risk
    p = SyntheticProvider()
    r = portfolio_risk(p, {"2330": 1000, "2317": 2000, "NVDA": 50}, horizon=5)
    rp = r["return_pct"]
    assert rp["p05"] < rp["p10"] < rp["p50"] < rp["p90"]
    assert r["es95_pct"] >= r["var95_pct"] > 0
    assert abs(sum(x["tail_loss_share"] for x in r["positions"]) - 1) < 1e-3
    assert any("美元資產" in f for f in r["flags"])
    single = portfolio_risk(p, {"2330": 1000}, horizon=5)
    assert single["concentration"]["effective_positions"] == 1.0
    assert any("超過 25%" in f for f in single["flags"])
    with pytest.raises(ValueError):
        portfolio_risk(p, {}, horizon=5)


def test_daily_report_is_compliant_and_renders():
    from fintech_agent.product.report import build_report, render_html, render_markdown
    rep = build_report(SyntheticProvider(), "TW", 5, ["2330", "2317"])
    assert ComplianceGuard("research").check({k: v for k, v in rep.items() if k != "compliance"}) == []
    assert "<table>" in render_html(rep) and render_markdown(rep).startswith("# 台股盤後研究報告")
    assert find_violations(render_markdown(rep)) == [] and find_violations(render_html(rep)) == []


def test_scorecard_lines_disclose_costs_honestly():
    from fintech_agent.product.report import render_markdown, scorecard_lines
    rep = {"title": "台股盤後研究報告", "date": "2026-09-30", "horizon": 5, "highlights": [], "watchlist": [],
           "overview": {"index": "^TWII", "last": 1, "ret_1d_pct": 0.1, "ret_5d_pct": 0.2, "ret_20d_pct": 0.3},
           "ranking": None, "disclaimer": "x",
           "scorecard": {"forecasting": None,
                         "ranking": [{"model": "xs-lgbm-chips", "universe": "twse-pit50", "horizon": 5, "ic_mean": 0.03,
                                      "ic_t": 3.4, "ic_positive_share": 0.58, "excess_gross": 0.075, "excess_net": -0.005,
                                      "cost_bps": 58.5}],
                         "live": {"since": "2026-09-24", "pending": 40, "resolved": 0, "models": []}}}
    lines = scorecard_lines(rep)
    assert "-0.5%" in lines[0] and "不足以單獨構成交易策略" in lines[0] and "等待到期" in lines[1]
    md = render_markdown(rep)
    assert "## 模型成績單" in md and find_violations(md) == []


def test_live_scorecard_scores_resolved_predictions():
    import numpy as np
    from fintech_agent.config import get_settings
    from fintech_agent.evaluation.experiments import ExperimentStore
    from fintech_agent.forecasting.base import ForecastResult
    from fintech_agent.product.scorecard import live_scorecard
    s = get_settings()
    st = ExperimentStore(s)
    for m, point in (("lgbm", 101.0), ("naive", 100.0)):
        fc = ForecastResult(m, np.full(5, point), {q: np.full(5, point * (0.94 + q * 0.12)) for q in
                                                    (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)})
        st.log_prediction(fc, "2330", "TW", "2026-01-02", 100.0)
        st.log_prediction(fc, "2330", "TW", "2026-01-02", 100.0)       # duplicate analyses count once
    import pandas as pd
    st.resolve(lambda t: pd.Series([100.5, 101.0, 102.0, 102.5, 103.0], index=pd.bdate_range("2026-01-05", periods=5)))
    card = live_scorecard(s)
    by = {m["model"]: m for m in card["models"]}
    assert by["lgbm"]["n"] == 1 and by["lgbm"]["dir_acc"] == 1.0 and by["lgbm"]["mae_vs_naive_pct"] > 0


def test_tracking_universe_logs_champion_and_naive():
    from fintech_agent.config import get_settings
    from fintech_agent.product.scorecard import live_scorecard, track_universe
    n = track_universe(SyntheticProvider(), ["2330", "2317"], (5,), get_settings())
    assert n == 2                         # no trained champion here -> random walk only, logged once per ticker
    assert live_scorecard(get_settings())["pending"] == 2


def test_scheduler_next_runs_skip_weekends():
    import importlib.util
    from datetime import datetime
    from pathlib import Path
    from zoneinfo import ZoneInfo
    spec = importlib.util.spec_from_file_location("sched", Path(__file__).parents[1] / "scripts" / "scheduler.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    fri_evening = datetime(2026, 10, 2, 20, 0, tzinfo=ZoneInfo("Asia/Taipei"))      # Friday
    runs = dict((m, t) for t, m in mod.next_runs(fri_evening))
    assert runs["TW"].weekday() == 0 and runs["TW"].hour == 15                        # next Monday
    assert runs["US"].weekday() == 5 and runs["US"].hour == 6                         # Saturday morning (Fri US close)


def test_quoted_headlines_pass_through_labelled():
    g = ComplianceGuard("research")
    obj = {"evidence": {"headlines": [{"title": "外資調升目標價至 1500 元，建議買進", "score": 0.8}]},
           "summary": "新聞偏多。建議逢低布局。"}
    clean, rep = g.apply(obj)
    assert clean["evidence"]["headlines"][0]["title"].startswith("外資調升目標價")   # quotation kept verbatim
    assert clean["summary"] == "新聞偏多。"                                          # own words still scrubbed
    assert g.check(clean) == [] and "quote_note" in rep.public() and "removed_text" not in rep.public()


def test_live_scorecard_reports_start_date_before_anything_matures():
    from fintech_agent.data import SyntheticProvider
    from fintech_agent.product.scorecard import live_scorecard, track_universe
    track_universe(SyntheticProvider(), ["2330"], (5,))
    card = live_scorecard()
    assert card["resolved"] == 0 and card["pending"] >= 1 and card["since"]
