"""Regression tests for the QA review: compliance leaks, metering, audit integrity, input validation."""
import json
import sqlite3
import threading

import pandas as pd
import pytest

from fintech_agent.agents import ClientProfile, InvestmentAdvisor
from fintech_agent.data import SyntheticProvider
from fintech_agent.llm import LLMClient, NullLLM
from fintech_agent.product.audit import AuditLog
from fintech_agent.product.compliance import ComplianceGuard, find_violations
from fintech_agent.product.plans import QuotaExceeded, TenantStore
from fintech_agent.product.portfolio import portfolio_risk

PANEL = ["naive", "drift"]


class StrictProvider(SyntheticProvider):
    """Synthetic data, but unknown tickers have no prices (like the live provider)."""
    KNOWN = {"2330", "2317", "2454", "NVDA", "AAPL", "TWD=X", "^TWII", "^GSPC"}

    def prices(self, sym, years=None):
        if sym.code not in self.KNOWN and sym.raw not in self.KNOWN:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        return super().prices(sym, years)


# ------------------------------------------------------------------------------------------ compliance
@pytest.mark.parametrize("text", [
    "建议买进", "可以 買 進", "現在**買**進", "&#36023;&#36914;", "ＢＵＹ", "目 標 價 1200", "go long here", "TP1 1200, SL 950",
    "strong_buy", "建議觀望", "防守價 950", "可在 980 附近低接", "上看 1200", "跟著外資買進", "外資買進你也買進", "散戶就該買進",
    "下檔關注 950 元",
])
def test_bypass_attempts_are_caught(text):
    assert find_violations(text, [1000])


@pytest.mark.parametrize("text", [
    "壓力測試顯示資本適足", "成本壓力上升", "政策支撐內需", "公司海外布局加速", "外資連 3 日買進", "投信加碼", "外資逢低承接",
    "融券放空張數增加", "外資持股比例20日增 0.5 個百分點", "Best Buy 公布財報", "EPS 12.5 元", "營收 1,234 億元",
    "RSI 65、KD 80", "台積電（2330）近 5 日 +0.6%", "sell-off 擴大", "收盤 1,000 元",
])
def test_ordinary_research_text_is_kept(text):
    assert find_violations(text, [1000]) == []


def test_dict_keys_and_quotes_outside_evidence_are_scrubbed():
    g = ComplianceGuard("research")
    clean, rep = g.apply({"risks": {"目標價 1200 元，建議買進": 1}, "thesis": {"headlines": ["加碼布局"]},
                          "evidence": {"headlines": [{"title": "外資調升評等", "date": "2026-09-01", "evil": "建議買進"}]}})
    assert clean["risks"] == {} and clean["thesis"]["headlines"] == []
    assert clean["evidence"]["headlines"] == [{"title": "外資調升評等", "date": "2026-09-01"}]
    assert g.check(clean) == [] and rep.public()["removed_sentences"] >= 2


class BrokenLLM(LLMClient):
    provider, model = "mock", "broken"

    def chat(self, *a, **k):
        raise TimeoutError("upstream timeout")


def test_chat_survives_llm_errors_and_is_audited(tmp_path):
    log = AuditLog(path=tmp_path / "a.sqlite")
    adv = InvestmentAdvisor(SyntheticProvider(), llm=BrokenLLM(), quant_panel=PANEL, mode="research", audit=log)
    answer, _ = adv.chat([], "2330 怎麼看？", ClientProfile())
    assert "沒有完成" in answer and log.rows()[0]["kind"] == "chat"


def test_rule_chat_picks_the_real_ticker():
    adv = InvestmentAdvisor(StrictProvider(), llm=NullLLM(), quant_panel=PANEL, mode="research", audit=False)
    answer, _ = adv.chat([], "IS NVDA GOOD?", ClientProfile())
    assert "NVDA" in answer
    answer, _ = adv.chat([], "ZZZZ 呢?", ClientProfile())
    assert "找不到" in answer


def test_fallback_message_matches_the_decision():
    class Scrubbed(LLMClient):
        provider, model = "mock", "scrubbed"

        def chat(self, messages, **k):
            from fintech_agent.llm import LLMResponse
            if "client_message" in messages[-1].content:
                return LLMResponse(json.dumps({"client_message": "建議買進。"}, ensure_ascii=False))
            return LLMResponse("{}")
    res = InvestmentAdvisor(SyntheticProvider(), llm=Scrubbed(), quant_panel=PANEL, mode="research",
                            audit=False).analyze("2330", 5)
    assert res.decision["signal"] in res.decision["client_message"]
    assert f"{res.decision['signal_score']:+.2f}" in res.decision["client_message"]


# ------------------------------------------------------------------------------------------ audit
def test_audit_detects_tail_deletion_after_anchor_and_rewrites(tmp_path):
    p = tmp_path / "a.sqlite"
    log = AuditLog(path=p, key="k1")
    for i in range(5):
        log.record(actor="t", mode="research", kind="x", subject=str(i), response={"i": i})
    log.anchor()
    assert log.verify() == (True, None)
    with sqlite3.connect(p) as c:
        c.execute("DELETE FROM audit WHERE id >= 4")
    assert log.verify()[0] is False                 # deletion of anchored rows is caught
    # an attacker without the key cannot rebuild a valid chain
    p2 = tmp_path / "b.sqlite"
    log2 = AuditLog(path=p2, key="k1")
    for i in range(3):
        log2.record(actor="t", mode="research", kind="x", subject=str(i))
    with sqlite3.connect(p2) as c:
        c.execute("UPDATE audit SET response = '\"建議買進\"' WHERE id = 2")
    assert log2.verify() == (False, 2)
    assert AuditLog(path=p2, key="other").verify()[0] is False


def test_audit_chain_survives_concurrent_writers(tmp_path):
    p = tmp_path / "a.sqlite"
    AuditLog(path=p, key="k")

    def write(n):
        log = AuditLog(path=p, key="k")
        for i in range(15):
            log.record(actor=f"w{n}", mode="research", kind="x", subject=str(i))
    ts = [threading.Thread(target=write, args=(n,)) for n in range(4)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    log = AuditLog(path=p, key="k")
    assert log.verify() == (True, None) and log.head()["count"] == 60


# ------------------------------------------------------------------------------------------ metering
def test_quota_is_atomic_under_concurrency(tmp_path):
    store = TenantStore(path=tmp_path / "t.sqlite")
    t, _ = store.create("x", "free")
    granted = []

    def hit():
        try:
            TenantStore(path=tmp_path / "t.sqlite").charge(t, "analyze", analysis=True)
            granted.append(1)
        except QuotaExceeded:
            pass
    ts = [threading.Thread(target=hit) for _ in range(12)]
    [x.start() for x in ts]
    [x.join() for x in ts]
    assert len(granted) == t.plan.analyses_per_day


def test_key_rotation_and_licence_revocation(tmp_path):
    store = TenantStore(path=tmp_path / "t.sqlite")
    t, key = store.create("投顧", "enterprise", licensed=True, mode="advisor")
    new = store.rotate_key(t.id)
    assert store.authenticate(key) is None and store.authenticate(new).id == t.id
    t2 = store.set_plan(t.id, "enterprise", licensed=False)
    assert t2.mode == "research" and not t2.licensed


# ------------------------------------------------------------------------------------------ portfolio
def test_portfolio_merges_aliases_and_converts_currency():
    prov = StrictProvider()
    r = portfolio_risk(prov, {"2330": 1000, "2330.TW": 1000, "NVDA": 10}, horizon=5)
    assert r["n_positions"] == 2
    tw = next(p for p in r["positions"] if p["ticker"] == "2330")
    assert tw["shares"] == 2000
    usd = portfolio_risk(prov, {"2330": 1000, "NVDA": 10}, horizon=5, base_ccy="USD")
    twd = portfolio_risk(prov, {"2330": 1000, "NVDA": 10}, horizon=5, base_ccy="TWD")
    fx = float(prov.prices(prov.symbol("TWD=X"))["close"].iloc[-1])
    assert usd["total_value"] == pytest.approx(twd["total_value"] / fx, abs=0.02)
    for bad in ({"2330": 0}, {"2330": -5}, {"2330": float("nan")}):
        with pytest.raises(ValueError):
            portfolio_risk(prov, bad)
