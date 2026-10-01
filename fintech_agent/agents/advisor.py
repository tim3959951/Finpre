"""Investment advisor: orchestrates the three specialists, makes the final call, and chats with the client."""
from __future__ import annotations

import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import date
from typing import Callable

import numpy as np

from ..config import Settings, get_settings
from ..data.provider import DataProvider
from ..evaluation.abtest import compare
from ..evaluation.backtest import BacktestConfig, run_backtest
from ..evaluation.experiments import ExperimentStore
from ..features.indicators import atr
from ..forecasting.registry import list_models
from ..llm.base import LLMClient, Message, Tool, extract_json, run_tool_loop
from ..llm.factory import get_llm
from ..product.audit import AuditLog
from ..product.compliance import (RESEARCH_DISCLAIMER, ComplianceGuard, ComplianceReport, as_text, as_text_list,
                                  research_decision)
from . import prompts as P
from .base import AgentReport, AnalysisContext, ClientProfile, _jsonable
from .specialists import FundamentalAgent, QuantAgent, TechnicalAgent

log = logging.getLogger(__name__)
ACTIONS = ["賣出", "減碼", "持有", "觀望", "分批買進", "買進", "避開"]
DISCLAIMER = ("本內容由 AI 模型根據公開資料自動產生，僅供研究與教育用途，不構成任何投資建議或要約；"
              "投資有風險，過去績效不代表未來表現，請自行判斷並承擔風險。")


@dataclass
class AnalysisResult:
    ctx: AnalysisContext
    reports: dict[str, AgentReport]
    decision: dict
    elapsed_s: float
    events: list[str] = field(default_factory=list)
    mode: str = "advisor"
    compliance: dict = field(default_factory=dict)               # public: counts only
    audit_compliance: dict = field(default_factory=dict, repr=False)   # full record (removed text) — audit log only

    def brief(self) -> dict:
        return {"ticker": self.ctx.symbol.code, "name": self.ctx.name, "market": self.ctx.symbol.market,
                "close": round(self.ctx.close, 2), "as_of": str(self.ctx.prices.index[-1].date()), "mode": self.mode,
                "decision": {k: v for k, v in self.decision.items() if k not in ("draft",)},
                "agents": {k: r.brief() for k, r in self.reports.items()}}


class InvestmentAdvisor:
    def __init__(self, provider: DataProvider, settings: Settings | None = None, llm: LLMClient | None = None,
                 agent_llms: dict[str, LLMClient] | None = None, quant_panel: list[str] | None = None,
                 mode: str | None = None, actor: str = "local", audit: AuditLog | bool | None = None):
        self.s = settings or get_settings()
        self.provider = provider
        # research (default): no buy/sell advice or price levels; advisor: licensed firms only (see product/compliance)
        self.mode = mode or self.s.get_path("product.mode", "research")
        self.guard = ComplianceGuard(self.mode)
        self.actor = actor
        if audit is None:
            audit = bool(self.s.get_path("product.audit", True))
        self.audit = AuditLog(self.s) if audit is True else (audit or None)
        self.llm = llm if llm is not None else get_llm("advisor", self.s)
        al = agent_llms or {}
        self.technical = TechnicalAgent(provider, al.get("technical", get_llm("technical", self.s) if llm is None else llm), self.s)
        self.fundamental = FundamentalAgent(provider, al.get("fundamental", get_llm("fundamental", self.s) if llm is None else llm), self.s)
        self.quant = QuantAgent(provider, al.get("quant", get_llm("quant", self.s) if llm is None else llm), self.s,
                                panel=quant_panel)
        for a in (self.technical, self.fundamental, self.quant):
            a.mode = self.mode
        self.weights = dict(self.s.get_path("agents.weights", {"technical": 0.3, "fundamental": 0.3, "quant": 0.4}))
        self.last_result: AnalysisResult | None = None
        self.results: dict[str, AnalysisResult] = {}

    # ------------------------------------------------------------------ context
    def build_context(self, ticker: str, horizon: int, client: ClientProfile) -> AnalysisContext:
        sym = self.provider.symbol(ticker)
        prices = self.provider.prices(sym)
        if prices is None or len(prices) < 60:
            raise ValueError(f"找不到 {ticker} 的足夠價格資料（請確認代號，台股如 2330、美股如 NVDA）")
        return AnalysisContext(sym, prices, self.provider.profile(sym), horizon, client,
                               self.provider.market_context(sym.market))

    # ------------------------------------------------------------------ decision engine
    def draft_decision(self, ctx: AnalysisContext, reports: dict[str, AgentReport]) -> dict:
        w = {k: self.weights.get(k, 0.0) for k in reports}
        q = reports.get("quant")
        # a quant model that doesn't beat random walk gets its weight cut
        if q and q.evidence.get("skill_factor") is not None:
            w["quant"] *= float(np.clip(q.evidence["skill_factor"] * 1.5, 0.3, 1.0))
        num = sum(w[k] * r.confidence * r.score for k, r in reports.items())
        den = sum(w[k] * r.confidence for k, r in reports.items()) or 1e-9
        score = num / den
        scores = np.array([r.score for r in reports.values()])
        dispersion = float(scores.std()) if len(scores) > 1 else 0.0
        conf = float(np.clip(den / (sum(w.values()) or 1) * (1 - min(dispersion, 2) / 4), 0.05, 0.95))
        holding = ctx.symbol.code in (ctx.client.holdings or {})
        aggressive = ctx.client.risk == "積極"
        buy_th = 0.8 if aggressive else 1.0
        if score >= buy_th:
            action = "買進"
        elif score >= 0.35:
            action = "分批買進"
        elif score > -0.35:
            action = "持有" if holding else "觀望"
        elif score > -buy_th:
            action = "減碼" if holding else "避開"
        else:
            action = "賣出" if holding else "避開"
        close = ctx.close
        p = ctx.prices
        a = float(atr(p["high"], p["low"], p["close"], 14).iloc[-1])
        stop = close - ctx.client.atr_stop_mult * a
        support = [lv for _, lv in (reports["technical"].evidence.get("support") or [])] if "technical" in reports else []
        resist = [lv for _, lv in (reports["technical"].evidence.get("resistance") or [])] if "technical" in reports else []
        fq = (q.artifacts.get("forecasts") or {}).get(q.evidence.get("champion")) if q else None
        p10 = float(fq.q(0.1)[-1]) if fq is not None else None
        p90 = float(fq.q(0.9)[-1]) if fq is not None else None
        t1 = min([x for x in resist if x > close] + ([p90] if p90 and p90 > close else []), default=close + 2 * a)
        t2 = close + 2 * (close - stop)
        hv = float(np.log(p["close"]).diff().tail(20).std() * np.sqrt(252))
        vol_scale = float(np.clip(0.30 / hv, 0.3, 1.0)) if hv > 0 else 1.0
        pos = 0.0
        if action in ("買進", "分批買進"):
            pos = ctx.client.max_position_pct * min(1.0, abs(score) / 1.5) * (0.5 + conf / 2) * vol_scale
        entry_lo = max(support[0], close - a) if support else close - a
        return _jsonable({
            "action": action, "score": round(score, 3), "confidence": round(conf, 3),
            "effective_weights": {k: round(v, 3) for k, v in w.items()}, "agent_dispersion": round(dispersion, 3),
            "position_pct": round(pos, 1), "entry_zone": [round(entry_lo, 2), round(close, 2)],
            "stop_loss": round(stop, 2), "take_profit": [round(t1, 2), round(t2, 2)],
            "atr14": round(a, 2), "hv20_ann_pct": round(hv * 100, 1),
            "model_band_p10_p90": [round(p10, 2) if p10 else None, round(p90, 2) if p90 else None],
        })

    def decide(self, ctx: AnalysisContext, reports: dict[str, AgentReport]) -> dict:
        if self.guard.research:
            return self.decide_research(ctx, reports)
        draft = self.draft_decision(ctx, reports)
        decision = {**draft, "conviction": int(round(draft["confidence"] * 100)),
                    "horizon": f"{ctx.horizon} 個交易日（{ctx.client.horizon}）", "draft": draft,
                    "advisor_llm": "rule-only", "disclaimer": DISCLAIMER}
        decision["thesis"] = "；".join(f"{r.title}{r.stance}（{r.score:+.2f}）" for r in reports.values())
        decision["risks"] = sum((r.risks[:2] for r in reports.values()), [])[:5]
        decision["monitoring"] = [f"收盤跌破停損價 {draft['stop_loss']}", f"突破壓力/目標 {draft['take_profit'][0]}",
                                  "法人籌碼或月營收出現反轉", "量化模型上漲機率跌破 45%"]
        decision["client_message"] = (f"綜合三位專家，{ctx.name} 目前建議「{draft['action']}」，信心約 {decision['conviction']}%。"
                                      f"建議部位 {draft['position_pct']}%，停損 {draft['stop_loss']}，"
                                      f"第一目標 {draft['take_profit'][0]}。")
        if not self.llm.enabled:
            return decision
        prompt = P.DECISION_TEMPLATE.format(
            client=json.dumps(ctx.client.to_dict(), ensure_ascii=False), name=ctx.name, code=ctx.symbol.code,
            market=ctx.symbol.market, currency=ctx.symbol.currency, close=round(ctx.close, 2), horizon=ctx.horizon,
            reports=json.dumps({k: r.brief() for k, r in reports.items()}, ensure_ascii=False, indent=1),
            weights=json.dumps(self.weights), draft=json.dumps(draft, ensure_ascii=False, indent=1),
            max_pos=ctx.client.max_position_pct)
        try:
            resp = self.llm.chat([Message("user", prompt)], system=P.ADVISOR, max_tokens=2500, json_mode=True)
            data = extract_json(resp.text) or {}
        except Exception as e:
            log.warning("advisor LLM failed: %s", e)
            decision["advisor_llm"] = f"rule-only (LLM error: {type(e).__name__})"
            return decision
        if data.get("action") not in ACTIONS:
            decision["advisor_llm"] = f"rule-only（{self.llm.provider}:{self.llm.model} 回覆無法解析）"
        else:
            gap = abs(ACTIONS.index(data["action"]) - ACTIONS.index(draft["action"])) if draft["action"] in ACTIONS else 0
            decision.update({k: data[k] for k in ("action", "thesis", "dissent", "risks", "monitoring", "client_message",
                                                  "horizon", "entry_zone", "stop_loss", "take_profit") if data.get(k)})
            if data.get("conviction") is not None:
                decision["conviction"] = int(np.clip(float(data["conviction"]), 0, 100))
            if data.get("position_pct") is not None:
                decision["position_pct"] = round(float(np.clip(float(data["position_pct"]), 0, ctx.client.max_position_pct)), 1)
            if gap >= 2:
                decision["advisor_override"] = f"顧問將決策引擎草案「{draft['action']}」調整為「{data['action']}」"
            decision["advisor_llm"] = f"{self.llm.provider}:{self.llm.model}"
        return decision

    def decide_research(self, ctx: AnalysisContext, reports: dict[str, AgentReport]) -> dict:
        """Research mode: the same engine, but the output is a signal with probabilities — never an action or a price."""
        draft = self.draft_decision(ctx, reports)
        base = research_decision(draft, name=ctx.name, horizon=ctx.horizon, close=ctx.close)
        base["rule_message"] = base["client_message"]          # fallback if the LLM text gets scrubbed away
        base["thesis"] = "；".join(f"{r.title}{r.stance}（{r.score:+.2f}）" for r in reports.values())
        base["risks"] = sum((r.risks[:2] for r in reports.values()), [])[:5]
        base["horizon"] = f"{ctx.horizon} 個交易日"
        base["advisor_llm"] = "rule-only"
        if not self.llm.enabled:
            return base
        summary = {k: base[k] for k in ("signal", "signal_score", "conviction", "return_band_pct", "hv20_ann_pct",
                                         "atr14_pct", "agent_dispersion")}
        prompt = P.RESEARCH_DECISION_TEMPLATE.format(
            name=ctx.name, code=ctx.symbol.code, market=ctx.symbol.market, currency=ctx.symbol.currency,
            close=round(ctx.close, 2), horizon=ctx.horizon, weights=json.dumps(self.weights),
            reports=json.dumps(self.guard.apply({k: r.brief() for k, r in reports.items()})[0], ensure_ascii=False, indent=1),
            summary=json.dumps(summary, ensure_ascii=False, indent=1))
        try:
            resp = self.llm.chat([Message("user", prompt)], system=P.RESEARCH_ANALYST, max_tokens=1500, json_mode=True)
            data = extract_json(resp.text) or {}
        except Exception as e:
            log.warning("research LLM failed: %s", e)
            base["advisor_llm"] = f"rule-only (LLM error: {type(e).__name__})"
            return base
        if not data.get("client_message"):
            base["advisor_llm"] = f"rule-only（{self.llm.provider}:{self.llm.model} 回覆無法解析）"
            return base
        # plain text only: a dict or list here could smuggle content past the guard
        if data.get("thesis"):
            base["thesis"] = as_text(data["thesis"])
        if data.get("risks"):
            base["risks"] = as_text_list(data["risks"], 5)
        base["client_message"] = as_text(data["client_message"])
        base["advisor_llm"] = f"{self.llm.provider}:{self.llm.model}"
        return base

    # ------------------------------------------------------------------ full pipeline
    def analyze(self, ticker: str, horizon: int | None = None, client: ClientProfile | None = None,
                on_event: Callable[[str], None] | None = None) -> AnalysisResult:
        t0 = time.time()
        events: list[str] = []

        def emit(msg: str):
            events.append(msg)
            if on_event:
                on_event(msg)

        horizon = int(horizon or self.s.get_path("forecasting.default_horizon", 5))
        emit(f"載入 {ticker} 資料…")
        ctx = self.build_context(ticker, horizon, client or ClientProfile())
        emit(f"{ctx.name}：{len(ctx.prices)} 根日K，最新 {ctx.prices.index[-1].date()} 收盤 {ctx.close:.2f}")
        reports: dict[str, AgentReport] = {}
        agents = [self.technical, self.fundamental, self.quant]
        with ThreadPoolExecutor(max_workers=3) as ex:
            futs = {ex.submit(a.analyze, ctx): a for a in agents}
            for fut, a in futs.items():
                try:
                    reports[a.key] = fut.result()
                    r = reports[a.key]
                    emit(f"{a.title} 完成：{r.stance}（{r.score:+.2f}，信心 {r.confidence:.0%}，{r.elapsed_s:.1f}s）")
                except Exception as e:
                    log.exception("%s failed", a.key)
                    emit(f"{a.title} 失敗：{e}")
        if not reports:
            raise RuntimeError("所有專家分析皆失敗")
        self._log_shadow(ctx, reports.get("quant"))
        emit("首席研究分析師整合結論中…" if self.guard.research else "首席投資顧問整合決策中…")
        decision = self.decide(ctx, reports)
        decision, rep = self._enforce(ctx, reports, decision)
        res = AnalysisResult(ctx, reports, decision, round(time.time() - t0, 1), events,
                             mode=self.mode, compliance=rep.public(), audit_compliance=rep.to_dict())
        self.last_result = res
        self.results[ctx.symbol.code] = res
        if self.audit:
            try:
                self.audit.record(actor=self.actor, mode=self.mode, kind="analysis", subject=ctx.symbol.code,
                                  request={"ticker": ticker, "horizon": horizon, "client": ctx.client.to_dict()},
                                  response=res.brief(), compliance=rep.to_dict())
            except Exception as e:  # audit failure must be visible but never lose the answer
                log.error("audit log write failed: %s", e)
        return res

    def _enforce(self, ctx: AnalysisContext, reports: dict[str, AgentReport], decision: dict) -> tuple[dict, ComplianceReport]:
        """Research mode: scrub every user-visible string and drop advice-only fields, in place. Price levels are
        judged against this stock's current price (any other price within ±50% of it is removed)."""
        rep = ComplianceReport(self.mode)
        rule_message = decision.pop("rule_message", None)
        if not self.guard.research:
            return decision, rep
        guard = self.guard.with_refs([ctx.close], [ctx.symbol.code])
        for r in reports.values():
            clean, rr = guard.apply({"summary": r.summary, "key_points": r.key_points, "risks": r.risks,
                                     "evidence": r.evidence, "adjustment_reason": r.adjustment_reason,
                                     "signals": [s for s, _ in r.rule_signals]})
            r.summary = clean["summary"] or f"{r.title}綜合訊號{r.stance}（{r.score:+.2f}）。"
            r.key_points, r.risks, r.evidence = clean["key_points"], clean["risks"], clean["evidence"]
            r.adjustment_reason = clean["adjustment_reason"]
            keep = set(clean["signals"])
            r.rule_signals = [(s, v) for s, v in r.rule_signals if s in keep]
            rep.merge(rr)
        decision, rr = guard.apply(decision)
        rep.merge(rr)
        if not decision.get("client_message"):
            msg, _ = guard.apply(rule_message or "")
            decision["client_message"] = msg or "本服務為研究工具，以下僅呈現訊號、機率區間與風險資訊。"
        return decision, rep

    def _log_shadow(self, ctx: AnalysisContext, q: AgentReport | None) -> None:
        """Every live forecast is logged so champion vs challengers can be A/B tested online once actuals arrive."""
        if not q or not q.artifacts.get("forecasts"):
            return
        try:
            store = self.quant.store
            champ = q.evidence.get("champion")
            origin = str(ctx.prices.index[-1].date())
            for m, fc in q.artifacts["forecasts"].items():
                store.log_prediction(fc, ctx.symbol.code, ctx.symbol.market, origin, ctx.close,
                                     role="champion" if m == champ else "shadow")
        except Exception as e:  # logging must never break an analysis
            log.warning("shadow logging failed: %s", e)

    # ------------------------------------------------------------------ chat with client
    def tools(self, client: ClientProfile, on_event: Callable[[str], None] | None = None,
              refs: list | None = None) -> list[Tool]:
        """Chat tools. `refs` collects (current price, code) of every stock a tool touched, so the chat answer can be
        checked for price levels."""
        refs = refs if refs is not None else []
        build = self.build_context

        def build_context(ticker, horizon, client_):
            ctx = build(ticker, horizon, client_)
            refs.append((ctx.close, ctx.symbol.code))
            return ctx

        def _h(x) -> int:
            try:
                return int(min(max(int(x), 1), 60))
            except (TypeError, ValueError):
                return 5

        def analyze_stock(ticker: str, horizon_days: int = 5) -> dict:
            res = self.analyze(ticker, _h(horizon_days), client, on_event)
            refs.append((res.ctx.close, res.ctx.symbol.code))
            return res.brief()

        def technical_analysis(ticker: str) -> dict:
            ctx = build_context(ticker, 5, client)
            r = self.technical.analyze(ctx)
            return r.brief() | {"evidence": {k: r.evidence.get(k) for k in ("close", "ma_alignment", "support", "resistance",
                                                                          "rsi", "kd", "macd", "atr14")}}

        def fundamental_analysis(ticker: str) -> dict:
            ctx = build_context(ticker, 5, client)
            r = self.fundamental.analyze(ctx)
            return r.brief() | {"sub_scores": r.evidence.get("sub_scores"), "chips": r.evidence.get("chips")}

        def quant_forecast(ticker: str, horizon_days: int = 5) -> dict:
            ctx = build_context(ticker, _h(horizon_days), client)
            r = self.quant.analyze(ctx)
            return r.brief() | {"forecasts": r.evidence.get("forecasts"), "champion": r.evidence.get("champion"),
                                "backtest": r.evidence.get("backtest_on_this_ticker")}

        def compare_models(ticker: str, horizon_days: int = 5, models: list[str] | None = None, windows: int = 40) -> dict:
            sym = self.provider.symbol(ticker)
            px = self.provider.prices(sym)
            horizon_days = _h(horizon_days)
            try:
                windows = int(min(max(int(windows), 5), 60))
            except (TypeError, ValueError):
                windows = 40
            allowed = {m.name for m in list_models(self.s)}            # licence-checked model list
            models = [m for m in (models or ["timesfm-2.5", "chronos-2", "chronos-bolt-small", "naive", "drift"])
                      if m in allowed][:6] or ["naive", "drift"]
            refs.append((float(px["close"].iloc[-1]), sym.code))
            cfg = BacktestConfig(horizon=horizon_days, n_windows=windows, step=horizon_days,
                                 cost_bps=float(self.s.get_path(f"evaluation.cost_bps.{sym.market}", 0)))
            w, lb = run_backtest({sym.code: px["close"]}, models, cfg, self.s)
            champ = ExperimentStore(self.s).champion(sym.market, horizon_days)
            ab = []
            for m in lb["model"]:
                if m != champ and m in set(w["model"]) and champ in set(w["model"]):
                    try:
                        ab.append(compare(w, champ, m, self.s.get_path("evaluation.ab_metric", "crps_rel"), horizon_days,
                                          step=horizon_days).to_dict())
                    except ValueError:
                        pass
            cols = ["model", "crps_rel", "crps_skill", "mase", "skill_vs_naive", "dir_acc", "ic", "cov80", "strat_sharpe", "bh_sharpe"]
            return {"champion": champ, "leaderboard": _jsonable(lb[[c for c in cols if c in lb]].round(4).to_dict("records")),
                    "ab_tests_vs_champion": _jsonable(ab)}

        def get_quote(ticker: str) -> dict:
            sym = self.provider.symbol(ticker)
            px = self.provider.prices(sym)
            c = px["close"]
            refs.append((float(c.iloc[-1]), sym.code))
            return {"ticker": sym.code, "name": self.provider.profile(sym).get("name"), "date": str(c.index[-1].date()),
                    "close": round(float(c.iloc[-1]), 2), "chg_1d_pct": round(float(c.pct_change().iloc[-1] * 100), 2),
                    "ret_20d_pct": round(float((c.iloc[-1] / c.iloc[-21] - 1) * 100), 2)}

        def available_models() -> list[dict]:
            return [{"name": m.name, "family": m.family, "license": m.license, "description": m.description}
                    for m in list_models(self.s)]

        str_t = {"type": "string", "description": "股票代號，台股如 2330 / 00878，美股如 NVDA"}
        hz = {"type": "integer", "description": "預測天數（交易日，1-60；短線 5、波段 20）", "default": 5}
        return [self._guard_tool(t, refs) for t in [
            Tool("analyze_stock", "完整分析：技術、基本面/籌碼/情緒、量化預測三位專家 + 首席顧問決策。",
                 {"type": "object", "properties": {"ticker": str_t, "horizon_days": hz}, "required": ["ticker"]}, analyze_stock),
            Tool("technical_analysis", "只做技術面分析（均線、KD、MACD、RSI、支撐壓力）。",
                 {"type": "object", "properties": {"ticker": str_t}, "required": ["ticker"]}, technical_analysis),
            Tool("fundamental_analysis", "只做基本面、籌碼面（三大法人、融資券）、新聞情緒與大盤分析。",
                 {"type": "object", "properties": {"ticker": str_t}, "required": ["ticker"]}, fundamental_analysis),
            Tool("quant_forecast", "TimesFM 等時間序列模型的價格預測、上漲機率與模型歷史準確度。",
                 {"type": "object", "properties": {"ticker": str_t, "horizon_days": hz}, "required": ["ticker"]}, quant_forecast),
            Tool("compare_models", "在該標的上回測比較多個預測模型並做 champion/challenger A/B 檢定。",
                 {"type": "object", "properties": {"ticker": str_t, "horizon_days": hz,
                                                   "models": {"type": "array", "items": {"type": "string"}},
                                                   "windows": {"type": "integer", "default": 40}},
                  "required": ["ticker"]}, compare_models),
            Tool("get_quote", "最新收盤價與近期漲跌。",
                 {"type": "object", "properties": {"ticker": str_t}, "required": ["ticker"]}, get_quote),
            Tool("available_models", "列出可用的預測模型與授權。", {"type": "object", "properties": {}}, available_models),
        ]]

    def _guard_tool(self, tool: Tool, refs: list) -> Tool:
        """Research mode: the LLM never even sees advice fields (stop-loss, support/resistance, …) or price levels in
        tool results."""
        if not self.guard.research:
            return tool
        fn = tool.fn

        def guarded(**kw):
            out = fn(**kw)
            return self.guard.with_refs([r for r, _ in refs], [c for _, c in refs]).apply(out)[0]
        return Tool(tool.name, tool.description, tool.parameters, guarded)

    def chat(self, history: list[Message], user_text: str, client: ClientProfile,
             on_event: Callable[[str], None] | None = None) -> tuple[str, list[dict]]:
        history.append(Message("user", user_text))
        refs: list = []
        trace: list[dict] = []
        try:
            if not self.llm.enabled:
                answer = self._rule_chat(user_text, client, on_event, refs)
            else:
                persona = P.RESEARCH_ANALYST if self.guard.research else P.ADVISOR
                system = persona + P.CHAT_SYSTEM_SUFFIX.format(client=json.dumps(client.to_dict(), ensure_ascii=False),
                                                               today=date.today().isoformat())
                tool_events = (lambda kind, d: on_event(f"{'呼叫' if kind == 'tool_call' else '完成'} {d['name']}")) if on_event else None
                answer, trace = run_tool_loop(self.llm, history, system, self.tools(client, on_event, refs), max_steps=6,
                                              on_event=tool_events)
        except Exception as e:  # an LLM/network/data error must still produce an answer and an audit row
            log.warning("chat failed: %s", e)
            answer = (f"這次查詢沒有完成（{type(e).__name__}）。請確認股票代號（台股如 2330、美股如 NVDA）後再試一次；"
                      "也可以在「個股分析」頁直接執行完整分析。")
        guard = self.guard.with_refs([r for r, _ in refs], [c for _, c in refs])
        answer, rep = guard.apply(answer)
        if self.guard.research and not answer.strip():
            answer = "本服務為研究工具，無法提供個股買賣建議或價位；可以為您說明這檔股票的訊號強度、機率區間、波動與模型歷史成績。"
        disclaimer = RESEARCH_DISCLAIMER if self.guard.research else DISCLAIMER
        if self.audit:
            try:
                self.audit.record(actor=self.actor, mode=self.mode, kind="chat", subject=",".join(sorted({c for _, c in refs})),
                                  request={"text": user_text}, response={"answer": answer, "tools": [t.get("tool") for t in trace]},
                                  compliance=rep.to_dict())
            except Exception as e:
                log.error("audit log write failed: %s", e)
        return answer + f"\n\n> {disclaimer}", trace

    _NOT_TICKERS = {"I", "A", "AI", "ETF", "PE", "EPS", "USD", "TWD", "OK", "IS", "IT", "THE", "AND", "OR", "TO", "OF",
                    "IN", "ON", "AT", "BE", "DO", "GOOD", "BAD", "BUY", "SELL", "HOLD", "WHAT", "HOW", "WHY", "NOW",
                    "ME", "MY", "YOU", "CAN", "WILL", "DAY", "DAYS", "VS", "KD", "RSI", "MACD", "MA", "IPO", "CEO",
                    "Q", "QOQ", "YOY", "GDP", "CPI", "FED", "US", "TW"}

    def _rule_chat(self, text: str, client: ClientProfile, on_event=None, refs: list | None = None) -> str:
        cands = re.findall(r"(?<![A-Z0-9])(\d{4,6}[A-Z]?|[A-Z]{1,5})(?![A-Z0-9])", text.upper())
        cands = [t for t in dict.fromkeys(cands) if t not in self._NOT_TICKERS]
        if not cands:
            return ("目前為規則模式（未設定 LLM）。請輸入股票代號，例如「分析 2330」或「NVDA 5 日展望」。"
                    "若要自然語言對話，請在側邊欄設定 Anthropic / OpenAI / Ollama。")
        ticker = None
        for c in cands[:5]:                       # first candidate that is a real ticker with price history
            try:
                px = self.provider.prices(self.provider.symbol(c))
                if px is not None and len(px) >= 60:
                    ticker = c
                    break
            except Exception:
                continue
        if ticker is None:
            return f"找不到 {', '.join(cands[:3])} 的價格資料。請輸入正確代號，台股如 2330、美股如 NVDA。"
        m = re.search(r"(\d+)\s*(日|天|day)", text)
        res = self.analyze(ticker, min(max(int(m.group(1)), 1), 60) if m else None, client, on_event)
        if refs is not None:
            refs.append((res.ctx.close, res.ctx.symbol.code))
        d = res.decision
        lines = [f"**{res.ctx.name}（{res.ctx.symbol.code}）** 收盤 {res.ctx.close:.2f}"]
        if res.mode == "research":
            lo, hi = d.get("return_band_pct") or [None, None]
            lines.append(f"- 綜合訊號：**{d['signal']}**（分數 {d['signal_score']:+.2f}，信心 {d['conviction']}%）")
            if lo is not None:
                lines.append(f"- 未來 {res.ctx.horizon} 個交易日報酬率 80% 區間：{lo:+.1f}% ～ {hi:+.1f}%")
        else:
            lines += [f"- 建議：**{d['action']}**（信心 {d['conviction']}%），部位 {d['position_pct']}%",
                      f"- 進場區間 {d['entry_zone']}，停損 {d['stop_loss']}，目標 {d['take_profit']}"]
        for r in res.reports.values():
            lines.append(f"- {r.title}：{r.stance}（{r.score:+.2f}）— {r.summary}")
        return "\n".join(lines)
