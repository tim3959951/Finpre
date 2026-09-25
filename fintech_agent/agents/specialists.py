"""The three specialist agents: technical, fundamental/sentiment/chips, quant-ML."""
from __future__ import annotations

import logging
from datetime import datetime

import numpy as np

from ..data.cache import DiskCache
from ..evaluation.abtest import direction_significance
from ..evaluation.backtest import BacktestConfig, run_backtest
from ..evaluation.experiments import ExperimentStore
from ..features.fundamental_signals import fundamental_view
from ..features.sentiment import score_headlines
from ..features.technical_signals import technical_view
from ..forecasting.registry import SPECS, get_forecaster
from ..llm.base import Message, extract_json
from . import prompts as P
from .base import AnalysisContext, BaseAgent

log = logging.getLogger(__name__)


class TechnicalAgent(BaseAgent):
    key, title, role, persona = "technical", "技術分析師", "技術面", P.TECHNICAL

    def gather(self, ctx: AnalysisContext):
        tv = technical_view(ctx.prices)
        ev = dict(tv.evidence)
        ev["last_10_closes"] = [round(float(x), 2) for x in ctx.prices["close"].tail(10)]
        return ev, tv.score, tv.confidence, tv.signals, {}


class FundamentalAgent(BaseAgent):
    key, title, role, persona = "fundamental", "基本面／籌碼／情緒分析師", "基本面、籌碼面、新聞情緒與大盤行情", P.FUNDAMENTAL

    def _sentiment(self, ctx: AnalysisContext, news: list[dict]) -> dict:
        base = score_headlines(news)
        if not news or not self.llm.enabled:
            return base
        heads = "\n".join(f"{i + 1}. [{n.get('date', '')}] {n.get('title')}" for i, n in enumerate(news))
        try:
            resp = self.llm.chat([Message("user", P.SENTIMENT_TEMPLATE.format(name=ctx.name, code=ctx.symbol.code,
                                                                              headlines=heads))],
                                 system="你是金融新聞情緒分析模型，只輸出 JSON。", max_tokens=800,
                                 json_mode=True)
            data = extract_json(resp.text) or {}
            scores = [float(np.clip(float(x), -1, 1)) for x in data.get("scores", [])][: len(news)]
            if scores:
                return {"method": "llm", "n": len(scores), "mean": round(float(np.mean(scores)), 3),
                        "positive": sum(s > 0 for s in scores), "negative": sum(s < 0 for s in scores),
                        "themes": data.get("themes", []),
                        "items": [{"title": n.get("title"), "date": n.get("date"), "score": round(s, 2)}
                                  for n, s in zip(news, scores)]}
        except Exception as e:
            log.warning("LLM sentiment failed, lexicon fallback: %s", e)
        return base

    def gather(self, ctx: AnalysisContext):
        sym = ctx.symbol
        fund = self.provider.fundamentals(sym)
        chips = self.provider.chips(sym, ctx.prices)
        news = self.provider.news(sym, int(self.settings.get_path("agents.news_limit", 15)))
        market = ctx.market or self.provider.market_context(sym.market)
        senti = self._sentiment(ctx, news)
        c = ctx.prices["close"]
        ret20 = float((c.iloc[-1] / c.iloc[-21] - 1) * 100) if len(c) > 21 else None
        fv = fundamental_view(fund, chips, senti, market, sym.market, ret20)
        ev = {"profile": ctx.profile, "fundamentals": fund, "chips": chips, "market": market,
              "sentiment": {k: v for k, v in senti.items() if k != "items"},
              "headlines": senti.get("items", [])[:10], "sub_scores": fv.sub_scores}
        return ev, fv.score, fv.confidence, fv.signals, {"news": news}


class QuantAgent(BaseAgent):
    key, title, role, persona = "quant", "量化 ML 工程師", "時間序列模型預測與模型評估", P.QUANT

    def __init__(self, *a, panel: list[str] | None = None, skill_windows: int = 30, **kw):
        super().__init__(*a, **kw)
        self.store = ExperimentStore(self.settings)
        self.panel = panel
        self.skill_windows = skill_windows
        self.cache = DiskCache(self.settings.resolve_path("evaluation.runs_dir") / "skill_cache", ttl_hours=24)

    def _panel(self, champion: str) -> list[str]:
        if self.panel is not None:
            names = self.panel
        else:
            names = [champion, "chronos-2", "chronos-bolt-small", "naive"]
        allow_nc = self.settings.get_path("forecasting.allow_noncommercial_models", False)
        out = []
        for n in names:
            if n in out or (n in SPECS and SPECS[n].trainable):
                continue
            if n in SPECS and not SPECS[n].commercial_ok and not allow_nc:
                continue
            out.append(n)
        return out

    def _skill(self, ctx: AnalysisContext, models: list[str]) -> dict:
        """Quick walk-forward check of the panel on *this* ticker (cached daily)."""
        key = f"skill_{ctx.symbol.code}_{ctx.horizon}_{'-'.join(models)}_{datetime.now():%Y%m%d}"

        def load():
            cfg = BacktestConfig(horizon=ctx.horizon, n_windows=self.skill_windows, step=ctx.horizon,
                                 context_length=int(self.settings.get_path("forecasting.context_length", 512)),
                                 cost_bps=float(self.settings.get_path(f"evaluation.cost_bps.{ctx.symbol.market}", 0)))
            w, lb = run_backtest({ctx.symbol.code: ctx.prices["close"]}, models, cfg, self.settings)
            out = {}
            for _, r in lb.iterrows():
                hits = w[w["model"] == r["model"]]["dir_hit"].to_numpy()
                out[r["model"]] = {k: (None if isinstance(v, float) and not np.isfinite(v) else
                                       round(float(v), 4) if isinstance(v, (float, np.floating)) else v)
                                   for k, v in r.items() if k in ("mase", "skill_vs_naive", "dir_acc", "cov80", "crps_skill",
                                                                   "strat_sharpe", "bh_sharpe", "n_windows", "ic")}
                out[r["model"]]["dir_p_value"] = round(direction_significance(hits)["p_value"], 4)
            return out
        try:
            return self.cache.obj(key, load) or {}
        except Exception as e:
            log.warning("skill backtest failed: %s", e)
            return {}

    def gather(self, ctx: AnalysisContext):
        h = ctx.horizon
        champion = self.store.champion(ctx.symbol.market, h)
        models = self._panel(champion)
        ctx_len = int(self.settings.get_path("forecasting.context_length", 512))
        series = ctx.prices["close"].to_numpy()[-ctx_len:]
        last = float(series[-1])
        forecasts, results = {}, {}
        for m in models:
            try:
                fc = get_forecaster(m, h, self.settings).predict([series], h)[0]
            except Exception as e:
                log.warning("forecast %s failed: %s", m, e)
                continue
            results[m] = fc
            forecasts[m] = {"ret_pct": round((fc.point[-1] / last - 1) * 100, 2),
                            "p_up": round(fc.prob_above(last), 3),
                            "p10_pct": round((fc.q(0.1)[-1] / last - 1) * 100, 2),
                            "p90_pct": round((fc.q(0.9)[-1] / last - 1) * 100, 2),
                            "price_h": round(float(fc.point[-1]), 2)}
        if champion not in results and results:
            champion = next(iter(k for k in results if k != "naive"), next(iter(results)))
        skill = self._skill(ctx, [m for m in models if m in results])
        signals: list[tuple[str, float]] = []
        if champion not in forecasts:
            return {"error": "所有模型預測失敗"}, 0.0, 0.1, [], {}
        f = forecasts[champion]
        sk = skill.get(champion, {})
        # --- signal from champion's probabilistic forecast
        raw = float(np.clip((f["p_up"] - 0.5) * 8, -2, 2))
        dir_acc = sk.get("dir_acc") or 0.5
        skill_vs_naive = sk.get("skill_vs_naive") or 0.0
        crps_skill = sk.get("crps_skill") or 0.0
        dir_p = sk.get("dir_p_value", 1.0)
        skill_factor = float(np.clip(0.4 + (dir_acc - 0.5) * 4 + skill_vs_naive * 5 + crps_skill * 3
                                     + (0.2 if dir_p < 0.1 else 0), 0.15, 1.0))
        score = raw * skill_factor
        signals.append((f"{champion} 預測 {h} 日報酬 {f['ret_pct']:+.2f}%，上漲機率 {f['p_up']:.0%}", raw))
        signals.append((f"{champion} 在此標的回測：方向準確率 {dir_acc:.0%}（p={dir_p:.2f}），相對 naive 點預測技能 {skill_vs_naive:+.1%}、機率預測技能 {crps_skill:+.1%}"
                        f"，訊號折減係數 {skill_factor:.2f}（分數 {raw:+.2f} → {score:+.2f}）", 0.0))
        dirs = [np.sign(v["ret_pct"]) for k, v in forecasts.items() if k != "naive" and v["ret_pct"] != 0]
        agreement = abs(float(np.mean(dirs))) if dirs else 0.0
        if len(dirs) >= 2:
            signals.append((f"模型方向一致度 {agreement:.0%}（{len(dirs)} 個模型）", 0.0))
        best = min(skill, key=lambda k: skill[k].get("mase") or 9e9) if skill else champion
        cov = sk.get("cov80")
        conf = float(np.clip(0.2 + 0.4 * skill_factor + 0.2 * agreement + (0.1 if cov and 0.7 <= cov <= 0.9 else 0), 0.1, 0.9))
        ev = {"champion": champion, "horizon_days": h, "last_close": round(last, 2), "forecasts": forecasts,
              "backtest_on_this_ticker": skill, "best_model_on_ticker_by_mase": best,
              "model_agreement": round(agreement, 2), "skill_factor": round(skill_factor, 2),
              "licenses": {m: SPECS[m].license for m in results if m in SPECS}}
        if best != champion and skill.get(best, {}).get("mase") and sk.get("mase"):
            gap = (sk["mase"] - skill[best]["mase"]) / sk["mase"]
            if gap > 0.03:
                signals.append((f"{best} 在此標的 MASE 低於 champion {gap:.1%}，建議排入 A/B 測試", 0.0))
        return ev, float(np.clip(score, -2, 2)), conf, signals, {"forecasts": results}
