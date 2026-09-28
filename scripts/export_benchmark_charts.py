#!/usr/bin/env python
"""Collect every benchmark result into one JSON for the chart dashboard and the PNG charts.

  python scripts/export_benchmark_charts.py            # defaults match the v0.3 runs (logs/ and runs/)
  python scripts/plot_benchmarks.py                    # PNGs from the JSON -> docs/benchmarks/v0.3/

Sections
  post_release : v0.2 test, 2025-10 → 2026-09 (after the foundation models' release — no pretraining leakage)
  long_history : 2018 → 2026 walk-forward, LightGBM refit monthly (foundation models may have seen this period)
  ranking      : cross-sectional stock ranking backtests
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fintech_agent.config import get_settings  # noqa: E402
from fintech_agent.data import LiveDataProvider, parse_symbol  # noqa: E402
from fintech_agent.evaluation.metrics import leaderboard  # noqa: E402
from fintech_agent.evaluation.regimes import market_regime, skill_by_period  # noqa: E402

LONG = ["TW:5:logs/lh_tw50_h5_ml.csv,logs/lh_tw50_h5_fm.csv", "TW:20:logs/lh_tw50_h20_ml.csv,logs/lh_tw50_h20_fm.csv",
        "US:5:logs/lh_us50_h5_ml.csv,logs/lh_us50_h5_fm.csv", "US:20:logs/lh_us50_h20_ml.csv,logs/lh_us50_h20_fm.csv"]
RANK = ["twse-pit50:TW:20:logs/rank_twpit_h20", "twse-pit50:TW:5:logs/rank_twpit_h5",
        "tw50:TW:20:logs/rank_tw50_h20", "tw50:TW:5:logs/rank_tw50_h5",
        "us50:US:20:logs/rank_us50_h20", "us50:US:5:logs/rank_us50_h5"]
V02 = {("TW", 5): "benchmark_TW50_h5.json", ("TW", 20): "benchmark_TW50_h20.json",
       ("US", 5): "benchmark_US50_h5.json", ("US", 20): "benchmark_US50_h20.json"}
V02_CSV = {("TW", 5): "logs/b50_tw50_h5.csv", ("TW", 20): "logs/b50_tw50_h20.csv",
           ("US", 5): "logs/b50_us50_h5.csv", ("US", 20): "logs/b50_us50_h20.csv"}
COST = {}


def strategy(w: pd.DataFrame, market: str, h: int) -> dict:
    """Strategy Sharpe per model with the v0.3 cost accounting (costs only when the position changes)."""
    lb = leaderboard(w, h, COST.get(market, 0.0), step=h)
    return {r["model"]: {"strat_sharpe": _f(r.get("strat_sharpe"), 3), "bh_sharpe": _f(r.get("bh_sharpe"), 3),
                         "exposure": _f(r.get("exposure"), 3)} for _, r in lb.iterrows()}
EQUITY_SCORERS = ["xs-lgbm-chips", "xs-lgbm", "momentum", "foreign_flow"]


def _f(x, nd=4):
    return None if x is None or (isinstance(x, float) and not np.isfinite(x)) else round(float(x), nd)


def overall(w: pd.DataFrame) -> list[dict]:
    pt = w.pivot_table(index="ticker", columns="model", values="crps_rel", aggfunc="mean")
    tick_skill = 1 - pt.div(pt["naive"], axis=0)
    g = w.groupby("model")
    naive = g["crps_rel"].mean()["naive"]
    rows = []
    for m, d in g:
        if m == "naive":
            continue
        k, n = int((tick_skill[m] > 0).sum()), int(tick_skill[m].notna().sum())
        rows.append({"model": m, "windows": int(len(d)), "crps_skill_pct": _f((1 - d["crps_rel"].mean() / naive) * 100, 3),
                     "median_ticker_skill_pct": _f(tick_skill[m].median() * 100, 3),
                     "share_beating_naive": _f(k / n, 3), "sign_p": _f(stats.binomtest(k, n, 0.5).pvalue, 5),
                     "dir_acc": _f(d["dir_hit"].mean()), "cov80": _f(d["cov80"].mean())})
    return sorted(rows, key=lambda r: -r["crps_skill_pct"])


def rolling_skill(w: pd.DataFrame, months: int = 12) -> list[dict]:
    d = w.assign(month=pd.to_datetime(w["origin"]).dt.to_period("M"))
    s = d.groupby(["month", "model"])["crps_rel"].sum().unstack("model").sort_index()
    roll = s.rolling(months, min_periods=months).sum()
    skill = (1 - roll.div(roll["naive"], axis=0)) * 100
    skill = skill.drop(columns="naive").dropna(how="all")
    return [{"month": str(p), **{m: _f(v, 3) for m, v in r.items()}} for p, r in skill.iterrows()]


def long_history(spec: str, regimes: dict) -> dict:
    market, h, paths = spec.split(":", 2)
    frames = [pd.read_csv(ROOT / p, parse_dates=["origin"]) for p in paths.split(",") if (ROOT / p).exists()]
    if not frames:
        return {}
    w = pd.concat(frames).drop_duplicates(["model", "ticker", "origin"])
    per = skill_by_period(w, regimes.get(market), market)
    post = w[w["origin"] >= "2025-10-01"]
    rows = overall(w)
    strat = strategy(w, market, int(h))
    for r in rows:
        r.update(strat.get(r["model"], {}))
    return {"market": market, "horizon": int(h), "tickers": int(w["ticker"].nunique()),
            "start": str(w["origin"].min().date()), "end": str(w["origin"].max().date()),
            "overall": rows, "post_release": overall(post) if len(post) else [],
            "by_year": per.get("year", []), "by_regime": per.get("regime", []), "by_episode": per.get("episode", []),
            "rolling_12m": rolling_skill(w)}


def ranking(spec: str, runs: Path) -> dict:
    label, market, h, prefix = spec.split(":", 3)
    rep_path, csv = runs / f"ranking_backtest_{label}_h{h}.json", ROOT / f"{prefix}.csv"
    if not rep_path.exists() or not csv.exists():
        return {}
    rep = json.loads(rep_path.read_text())
    per = pd.read_csv(csv, parse_dates=["date"])
    ppy = 252 / int(h)
    eq, ic = {}, {}
    for sc, g in per.groupby("scorer"):
        g = g.sort_values("date")
        ic[sc] = {"date": [str(d.date()) for d in g["date"]], "ic": [_f(v) for v in g["ic"]],
                  "roll": [_f(v) for v in g["ic"].rolling(int(round(ppy)), min_periods=int(round(ppy / 2))).mean()]}
        if sc in EQUITY_SCORERS:
            cols = {"top_net": "top", "ls_net": "ls"} | ({"buf_net": "buffer"} if "buf_net" in g else {})
            eq[sc] = {name: [_f(v) for v in (1 + g[c]).cumprod()] for c, name in cols.items()}
            eq[sc]["date"] = [str(d.date()) for d in g["date"]]
    # net excess of the buffered top-10 book vs equal weight as a function of the round-trip cost
    costs = list(range(0, 65, 5))
    sens = {}
    for sc, g in per.groupby("scorer"):
        if "buf" not in g:
            continue
        gross = float((g["buf"] - g["ew"]).mean() * ppy)
        turn = float(g["buf_turnover"].mean())
        sens[sc] = [_f(gross - turn * c / 1e4 * ppy, 5) for c in costs]
    g0 = per[per["scorer"] == per["scorer"].iloc[0]].sort_values("date")
    ew = {"date": [str(d.date()) for d in g0["date"]], "ew": [_f(v) for v in (1 + g0["ew"]).cumprod()]}
    return {"label": label, "market": market, "horizon": int(h), "config": rep["config"],
            "n_tickers": len(rep.get("tickers", [])), "summary": rep["summary"], "by_year": rep["by_year"],
            "by_regime": rep["by_regime"], "by_episode": rep["by_episode"],
            "feature_importance": rep.get("feature_importance", {}), "equity": eq, "ew": ew, "ic_series": ic,
            "cost_sensitivity": {"cost_bps": costs, "net_excess": sens}}


def post_release(v02_dir: Path) -> list[dict]:
    out = []
    for (market, h), name in V02.items():
        p = v02_dir / name
        if not p.exists():
            continue
        rep = json.loads(p.read_text())
        cs = {r["model"]: r for r in rep.get("cross_section") or []}
        csv = ROOT / V02_CSV[(market, h)]
        strat = strategy(pd.read_csv(csv, parse_dates=["origin"]), market, h) if csv.exists() else {}
        rows = []
        for r in rep["leaderboard"]:
            if r["model"] == "naive":
                continue
            c = cs.get(r["model"], {})
            rows.append({"model": r["model"], "crps_skill_pct": _f(r["crps_skill"] * 100, 3),
                         "share_beating_naive": c.get("share_beating_naive"), "sign_p": c.get("sign_test_p"),
                         "cov80": _f(r["cov80"]), "dir_acc": _f(r["dir_acc"]), "ic": _f(r["ic"]),
                         "strat_sharpe": strat.get(r["model"], {}).get("strat_sharpe", _f(r["strat_sharpe"], 3)),
                         "bh_sharpe": strat.get(r["model"], {}).get("bh_sharpe", _f(r["bh_sharpe"], 3))})
        cov = [{"challenger": r["challenger"], "base": r["champion"], "improvement_pct": _f(r["improvement_pct"], 3),
                "p": _f(r["dm_p"], 4), "win_rate": _f(r["ticker_win_rate"], 3), "decision": r["decision"]}
               for r in rep.get("covariate_ab", [])]
        out.append({"market": market, "horizon": h, "windows": int(rep["leaderboard"][0]["n_windows"]),
                    "models": sorted(rows, key=lambda r: -r["crps_skill_pct"]), "covariate_ab": cov})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--long", nargs="*", default=LONG)
    ap.add_argument("--rank", nargs="*", default=RANK)
    ap.add_argument("--v02", default="docs/benchmarks/v0.2")
    ap.add_argument("--out", default="docs/benchmarks/v0.3/chart_data.json")
    args = ap.parse_args()
    s = get_settings()
    COST.update({m: float(s.get_path(f"evaluation.cost_bps.{m}", 0)) for m in ("TW", "US")})
    prov = LiveDataProvider(s)
    prov.years = 13
    regimes, idx = {}, {}
    for m in ("TW", "US"):
        c = prov.prices(parse_symbol(s.get_path(f"data.market_index.{m}")))["close"]
        regimes[m] = market_regime(c)
        c = c[c.index >= "2017-06-01"]
        idx[m] = {"date": [str(d.date()) for d in c.index[::5]], "close": [_f(v, 2) for v in c.to_numpy()[::5]],
                  "regime": [None if pd.isna(r) else r for r in regimes[m].reindex(c.index).to_numpy()[::5]]}
    data = {"created": datetime.now().isoformat(timespec="seconds"),
            "post_release": post_release(ROOT / args.v02),
            "long_history": [x for x in (long_history(sp, regimes) for sp in args.long) if x],
            "ranking": [x for x in (ranking(sp, s.resolve_path("evaluation.runs_dir")) for sp in args.rank) if x],
            "market_index": idx}
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")))
    print(f"→ {out} ({out.stat().st_size / 1e3:.0f} kB): {len(data['post_release'])} post-release, "
          f"{len(data['long_history'])} long-history, {len(data['ranking'])} ranking runs")


if __name__ == "__main__":
    main()
