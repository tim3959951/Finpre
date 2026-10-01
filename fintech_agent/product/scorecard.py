"""成績單 (scorecard): how good the models really are — walk-forward backtests plus the live track record.

  backtest_scorecard() : headline numbers from the published benchmark (docs/benchmarks/v0.3/chart_data.json)
  live_scorecard()     : every forecast the product has served or tracked, scored after its horizon elapsed
                         (80% band coverage, direction hit rate, Brier score of P(up), error vs the random walk)
  ranking_live()       : realized rank IC of archived ranking snapshots once their holding period is over
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from ..config import PROJECT_ROOT, Settings, get_settings
from ..evaluation.experiments import ExperimentStore

BENCHMARK_FILE = PROJECT_ROOT / "docs" / "benchmarks" / "v0.3" / "chart_data.json"


def backtest_scorecard(path: Path | None = None) -> dict:
    p = Path(path or BENCHMARK_FILE)
    if not p.exists():
        return {"available": False}
    d = json.loads(p.read_text())
    fc = []
    for r in d.get("long_history", []):
        by = {o["model"]: o for o in r["overall"]}
        row = {"market": r["market"], "horizon": r["horizon"], "period": f"{r['start']} → {r['end']}",
               "tickers": r["tickers"]}
        for m in ("lgbm", "timesfm-2.5", "chronos-2"):
            if m in by:
                row[m] = {k: by[m].get(k) for k in ("crps_skill_pct", "share_beating_naive", "cov80", "dir_acc")}
        years = [y for y in r.get("by_year", []) if y["model"] == "lgbm"]
        if years:
            row["lgbm_positive_years"] = f"{sum(y['crps_skill_pct'] > 0 for y in years)}/{len(years)}"
        fc.append(row)
    rk = []
    for r in d.get("ranking", []):
        s = next((x for x in r["summary"] if x["scorer"] in ("xs-lgbm-chips", "xs-lgbm")), None)
        if s:
            rk.append({"universe": r["label"], "market": r["market"], "horizon": r["horizon"], "model": s["scorer"],
                       "ic_mean": s["ic_mean"], "ic_t": s["ic_t"], "ic_positive_share": s["ic_pos"],
                       "excess_gross": (r.get("cost_sensitivity") or {}).get("net_excess", {}).get(s["scorer"], [None])[0],
                       "excess_net": s.get("buf_excess_ann"), "cost_bps": r["config"]["cost_bps"]})
    return {"available": True, "source": str(p.relative_to(PROJECT_ROOT)) if p.is_relative_to(PROJECT_ROOT) else str(p),
            "created": d.get("created"), "forecasting": fc, "ranking": rk}


def _predictions(settings: Settings) -> pd.DataFrame:
    store = ExperimentStore(settings)
    with sqlite3.connect(store.path) as c:
        df = pd.read_sql("SELECT * FROM predictions", c)
    if df.empty:
        return df
    return df.sort_values("id").drop_duplicates(["model", "ticker", "origin_date", "horizon"], keep="first")


def live_scorecard(settings: Settings | None = None, min_n: int = 1) -> dict:
    s = settings or get_settings()
    df = _predictions(s)
    if df.empty:
        return {"pending": 0, "resolved": 0, "models": []}
    pending = int(df["resolved_at"].isna().sum())
    since = str(pd.to_datetime(df["origin_date"]).min().date())
    r = df[df["resolved_at"].notna()].copy()
    if r.empty:
        return {"since": since, "pending": pending, "resolved": 0, "models": []}
    r["up"] = (r["actual_h"] > r["last_price"]).astype(float)
    r["abs_err"] = (r["point_h"] - r["actual_h"]).abs() / r["last_price"]
    r["in80"] = ((r["actual_h"] >= r["q10_h"]) & (r["actual_h"] <= r["q90_h"])).astype(float)
    r["dir_hit"] = (np.sign(r["point_h"] - r["last_price"]) == np.sign(r["actual_h"] - r["last_price"])).astype(float)
    r.loc[r["point_h"] == r["last_price"], "dir_hit"] = np.nan
    r["brier"] = (r["p_up"] - r["up"]) ** 2
    naive = r[r["model"] == "naive"].set_index(["ticker", "origin_date", "horizon"])["abs_err"]
    rows = []
    for (m, h), g in r.groupby(["model", "horizon"]):
        if len(g) < min_n:
            continue
        row = {"model": m, "horizon": int(h), "n": int(len(g)), "cov80": round(float(g["in80"].mean()), 3),
               "dir_acc": None if g["dir_hit"].isna().all() else round(float(g["dir_hit"].mean()), 3),
               "brier": round(float(g["brier"].mean()), 4), "mae_pct": round(float(g["abs_err"].mean() * 100), 3)}
        if m != "naive" and len(naive):
            j = g.set_index(["ticker", "origin_date", "horizon"])["abs_err"].to_frame("m").join(naive.rename("n"), how="inner")
            if len(j) >= min_n:
                row["mae_vs_naive_pct"] = round(float((1 - j["m"].mean() / j["n"].mean()) * 100), 2)
                row["paired_n"] = int(len(j))
        rows.append(row)
    return {"since": since, "pending": pending, "resolved": int(len(r)), "models": sorted(rows, key=lambda x: -x["n"])}


def archive_ranking(snapshot: dict, settings: Settings | None = None) -> Path:
    """Keep every published ranking so its realized IC can be scored later (append-only)."""
    s = settings or get_settings()
    d = s.resolve_path("evaluation.runs_dir") / "ranking_history"
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{snapshot['market']}_h{snapshot['horizon']}_{snapshot['as_of']}.json"
    if not p.exists():
        p.write_text(json.dumps(snapshot, ensure_ascii=False))
    return p


def ranking_live(price_lookup, settings: Settings | None = None) -> dict:
    """Realized Spearman IC of each archived snapshot whose horizon (+1 day entry lag) has passed."""
    s = settings or get_settings()
    d = s.resolve_path("evaluation.runs_dir") / "ranking_history"
    out = []
    for p in sorted(d.glob("*.json")) if d.exists() else []:
        snap = json.loads(p.read_text())
        h, as_of = int(snap["horizon"]), pd.Timestamp(snap["as_of"])
        scores, rets = [], []
        for r in snap["ranks"]:
            px = price_lookup(r["ticker"])
            if px is None or len(px) == 0:
                continue
            after = px[px.index > as_of]
            if len(after) < h + 1:
                continue
            scores.append(r["score"])
            rets.append(after.iloc[h] / after.iloc[0] - 1)
        if len(scores) >= 10:
            ic = float(stats.spearmanr(scores, rets)[0])
            top = np.mean([x for _, x in sorted(zip(scores, rets), reverse=True)[:10]])
            out.append({"as_of": snap["as_of"], "market": snap["market"], "horizon": h, "n": len(scores),
                        "ic": round(ic, 4), "top10_vs_all_pct": round(float((top - np.mean(rets)) * 100), 2)})
    ics = [x["ic"] for x in out]
    return {"snapshots_scored": len(out), "mean_ic": round(float(np.mean(ics)), 4) if ics else None,
            "ic_positive_share": round(float(np.mean([i > 0 for i in ics])), 3) if ics else None, "history": out[-60:]}


def track_universe(provider, tickers: list[str], horizons=(5, 20), settings: Settings | None = None,
                   models: tuple[str | None, ...] = (None, "naive")) -> int:
    """Log today's forecasts for a fixed universe (champion + random walk) so the live scorecard grows every day
    regardless of what users happen to analyse. Returns the number of predictions logged (duplicates are ignored
    when scoring)."""
    from .forecast import forecast_result
    s = settings or get_settings()
    store = ExperimentStore(s)
    n = 0
    for t in tickers:
        try:
            sym = provider.symbol(t)
            px = provider.prices(sym)
            if px is None or len(px) < 80:
                continue
            last, origin = float(px["close"].iloc[-1]), str(px.index[-1].date())
            for h in horizons:
                used = set()
                for m in models:
                    fc, name = forecast_result(provider, sym, px, h, s, model_name=m)
                    if name in used:              # champion fell back to the random walk: log it once
                        continue
                    used.add(name)
                    store.log_prediction(fc, sym.code, sym.market, origin, last, role="tracking")
                    n += 1
        except Exception as e:  # one bad ticker must not stop the job
            import logging
            logging.getLogger(__name__).warning("tracking %s failed: %s", t, e)
    return n
