#!/usr/bin/env python
"""Walk-forward benchmark of forecasting models + champion/challenger A/B test.

Examples
  python scripts/benchmark.py --market TW --tickers 2330 2317 2454 2881 0050 --horizon 5
  python scripts/benchmark.py --market US --tickers AAPL MSFT NVDA SPY JPM --horizon 20 --windows 40 --step 20
  python scripts/benchmark.py --market TW --min-origin 2025-10-01 --promote     # post-release test, update champion
  python scripts/benchmark.py --universe tw50 --models timesfm-2.5 chronos-2 chronos-2-cov lgbm lgbm-cov naive \
      --min-origin 2025-10-01                     # 50 stocks + 籌碼/market/FX covariates
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fintech_agent.config import get_settings  # noqa: E402
from fintech_agent.data import LiveDataProvider  # noqa: E402
from fintech_agent.evaluation.abtest import compare  # noqa: E402
from fintech_agent.evaluation.backtest import BacktestConfig, run_backtest  # noqa: E402
from fintech_agent.evaluation.experiments import ExperimentStore  # noqa: E402
from fintech_agent.data.universe import UNIVERSES  # noqa: E402
from fintech_agent.forecasting.registry import SPECS  # noqa: E402

DEFAULT_TICKERS = {"TW": UNIVERSES["tw6"][1], "US": UNIVERSES["us6"][1]}
DEFAULT_MODELS = ["timesfm-2.5", "chronos-2", "chronos-bolt-small", "naive", "drift", "arima", "lgbm", "dlinear", "ensemble"]


def cross_section(w: pd.DataFrame, sectors: dict[str, str]) -> dict | None:
    """Per-ticker CRPS skill vs naive -> median, share of tickers improved, sign-test p-value, sector medians."""
    from scipy import stats
    if "crps_rel" not in w or "naive" not in set(w["model"]):
        return None
    pt = w.pivot_table(index="ticker", columns="model", values="crps_rel", aggfunc="mean")
    skill = 1 - pt.div(pt["naive"], axis=0)
    rows = []
    for m in skill.columns:
        if m == "naive":
            continue
        v = skill[m].dropna()
        k = int((v > 0).sum())
        rows.append({"model": m, "n_tickers": len(v), "median_skill_pct": round(v.median() * 100, 2),
                     "mean_skill_pct": round(v.mean() * 100, 2), "share_beating_naive": round(k / len(v), 3),
                     "sign_test_p": round(stats.binomtest(k, len(v), 0.5).pvalue, 4)})
    summary = pd.DataFrame(rows).sort_values("median_skill_pct", ascending=False)
    sec = None
    if sectors:
        skill["sector"] = [sectors.get(t, "?") for t in skill.index]
        counts = skill.groupby("sector").size()
        keep = counts[counts >= 3].index
        if len(keep):
            cols = [c for c in summary["model"].head(6)]
            sec = (skill[skill["sector"].isin(keep)].groupby("sector")[cols].median() * 100).round(2)
            sec.insert(0, "n", counts[keep])
    return {"summary": summary, "sector": sec}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", default="TW", choices=["TW", "US"])
    ap.add_argument("--universe", choices=sorted(UNIVERSES), default=None, help="predefined ticker list (sets --market)")
    ap.add_argument("--tickers", nargs="*")
    ap.add_argument("--models", nargs="*", default=DEFAULT_MODELS)
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--windows", type=int, default=60)
    ap.add_argument("--step", type=int, default=None)
    ap.add_argument("--context", type=int, default=None)
    ap.add_argument("--min-origin", default=None)
    ap.add_argument("--metric", default=None, help="A/B loss per window (default: evaluation.ab_metric = crps)")
    ap.add_argument("--promote", action="store_true", help="update champion if a challenger wins the A/B test")
    ap.add_argument("--out", default=None, help="write per-window CSV here")
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    s = get_settings()
    args.metric = args.metric or s.get_path("evaluation.ab_metric", "crps_rel")
    prov = LiveDataProvider(s)
    prov.fm.max_wait_s = 3600                    # bulk job: wait out FinMind's hourly quota instead of skipping
    if args.universe:
        args.market, tickers = UNIVERSES[args.universe]
    else:
        tickers = args.tickers or DEFAULT_TICKERS[args.market]
    need_cov = any(m in SPECS and SPECS[m].covariates or m.endswith("-cov") for m in args.models)
    prices, covs, sectors = {}, {}, {}
    for t in tickers:
        sym = prov.symbol(t)
        px = prov.prices(sym)
        if len(px) <= 300:
            print(f"  {t}: skipped (only {len(px)} bars)")
            continue
        prices[sym.code] = px["close"]
        prof = prov.profile(sym)
        sectors[sym.code] = prof.get("industry") or prof.get("sector") or "?"
        extra = ""
        if need_cov:
            c = prov.covariates(sym, px)
            if len(c):
                covs[sym.code] = c
                extra = f" · {len(c.columns)} covariates"
        print(f"  {t}: {len(px)} bars {px.index[0].date()} → {px.index[-1].date()}{extra}", flush=True)
    cfg = BacktestConfig(horizon=args.horizon, n_windows=args.windows, step=args.step or args.horizon,
                         context_length=args.context or int(s.get_path("forecasting.context_length", 512)),
                         min_origin=args.min_origin, cost_bps=float(s.get_path(f"evaluation.cost_bps.{args.market}", 0)))
    t0 = time.time()
    w, lb = run_backtest(prices, args.models, cfg, s,
                         progress=lambda k, n, m: print(f"[{k + 1}/{n}] {m} …", flush=True),
                         covariates=covs or None)
    print(f"\nfinished in {time.time() - t0:.0f}s · train cutoff for trainable models: {lb.attrs['train_cutoff']}\n")
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    cols = [c for c in ["model", "n_windows", "crps_rel", "crps_skill", "mase", "skill_vs_naive", "mape", "dir_acc", "ic", "cov80",
                        "strat_sharpe", "bh_sharpe", "sec_per_100"] if c in lb]
    print(lb[cols].round(4).to_string(index=False))

    per_ticker = cross_section(w, sectors)
    if per_ticker is not None:
        print("\nCross-section (per-ticker CRPS skill vs naive):")
        print(per_ticker["summary"].to_string(index=False))
        if per_ticker["sector"] is not None:
            print("\nMedian CRPS skill vs naive by sector (%, sectors with >=3 tickers):")
            print(per_ticker["sector"].to_string())

    store = ExperimentStore(s)
    run_id = store.log_run("benchmark", args.market, args.horizon, list(prices), args.models, cfg.__dict__, lb)
    champ = store.champion(args.market, args.horizon)
    print(f"\nrun_id={run_id} · champion={champ}\n\nA/B vs champion ({args.metric}):")
    results = []
    if champ in set(w["model"]):
        for m in lb["model"]:
            if m == champ:
                continue
            try:
                r = compare(w, champ, m, args.metric, args.horizon, float(s.get_path("evaluation.ab_test_alpha", 0.05)),
                            step=cfg.step)
            except ValueError:
                continue
            results.append(r)
            print(f"  {m:20s} Δ={r.improvement_pct:+6.2f}%  p={r.dm_p:.3f}  win={r.ticker_win_rate:.0%}  → {r.decision}")
    else:
        print(f"  champion {champ} not in this run")
    pairs = {"chronos-2-cov": "chronos-2", "lgbm-cov": "lgbm", "timesfm-2.5-xreg": "timesfm-2.5",
             "timesfm-3.0-cov": "timesfm-3.0", "ensemble-cov": "ensemble"}
    cov_ab = []
    present = set(w["model"])
    if any(c in present and b in present for c, b in pairs.items()):
        print(f"\nCovariate A/B — does adding 籌碼/market/FX help? ({args.metric})")
        for c, b in pairs.items():
            if c in present and b in present:
                r = compare(w, b, c, args.metric, args.horizon, float(s.get_path("evaluation.ab_test_alpha", 0.05)),
                            step=cfg.step)
                cov_ab.append(r)
                print(f"  {c:18s} vs {b:14s} Δ={r.improvement_pct:+6.2f}%  p={r.dm_p:.3f}  "
                      f"win={r.ticker_win_rate:.0%}  → {'covariates help' if r.decision == 'promote' else 'covariates hurt' if r.decision == 'keep' else 'no significant difference'}")
    if args.out:
        w.to_csv(args.out, index=False)
        print(f"per-window rows → {args.out}")
    baselines = {"naive", "drift"} if not s.get_path("evaluation.promote_baselines", False) else set()
    if not s.get_path("forecasting.allow_noncommercial_models", False):   # e.g. TimesFM 3.0 weights
        baselines |= {n for n, sp in SPECS.items() if not sp.commercial_ok}
    winners = sorted([r for r in results if r.decision == "promote" and r.challenger not in baselines],
                     key=lambda r: -r.improvement_pct)
    if any(r.decision == "promote" and r.challenger in baselines for r in results):
        print("\nNOTE: a random-walk baseline beats the champion -> no model has a reliable edge on this metric; "
              "the quant agent's skill factor already down-weights its signal.")
    if args.promote and winners:
        store.set_champion(args.market, args.horizon, winners[0].challenger, winners[0].reason, run_id)
        print(f"\nPROMOTED {winners[0].challenger} → champion for {args.market} h={args.horizon}")
    report = store.dir / f"benchmark_{run_id}.json"
    report.write_text(json.dumps({"config": cfg.__dict__, "tickers": list(prices), "champion": champ,
                                  "cross_section": None if per_ticker is None else
                                  json.loads(per_ticker["summary"].to_json(orient="records")),
                                  "by_sector": None if per_ticker is None or per_ticker["sector"] is None else
                                  json.loads(per_ticker["sector"].reset_index().to_json(orient="records")),
                                  "leaderboard": json.loads(lb.to_json(orient="records")),
                                  "ab": [r.to_dict() for r in results],
                                  "covariate_ab": [r.to_dict() for r in cov_ab]}, ensure_ascii=False, indent=1, default=str))
    print(f"report → {report}")


if __name__ == "__main__":
    main()
