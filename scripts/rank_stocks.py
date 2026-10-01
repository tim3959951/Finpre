#!/usr/bin/env python
"""Cross-sectional stock ranking (選股排序): walk-forward backtest and today's ranking snapshot.

  # backtest 2018 → today with monthly refits, then write today's ranking for the quant agent
  python scripts/rank_stocks.py --universe tw50 --horizon 5 --backtest --start 2018-01-01 --out logs/rank_tw50_h5
  # daily use (after the close): refit on all history and refresh runs/ranking_TW_h5.json
  python scripts/rank_stocks.py --universe tw50 --horizon 5
  # point-in-time universe: each day the 50 most liquid TWSE stocks (from all listed stocks) — less survivorship bias
  python scripts/rank_stocks.py --pool twse --pit-top 50 --horizon 20 --backtest --out logs/rank_twpit_h20

Outputs
  {out}.csv                          per (scorer, rebalance date) rows
  runs/ranking_backtest_{M}_h{H}.json summary, per-year / per-regime / per-episode breakdowns
  runs/ranking_{M}_h{H}.json          today's ranking (read by the quant agent)
  checkpoints/{model}_{M}_h{H}.pkl    the ranker refit on all labelled history
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fintech_agent.config import get_settings  # noqa: E402
from fintech_agent.data import LiveDataProvider, parse_symbol  # noqa: E402
from fintech_agent.data.universe import (UNIVERSES, bulk_prices, point_in_time_members,  # noqa: E402
                                         tw_listed_codes)
from fintech_agent.evaluation.regimes import market_regime  # noqa: E402
from fintech_agent.forecasting.registry import checkpoint_path  # noqa: E402
from fintech_agent.product.scorecard import archive_ranking  # noqa: E402
from fintech_agent.ranking import (LGBMRanker, RankConfig, breakdown, build_panel, factor_scorers,  # noqa: E402
                                   rank_backtest, summarize)


def load_universe(prov: LiveDataProvider, tickers: list[str], market: str, chips: bool):
    prices, covs, names = {}, {}, {}
    for t in tickers:
        sym = prov.symbol(t)
        px = prov.prices(sym)
        if len(px) < 300:
            print(f"  {t}: skipped ({len(px)} bars)")
            continue
        prices[sym.code] = px
        names[sym.code] = prov.profile(sym).get("name") or sym.code
        if chips:
            c = prov.covariates(sym, px)
            if len(c):
                covs[sym.code] = c
        print(f"  {t}: {len(px)} bars {px.index[0].date()} → {px.index[-1].date()}", flush=True)
    return prices, covs, names


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", choices=sorted(UNIVERSES), default="tw50")
    ap.add_argument("--pool", choices=["twse"], default=None,
                    help="point-in-time universe chosen daily from all listed TWSE stocks (overrides --universe)")
    ap.add_argument("--pit-top", type=int, default=50, help="size of the point-in-time universe")
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--years", type=float, default=13, help="price history to load")
    ap.add_argument("--backtest", action="store_true")
    ap.add_argument("--start", default="2018-01-01")
    ap.add_argument("--end", default=None)
    ap.add_argument("--retrain", default="M")
    ap.add_argument("--train-years", type=float, default=None)
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--out", default=None, help="per-period CSV path prefix")
    ap.add_argument("--no-snapshot", action="store_true")
    ap.add_argument("--smooth", nargs="*", type=float, default=[], help="EMA weights for low-turnover variants, e.g. 0.5 0.75")
    ap.add_argument("--scorers", nargs="*", default=None, help="limit the backtest to these scorers")
    ap.add_argument("--tag", default="", help="suffix for the backtest report file (experiments)")
    ap.add_argument("--cached-chips", type=int, default=0,
                    help="reuse stored 籌碼 history up to N days old instead of refreshing (backtests)")
    ap.add_argument("--stale-nonmembers", type=int, default=45,
                    help="daily snapshot: stocks no longer in the point-in-time universe only feed training rows, so "
                         "their 籌碼 history may be this many days old (saves ~3 FinMind calls per stock per day)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING)

    s = get_settings()
    prov = LiveDataProvider(s)
    prov.years = args.years
    prov.fm.max_wait_s = 3600
    prov.chips_stale_ok_days = args.cached_chips
    t0 = time.time()
    members = None
    if args.pool:
        market, label = "TW", f"twse-pit{args.pit_top}"
        codes = tw_listed_codes(prov)
        print(f"pool: {len(codes)} listed TWSE stocks", flush=True)
        allpx = bulk_prices(prov, codes, market, progress=lambda m: print("  " + m, flush=True))
        members = point_in_time_members(allpx, top=args.pit_top)
        recent = members.loc[members.index >= pd.Timestamp(args.start) - pd.Timedelta(days=400)]
        ever = [t for t, v in recent.any().items() if v]
        print(f"{len(ever)} stocks were in the point-in-time top {args.pit_top} since {args.start}", flush=True)
        info = prov._tw_info_table().drop_duplicates("stock_id").set_index("stock_id")["stock_name"]
        prices = {t: allpx[t] for t in ever}
        names = {t: str(info.get(t, t)) for t in ever}
        covs = {}
        latest = members.iloc[-1].fillna(False).astype(bool) if len(members) else None
        current = set(latest[latest].index) if latest is not None else set(ever)
        for k, t in enumerate(ever):
            fresh = args.backtest or t in current
            prov.chips_stale_ok_days = args.cached_chips if fresh else max(args.cached_chips, args.stale_nonmembers)
            c = prov.covariates(prov.symbol(t), prices[t])
            if len(c):
                covs[t] = c
            if k % 25 == 0:
                print(f"  籌碼 {k + 1}/{len(ever)}", flush=True)
        members = members[ever]
        chips = True
    else:
        market, tickers = UNIVERSES[args.universe]
        label = args.universe
        chips = market == "TW"
        prices, covs, names = load_universe(prov, tickers, market, chips)
    idx_sym = parse_symbol(s.get_path(f"data.market_index.{market}", "^TWII"))
    mkt = prov.prices(idx_sym)["close"]
    panel = build_panel(prices, mkt, args.horizon, covariates=covs or None, members=members)
    print(f"panel {panel.shape[0]:,} rows × {panel.shape[1]} cols, {len(prices)} tickers "
          f"({time.time() - t0:.0f}s)", flush=True)
    cost = float(s.get_path(f"evaluation.cost_bps.{market}", 0))
    cfg = RankConfig(horizon=args.horizon, start=args.start, end=args.end, retrain=args.retrain,
                     train_years=args.train_years, top_n=args.top, cost_bps=cost, smooth=tuple(args.smooth))
    runs = s.resolve_path("evaluation.runs_dir")
    ranker = LGBMRanker(chips=chips)
    bt_path = runs / f"ranking_backtest_{label}_h{args.horizon}{args.tag}.json"
    if args.backtest:
        scorers = [LGBMRanker(chips=False)] + ([LGBMRanker(chips=True)] if chips else []) + factor_scorers(panel)
        if args.scorers:
            scorers = [x for x in scorers if x.name in args.scorers]
        per = rank_backtest(panel, scorers, cfg, regime=market_regime(mkt), market=market,
                            progress=lambda m: print("  " + m, flush=True))
        summ = summarize(per, cfg)
        pd.set_option("display.width", 220)
        pd.set_option("display.max_columns", 30)
        cols = ["scorer", "periods", "ic_mean", "ic_t", "ic_pos", "top_ann_net", "ew_ann", "excess_ann", "excess_ann_gross",
                "turnover", "buf_excess_ann", "buf_turnover", "buf_sharpe_net", "ew_sharpe", "ls_ann_net"]
        print(f"\nRanking backtest {market} h={args.horizon} ({per['date'].min().date()} → {per['date'].max().date()}, "
              f"cost {cost} bps round trip):")
        print(summ[cols].round(3).to_string(index=False))
        by_year, by_regime, by_episode = (breakdown(per, cfg, b) for b in ("year", "regime", "episode"))
        print("\nIC by year:")
        print(by_year.pivot(index="year", columns="scorer", values="ic_mean").round(3).to_string())
        print("\nIC by regime:")
        print(by_regime.pivot(index="regime", columns="scorer", values="ic_mean").round(3).to_string())
        if args.out:
            per.to_csv(f"{args.out}.csv", index=False)
        imp = ranker.fit(panel[panel["fwd_rank"].notna()]).feature_importance()
        bt_path.write_text(json.dumps({
            "config": cfg.__dict__, "universe": label, "market": market, "tickers": list(prices),
            "summary": json.loads(summ.to_json(orient="records")),
            "by_year": json.loads(by_year.to_json(orient="records")),
            "by_regime": json.loads(by_regime.to_json(orient="records")),
            "by_episode": json.loads(by_episode.to_json(orient="records")),
            "feature_importance": {k: round(float(v), 4) for k, v in imp.head(20).items()},
            "created": datetime.now().isoformat(timespec="seconds")}, ensure_ascii=False, indent=1))
        print(f"\nreport → {bt_path}")
    if args.no_snapshot:
        return
    # ---- today's ranking: refit on every label known today, score the latest complete cross-section
    if ranker.model is None:
        ranker.fit(panel[panel["fwd_rank"].notna()])
    ranker.save(checkpoint_path(s, ranker.name, market, args.horizon))
    counts = panel.groupby(level="date").size()
    last = counts.index[counts >= 0.8 * counts.max()].max()
    x = panel.loc[last]
    sc = pd.Series(ranker.score(x), index=x.index).sort_values(ascending=False)
    bt = {}
    if bt_path.exists():
        rep = json.loads(bt_path.read_text())
        row = next((r for r in rep["summary"] if r["scorer"] == ranker.name), None)
        if row:
            bt = {k: row.get(k) for k in ("ic_mean", "ic_t", "excess_ann", "buf_excess_ann", "info_ratio",
                                          "top_ann_net", "ew_ann")}
            bt["top_n"] = rep["config"]["top_n"]
            bt["period"] = f"{rep['config']['start']} → {rep['created'][:10]}"
    snap = {"as_of": str(last.date()), "model": ranker.name, "universe": label, "market": market,
            "horizon": args.horizon, "n": len(sc), "backtest": bt,
            "ranks": [{"ticker": t, "name": names.get(t, t), "rank": i + 1, "score": round(float(v), 4)}
                      for i, (t, v) in enumerate(sc.items())]}
    out = runs / f"ranking_{market}_h{args.horizon}.json"
    out.write_text(json.dumps(snap, ensure_ascii=False, indent=1))
    print(f"\n{ranker.name} ranking as of {last.date()} (top {args.top}):")
    for r in snap["ranks"][:args.top]:
        print(f"  {r['rank']:>2}. {r['ticker']} {r['name']}  score {r['score']:.3f}")
    archive_ranking(snap, s)
    print(f"snapshot → {out}")


if __name__ == "__main__":
    main()
