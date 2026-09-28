#!/usr/bin/env python
"""Merge per-window benchmark CSVs (from `benchmark.py --out`) into markdown tables for the docs.

  python scripts/summarize_benchmarks.py TW:5:logs/b50_tw50_h5.csv TW:20:logs/b50_tw50_h20.csv \\
                                         US:5:logs/b50_us50_h5.csv US:20:logs/b50_us50_h20.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fintech_agent.evaluation.abtest import compare  # noqa: E402

PAIRS = {"chronos-2-cov": "chronos-2", "lgbm-cov": "lgbm", "timesfm-2.5-xreg": "timesfm-2.5",
         "timesfm-3.0-cov": "timesfm-3.0", "ensemble-cov": "ensemble"}


def per_run(w: pd.DataFrame) -> pd.DataFrame:
    pt = w.pivot_table(index="ticker", columns="model", values="crps_rel", aggfunc="mean")
    skill = 1 - pt.div(pt["naive"], axis=0)
    g = w.groupby("model")
    out = pd.DataFrame({
        "crps_skill_%": (1 - g["crps_rel"].mean() / g["crps_rel"].mean()["naive"]) * 100,
        "median_ticker_skill_%": skill.median() * 100,
        "share_tickers_beating_naive": (skill > 0).mean(),
        "cov80": g["cov80"].mean(),
        "dir_acc": g["dir_hit"].mean(),
        "ic": g[["ret_pred", "ret_true"]].apply(lambda x: x.corr(method="spearman").iloc[0, 1]
                      if x["ret_pred"].std() > 0 else np.nan),
    })
    out["sign_test_p"] = [stats.binomtest(int((skill[m] > 0).sum()), int(skill[m].notna().sum()), 0.5).pvalue
                          if m != "naive" else np.nan for m in out.index]
    return out.drop(index="naive").sort_values("crps_skill_%", ascending=False)


def main() -> None:
    runs = []
    for arg in sys.argv[1:]:
        market, h, path = arg.split(":", 2)
        w = pd.read_csv(path, parse_dates=["origin"])
        runs.append((market, int(h), w))
    for market, h, w in runs:
        n_t, n_w = w["ticker"].nunique(), len(w[w["model"] == "naive"])
        print(f"\n#### {market} · {h} 日 · {n_t} 檔 · {n_w} 個視窗\n")
        t = per_run(w)
        print("| 模型 | CRPS skill | 個股中位數 skill | 勝過 naive 的股票比例 | sign test p | 80% 覆蓋 | 方向準確率 | IC |")
        print("|---|---|---|---|---|---|---|---|")
        for m, r in t.iterrows():
            print(f"| {m} | {r['crps_skill_%']:+.2f}% | {r['median_ticker_skill_%']:+.2f}% | "
                  f"{r['share_tickers_beating_naive']:.0%} | {r['sign_test_p']:.3f} | {r['cov80']:.0%} | "
                  f"{r['dir_acc']:.1%} | {r['ic']:+.3f} |")
        step = 5 if h == 5 else 20
        rows = []
        for c, b in PAIRS.items():
            if c in set(w["model"]) and b in set(w["model"]):
                r = compare(w, b, c, "crps_rel", h, step=step)
                rows.append(f"| {c} vs {b} | {r.improvement_pct:+.2f}% | {r.dm_p:.3f} | {r.ticker_win_rate:.0%} | "
                            f"{ {'promote': '共變數有幫助', 'keep': '共變數變差', 'inconclusive': '無顯著差異'}[r.decision] } |")
        if rows:
            print("\n| 共變數 A/B | CRPS 改善 | DM p | 勝出股票比例 | 結論 |\n|---|---|---|---|---|")
            print("\n".join(rows))


if __name__ == "__main__":
    main()
