#!/usr/bin/env python
"""Charts for the decision-model benchmark (reads logs/dm_{universe}_h5_*.{json,csv}).

  python scripts/plot_decision_bench.py            # → docs/benchmarks/decision_model/*.png + summary.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LOGS, OUT = ROOT / "logs", ROOT / "docs" / "benchmarks" / "decision_model"
DM = {"dm_up": "Decision model: up? (yes/no)", "dm_move": "Decision model: rise/flat/fall",
      "dm_outlook": "Decision model: 5-level outlook"}
SHOW = ["dm_up", "dm_move", "dm_outlook", "lgbm-cov", "lgbm", "lgbm-monthly", "lgbm-cov-monthly", "timesfm-2.5",
        "chronos-2", "ensemble", "drift", "naive", "base_rate"]
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID, MUTED = "#0b0b0b", "#52514e", "#e6e5e0", "#a9a8a1"


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def ic_chart(lbs: dict[str, pd.DataFrame], path: Path) -> None:
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(lbs), figsize=(5.2 * len(lbs), 5.4), sharey=False)
    axes = np.atleast_1d(axes)
    for ax, (uni, lb) in zip(axes, lbs.items()):
        d = lb[lb["model"].isin(SHOW) & lb["ic"].notna()].copy()
        d["order"] = d["model"].map({m: i for i, m in enumerate(SHOW)})
        d = d.sort_values("order", ascending=False)
        y = np.arange(len(d))
        colors = [BLUE if m.startswith("dm_") else MUTED for m in d["model"]]
        ax.barh(y, d["ic"], color=colors, height=0.6)
        ax.errorbar(d["ic"], y, xerr=[d["ic"] - d["ic_lo"], d["ic_hi"] - d["ic"]], fmt="none", ecolor=INK2,
                    elinewidth=1, capsize=2)
        ax.axvline(0, color=INK2, linewidth=1)
        ax.set_yticks(y, [DM.get(m, m) for m in d["model"]], fontsize=9, color=INK)
        ax.set_title(f"{uni.upper()} · 5-day rank IC, 95% CI", fontsize=10, color=INK, loc="left")
        style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def reliability(rows: dict[str, pd.DataFrame], path: Path) -> None:
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(rows), figsize=(5.0 * len(rows), 4.6))
    axes = np.atleast_1d(axes)
    for ax, (uni, df) in zip(axes, rows.items()):
        y = (df["ret_true"] > 0).astype(float)
        best = "lgbm-cov-monthly" if "lgbm-cov-monthly" in df else "lgbm-cov"
        for col, color, label in (("dm_up", BLUE, DM["dm_up"]), (best, ORANGE, best),
                                  ("timesfm-2.5", AQUA, "timesfm-2.5")):
            if col not in df:
                continue
            p = df[col]
            bins = pd.cut(p, np.linspace(0, 1, 11), include_lowest=True)
            g = pd.DataFrame({"p": p, "y": y}).groupby(bins, observed=True).agg(p=("p", "mean"), y=("y", "mean"),
                                                                                 n=("y", "size"))
            g = g[g["n"] >= 20]
            ax.plot(g["p"], g["y"], color=color, linewidth=2, marker="o", markersize=5, label=label)
        ax.plot([0, 1], [0, 1], color=MUTED, linewidth=1, linestyle="--")
        ax.axhline(y.mean(), color=GRID, linewidth=1)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("predicted probability of a rise", fontsize=9, color=INK2)
        ax.set_ylabel("observed share of rises", fontsize=9, color=INK2)
        ax.set_title(f"{uni.upper()} · calibration (bins ≥20 windows)", fontsize=10, color=INK, loc="left")
        ax.legend(frameon=False, fontsize=8)
        style(ax)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def news_chart(news: dict, path: Path) -> None:
    """Rank IC (95% cluster-bootstrap CI) of each headline scorer vs the abnormal return, per market and window."""
    import matplotlib.pyplot as plt
    names = {"dm": "Decision model", "finbert": "FinBERT", "lexicon": "Keyword lexicon"}
    targets = {"reaction_abn": "Reaction (to next close)", "drift_abn": "Next session (tradable)"}
    markets = [m for m in ("US", "TW") if m in news.get("markets", {})]
    fig, axes = plt.subplots(1, len(markets), figsize=(5.2 * len(markets), 3.6))
    axes = np.atleast_1d(axes)
    for ax, mk in zip(axes, markets):
        res = pd.DataFrame(news["markets"][mk]["results"])
        res = res[(res["subset"] == "all") & res["ic"].notna()]
        scorers = [s for s in ("dm", "finbert", "lexicon") if s in set(res["scorer"])]
        y = np.arange(len(scorers))
        for j, (tgt, color) in enumerate((("reaction_abn", BLUE), ("drift_abn", MUTED))):
            d = res[res["target"] == tgt].set_index("scorer").reindex(scorers)
            off = y + (0.18 if j == 0 else -0.18)
            ax.barh(off, d["ic"], height=0.34, color=color, label=targets[tgt])
            ax.errorbar(d["ic"], off, xerr=[d["ic"] - d["ic_lo"], d["ic_hi"] - d["ic"]], fmt="none", ecolor=INK2,
                        elinewidth=1, capsize=2)
        ax.axvline(0, color=INK2, linewidth=1)
        ax.set_yticks(y, [names[s] for s in scorers], fontsize=9, color=INK)
        n = news["markets"][mk]["n_headlines"]
        ax.set_title(f"{mk} · {n} headlines · rank IC vs abnormal return", fontsize=10, color=INK, loc="left")
        ax.legend(frameon=False, fontsize=8, loc="lower right")
        style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    lbs, rows, summary = {}, {}, {}
    for uni in ("tw50", "us50"):
        m = LOGS / f"dm_{uni}_h5_metrics.json"
        if not m.exists():
            continue
        data = json.loads(m.read_text(encoding="utf-8"))
        lbs[uni] = pd.DataFrame(data["leaderboard"])
        rows[uni] = pd.read_csv(LOGS / f"dm_{uni}_h5_rows.csv")
        summary[uni] = data
    news = LOGS / "news_bench_metrics.json"
    if news.exists():
        summary["news"] = json.loads(news.read_text(encoding="utf-8"))
        if summary["news"].get("markets"):
            news_chart(summary["news"], OUT / "news_ic.png")
    if not lbs:
        sys.exit("no results yet")
    ic_chart(lbs, OUT / "updown_ic.png")
    reliability(rows, OUT / "updown_calibration.png")
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
