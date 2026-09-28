#!/usr/bin/env python
"""Static PNG charts of every benchmark (for the repo / GitHub), drawn from export_benchmark_charts.py's JSON.

  python scripts/plot_benchmarks.py --data docs/benchmarks/v0.3/chart_data.json --out docs/benchmarks/v0.3
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

INK, MUTED, GRID, SURFACE = "#0b0b0b", "#898781", "#e1e0d9", "#fcfcfb"
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
MODEL_COLOR = {"lgbm-cov": SLOTS[0], "lgbm": SLOTS[1], "timesfm-2.5": SLOTS[2], "chronos-2": SLOTS[3],
               "timesfm-3.0": SLOTS[4], "drift": SLOTS[6], "ensemble-cov": SLOTS[5], "ensemble": SLOTS[7]}
SCORER_COLOR = {"xs-lgbm-chips": SLOTS[0], "xs-lgbm": SLOTS[1], "momentum": SLOTS[2], "foreign_flow": SLOTS[3],
                "trust_flow": SLOTS[4], "reversal": SLOTS[5], "low_vol": SLOTS[6], "ew": MUTED}
NAMES = {"xs-lgbm-chips": "LightGBM 排序＋籌碼", "xs-lgbm": "LightGBM 排序", "momentum": "動能 (6-1 月)",
         "foreign_flow": "外資 20 日買超", "trust_flow": "投信 20 日買超", "reversal": "週反轉", "low_vol": "低波動",
         "ew": "等權持有全部"}
REGIME = {"bull": "多頭", "bear": "空頭", "mixed": "盤整/轉折"}
LONG_MODELS = ["lgbm-cov", "lgbm", "timesfm-2.5", "chronos-2", "drift"]

plt.rcParams.update({
    "font.sans-serif": ["PingFang TC", "Heiti TC", "Arial Unicode MS", "Noto Sans CJK TC", "Microsoft JhengHei",
                        "DejaVu Sans"],
    "axes.unicode_minus": False, "axes.edgecolor": GRID, "axes.labelcolor": INK, "axes.titlecolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "font.size": 10, "axes.titlesize": 11, "legend.frameon": False})


def _panel_title(r):
    return f"{'台股' if r['market'] == 'TW' else '美股'} {r['horizon']} 日"


def post_release(data, out: Path):
    runs = data["post_release"]
    fig, axes = plt.subplots(1, len(runs), figsize=(4.2 * len(runs), 5.2), sharey=False)
    for ax, r in zip(np.atleast_1d(axes), runs):
        ms = r["models"][::-1]
        vals = [m["crps_skill_pct"] for m in ms]
        cols = [MODEL_COLOR.get(m["model"], "#c3c2b7") for m in ms]
        ax.barh(range(len(ms)), vals, color=cols, height=0.6)
        ax.set_yticks(range(len(ms)), [m["model"] for m in ms], fontsize=8)
        ax.axvline(0, color=INK, lw=0.8)
        ax.set_title(_panel_title(r))
        ax.set_xlabel("CRPS skill vs random walk (%)")
    fig.suptitle("發布後測試 2025-10 → 2026-09（50 檔）：正值 = 勝過 random walk", color=INK)
    fig.tight_layout()
    fig.savefig(out / "01_post_release_crps_skill.png", dpi=150)
    plt.close(fig)


def long_by(data, out: Path, key: str, fname: str, title: str):
    runs = data["long_history"]
    fig, axes = plt.subplots(1, len(runs), figsize=(4.4 * len(runs), 4.2), sharey=True)
    for ax, r in zip(np.atleast_1d(axes), runs):
        df = pd.DataFrame(r[f"by_{key}"])
        if df.empty:
            continue
        pv = df.pivot(index=key, columns="model", values="crps_skill_pct")
        models = [m for m in LONG_MODELS if m in pv]
        if key == "year":
            for m in models:
                ax.plot(pv.index, pv[m], color=MODEL_COLOR[m], lw=2, marker="o", ms=4, label=m)
            ax.set_xticks(pv.index, [str(y)[2:] for y in pv.index])
        else:
            order = [k for k in (["bull", "mixed", "bear"] if key == "regime" else pv.index) if k in pv.index]
            x = np.arange(len(order))
            wd = 0.8 / len(models)
            for i, m in enumerate(models):
                ax.bar(x + i * wd - 0.4 + wd / 2, pv.loc[order, m], width=wd * 0.9, color=MODEL_COLOR[m], label=m)
            ax.set_xticks(x, [REGIME.get(k, k) for k in order], fontsize=8, rotation=0 if key == "regime" else 20)
        ax.axhline(0, color=INK, lw=0.8)
        ax.set_title(_panel_title(r))
    np.atleast_1d(axes)[0].set_ylabel("CRPS skill vs random walk (%)")
    np.atleast_1d(axes)[-1].legend(fontsize=8, loc="lower left")
    fig.suptitle(title, color=INK)
    fig.tight_layout()
    fig.savefig(out / fname, dpi=150)
    plt.close(fig)


def long_rolling(data, out: Path):
    runs = data["long_history"]
    fig, axes = plt.subplots(len(runs), 1, figsize=(11, 2.6 * len(runs)), sharex=True)
    for ax, r in zip(np.atleast_1d(axes), runs):
        df = pd.DataFrame(r["rolling_12m"])
        if df.empty:
            continue
        x = pd.PeriodIndex(df["month"], freq="M").to_timestamp()
        for m in LONG_MODELS:
            if m in df:
                ax.plot(x, df[m], color=MODEL_COLOR[m], lw=2 if m.startswith("lgbm") else 1.4, label=m)
        ax.axhline(0, color=INK, lw=0.8)
        ax.set_ylabel(_panel_title(r))
    np.atleast_1d(axes)[0].legend(ncol=5, fontsize=8, loc="upper left")
    fig.suptitle("滾動 12 個月 CRPS skill vs random walk（%）", color=INK)
    fig.tight_layout()
    fig.savefig(out / "04_long_rolling_skill.png", dpi=150)
    plt.close(fig)


def ranking_equity(data, out: Path):
    runs = data["ranking"]
    fig, axes = plt.subplots((len(runs) + 1) // 2, 2, figsize=(12, 3.6 * ((len(runs) + 1) // 2)))
    for ax, r in zip(np.ravel(axes), runs):
        ew = r["ew"]
        ax.plot(pd.to_datetime(ew["date"]), ew["ew"], color=MUTED, lw=1.6, label=NAMES["ew"])
        for sc in ("xs-lgbm-chips", "xs-lgbm", "momentum"):
            e = r["equity"].get(sc)
            if not e:
                continue
            series = e.get("buffer") or e["top"]
            ax.plot(pd.to_datetime(e["date"]), series, color=SCORER_COLOR[sc], lw=2, label=f"{NAMES[sc]}（前 10，扣成本）")
        ax.set_yscale("log")
        ax.set_title(f"{r['label']} · {r['horizon']} 日換股")
    np.ravel(axes)[0].legend(fontsize=8, loc="upper left")
    fig.suptitle("選股排序：前 10 名組合淨值（對數座標，已扣交易成本）", color=INK)
    fig.tight_layout()
    fig.savefig(out / "05_ranking_equity.png", dpi=150)
    plt.close(fig)


def ranking_ic_year(data, out: Path):
    runs = [r for r in data["ranking"] if r["market"] == "TW"]
    fig, axes = plt.subplots(1, len(runs), figsize=(4.4 * len(runs), 4), sharey=True)
    for ax, r in zip(np.atleast_1d(axes), runs):
        df = pd.DataFrame(r["by_year"])
        pv = df.pivot(index="year", columns="scorer", values="ic_mean")
        scs = [s for s in ("xs-lgbm-chips", "xs-lgbm", "momentum", "foreign_flow") if s in pv]
        x = np.arange(len(pv.index))
        wd = 0.8 / len(scs)
        for i, sc in enumerate(scs):
            ax.bar(x + i * wd - 0.4 + wd / 2, pv[sc], width=wd * 0.9, color=SCORER_COLOR[sc], label=NAMES[sc])
        ax.set_xticks(x, [str(y)[2:] for y in pv.index])
        ax.axhline(0, color=INK, lw=0.8)
        ax.set_title(f"{r['label']} · {r['horizon']} 日")
    np.atleast_1d(axes)[0].set_ylabel("平均 rank IC")
    np.atleast_1d(axes)[-1].legend(fontsize=8)
    fig.suptitle("台股選股排序：各年度 rank IC", color=INK)
    fig.tight_layout()
    fig.savefig(out / "06_ranking_ic_by_year.png", dpi=150)
    plt.close(fig)


def ranking_quintiles(data, out: Path):
    runs = data["ranking"]
    fig, axes = plt.subplots(1, len(runs), figsize=(3.4 * len(runs), 3.6), sharey=False)
    for ax, r in zip(np.atleast_1d(axes), runs):
        row = next((s for s in r["summary"] if s["scorer"] == ("xs-lgbm-chips" if r["market"] == "TW" else "xs-lgbm")), None)
        if not row:
            continue
        q = [row.get(f"q{k}_ann", np.nan) * 100 for k in range(1, 6)]
        ax.bar(range(1, 6), q, color=SLOTS[0], width=0.55)
        ax.set_xticks(range(1, 6), ["Q1\n最差", "Q2", "Q3", "Q4", "Q5\n最佳"], fontsize=8)
        ax.set_title(f"{r['label']} · {r['horizon']} 日", fontsize=9)
    np.atleast_1d(axes)[0].set_ylabel("年化報酬 %（未扣成本）")
    fig.suptitle("依模型分數分五組的平均年化報酬（單調遞增 = 排序有效）", color=INK)
    fig.tight_layout()
    fig.savefig(out / "07_ranking_quintiles.png", dpi=150)
    plt.close(fig)


def ranking_survivorship(data, out: Path):
    runs = data["ranking"]
    hs = sorted({r["horizon"] for r in runs if r["market"] == "TW"})
    fig, axes = plt.subplots(1, len(hs), figsize=(6 * len(hs), 3.8), sharey=True)
    for ax, h in zip(np.atleast_1d(axes), hs):
        pit = next((r for r in runs if r["label"] == "twse-pit50" and r["horizon"] == h), None)
        fix = next((r for r in runs if r["label"] == "tw50" and r["horizon"] == h), None)
        if not (pit and fix):
            continue
        scs = [q for q in ("xs-lgbm-chips", "xs-lgbm", "momentum", "foreign_flow", "low_vol")
               if any(x["scorer"] == q for x in pit["summary"])]
        ic = lambda run, q: next(x["ic_mean"] for x in run["summary"] if x["scorer"] == q)   # noqa: E731
        x = np.arange(len(scs))
        ax.bar(x - 0.2, [ic(fix, q) for q in scs], width=0.38, color="#a3a9b2", label="現行台灣 50（存活者偏差）")
        ax.bar(x + 0.2, [ic(pit, q) for q in scs], width=0.38, color=SLOTS[0], label="上市流動性前 50（逐日選取）")
        ax.set_xticks(x, [NAMES[q] for q in scs], fontsize=8)
        ax.axhline(0, color=INK, lw=0.8)
        ax.set_title(f"{h} 日換股")
    np.atleast_1d(axes)[0].set_ylabel("平均 rank IC")
    np.atleast_1d(axes)[0].legend(fontsize=8)
    fig.suptitle("同一方法、兩種股票池：用今天的權值股回測會高估動能", color=INK)
    fig.tight_layout()
    fig.savefig(out / "09_ranking_survivorship.png", dpi=150)
    plt.close(fig)


def ranking_cost(data, out: Path):
    runs = [r for r in data["ranking"] if r["label"] == "twse-pit50"]
    fig, axes = plt.subplots(1, len(runs), figsize=(6 * len(runs), 3.8), sharey=True)
    for ax, r in zip(np.atleast_1d(axes), runs):
        cs = r.get("cost_sensitivity") or {}
        for q in ("xs-lgbm-chips", "xs-lgbm", "momentum", "foreign_flow"):
            if q in cs.get("net_excess", {}):
                ax.plot(cs["cost_bps"], np.array(cs["net_excess"][q], dtype=float) * 100, color=SCORER_COLOR[q], lw=2,
                        label=NAMES[q])
        for v, lab in ((58.5, "牌價"), (38, "手續費 2.8 折")):
            ax.axvline(v, color=MUTED, lw=1)
            ax.text(v + 0.8, ax.get_ylim()[1] * 0.9, lab, fontsize=8, color=MUTED)
        ax.axhline(0, color=INK, lw=0.8)
        ax.set_xlabel("來回交易成本（bps）")
        ax.set_title(f"上市流動性前 50 · {r['horizon']} 日換股（緩衝）")
    np.atleast_1d(axes)[0].set_ylabel("前 10 名相對等權的年化超額 %")
    np.atleast_1d(axes)[0].legend(fontsize=8)
    fig.suptitle("交易成本敏感度", color=INK)
    fig.tight_layout()
    fig.savefig(out / "10_ranking_cost_sensitivity.png", dpi=150)
    plt.close(fig)


def covariate_ab(data, out: Path):
    rows = []
    for r in data["post_release"]:
        for c in r["covariate_ab"]:
            rows.append({"run": _panel_title(r), **c})
    df = pd.DataFrame(rows)
    if df.empty:
        return
    pairs = list(dict.fromkeys(df["challenger"]))
    runs = list(dict.fromkeys(df["run"]))
    fig, ax = plt.subplots(figsize=(10, 4))
    wd = 0.8 / len(runs)
    for i, run in enumerate(runs):
        d = df[df["run"] == run].set_index("challenger").reindex(pairs)
        ax.bar(np.arange(len(pairs)) + i * wd - 0.4 + wd / 2, d["improvement_pct"], width=wd * 0.9,
               color=SLOTS[i], label=run)
    ax.set_xticks(range(len(pairs)), [f"{p}\nvs 無共變數" for p in pairs], fontsize=8)
    ax.axhline(0, color=INK, lw=0.8)
    ax.set_ylabel("CRPS 改善 %")
    ax.legend(fontsize=8, ncol=4)
    ax.set_title("加入籌碼／大盤／匯率共變數的效果（正值 = 有幫助）", color=INK)
    fig.tight_layout()
    fig.savefig(out / "08_covariate_ab.png", dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="docs/benchmarks/v0.3/chart_data.json")
    ap.add_argument("--out", default="docs/benchmarks/v0.3")
    args = ap.parse_args()
    data = json.loads(Path(args.data).read_text())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    post_release(data, out)
    long_by(data, out, "year", "02_long_skill_by_year.png", "2018 → 2026 各年度 CRPS skill（LightGBM 每月重訓）")
    long_by(data, out, "regime", "03_long_skill_by_regime.png", "依市場狀態（大盤 200 日均線 + 60 日動能）")
    long_rolling(data, out)
    ranking_equity(data, out)
    ranking_ic_year(data, out)
    ranking_quintiles(data, out)
    covariate_ab(data, out)
    ranking_survivorship(data, out)
    ranking_cost(data, out)
    print("charts →", out)


if __name__ == "__main__":
    main()
