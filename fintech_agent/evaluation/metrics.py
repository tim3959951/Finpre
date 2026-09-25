"""Forecast accuracy, probabilistic calibration and trading-strategy metrics."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..forecasting.base import ForecastResult


def window_metrics(actual: np.ndarray, fc: ForecastResult, context: np.ndarray) -> dict:
    """Metrics for one forecast window. `context` is the history the model saw (for MASE scale & last price)."""
    y = np.asarray(actual, float)
    p = np.asarray(fc.point, float)[: len(y)]
    last = float(context[-1])
    err = y - p
    scale = float(np.mean(np.abs(np.diff(context[-252:])))) or 1e-9
    out = {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "mape": float(np.mean(np.abs(err) / np.abs(y)) * 100),
        "smape": float(np.mean(2 * np.abs(err) / (np.abs(y) + np.abs(p))) * 100),
        "mase": float(np.mean(np.abs(err)) / scale),
        "last": last,
        "ret_pred": float(p[-1] / last - 1),
        "ret_true": float(y[-1] / last - 1),
        "abs_sum": float(np.sum(np.abs(y))),
    }
    out["dir_hit"] = (np.nan if out["ret_pred"] == 0 else float(np.sign(out["ret_pred"]) == np.sign(out["ret_true"])))
    if fc.quantiles:
        levels = sorted(fc.quantiles)
        pin = 0.0
        for q in levels:
            e = y - np.asarray(fc.quantiles[q][: len(y)])
            pin += float(np.sum(np.maximum(q * e, (q - 1) * e)))
        out["pinball_sum"] = pin / len(levels)
        out["crps"] = 2 * out["pinball_sum"] / len(y)
        out["crps_rel"] = out["crps"] / last * 100          # scale-free (% of price): comparable across tickers
        lo, hi = fc.q(0.1)[: len(y)], fc.q(0.9)[: len(y)]
        out["cov80"] = float(np.mean((y >= lo) & (y <= hi)))
        out["width80_pct"] = float(np.mean((hi - lo) / y) * 100)
        out["p_up"] = float(fc.prob_above(last))
    return out


def strategy_stats(ret_pred: np.ndarray, ret_true: np.ndarray, horizon: int, cost_bps: float = 0.0,
                   threshold: float = 0.0, allow_short: bool = False) -> dict:
    """Trade each (non-overlapping) window: long if predicted return > threshold (short if < -threshold)."""
    rp, rt = np.asarray(ret_pred), np.asarray(ret_true)
    pos = np.where(rp > threshold, 1.0, np.where((rp < -threshold) & allow_short, -1.0, 0.0))
    cost = cost_bps / 1e4
    strat = pos * rt - np.abs(pos) * cost
    n = len(strat)
    if n == 0:
        return {}
    per_year = 252 / horizon

    def summarize(r: np.ndarray, prefix: str) -> dict:
        eq = np.cumprod(1 + r)
        dd = eq / np.maximum.accumulate(eq) - 1
        sd = r.std(ddof=1) if len(r) > 1 else 0.0
        return {f"{prefix}total_ret": float(eq[-1] - 1),
                f"{prefix}ann_ret": float(eq[-1] ** (per_year / n) - 1) if eq[-1] > 0 else -1.0,
                f"{prefix}sharpe": float(r.mean() / sd * np.sqrt(per_year)) if sd > 0 else 0.0,
                f"{prefix}max_dd": float(dd.min())}

    out = summarize(strat, "strat_")
    out.update(summarize(rt - cost, "bh_"))  # buy & hold each window (same cost basis)
    traded = pos != 0
    out["hit_rate"] = float(np.mean(np.sign(pos[traded]) == np.sign(rt[traded]))) if traded.any() else np.nan
    out["exposure"] = float(traded.mean())
    out["n_trades"] = int(traded.sum())
    return out


def leaderboard(windows: pd.DataFrame, horizon: int, cost_bps: float = 0.0, step: int = 1,
                threshold: float = 0.0, baseline: str = "naive") -> pd.DataFrame:
    """Aggregate per-window rows (columns: model, ticker, origin, metrics...) into a model leaderboard."""
    rows = []
    stride = max(1, int(np.ceil(horizon / max(step, 1))))       # non-overlapping windows for the strategy
    for model, g in windows.groupby("model"):
        r = {"model": model, "n_windows": len(g),
             "mae": g["mae"].mean(), "rmse": g["rmse"].mean(), "mape": g["mape"].mean(),
             "smape": g["smape"].mean(), "mase": g["mase"].mean(),
             "dir_acc": g["dir_hit"].mean(skipna=True)}
        if "pinball_sum" in g and g["pinball_sum"].notna().any():
            r["wql"] = g["pinball_sum"].sum() / g["abs_sum"].sum()
            r["crps"] = g["crps"].mean()
            r["crps_rel"] = g["crps_rel"].mean()
            r["cov80"] = g["cov80"].mean()
            r["width80_pct"] = g["width80_pct"].mean()
        r["ic"] = g[["ret_pred", "ret_true"]].corr(method="spearman").iloc[0, 1] if g["ret_pred"].std() > 0 else np.nan
        strat_parts = []
        for _, gt in g.sort_values("origin").groupby("ticker"):
            sub = gt.iloc[::-1].iloc[::stride].iloc[::-1]       # anchor on the most recent window
            strat_parts.append(strategy_stats(sub["ret_pred"].to_numpy(), sub["ret_true"].to_numpy(), horizon,
                                              cost_bps, threshold))
        if strat_parts:
            sp = pd.DataFrame(strat_parts)
            for c in ("strat_sharpe", "strat_ann_ret", "strat_max_dd", "bh_sharpe", "hit_rate", "exposure"):
                if c in sp:
                    r[c] = sp[c].mean()
        r["sec_per_100"] = g["seconds"].iloc[0] / max(len(g), 1) * 100 if "seconds" in g else np.nan
        rows.append(r)
    lb = pd.DataFrame(rows)
    if baseline in set(lb["model"]):
        base_mae = lb.loc[lb["model"] == baseline, "mae"].iloc[0]
        lb["skill_vs_naive"] = 1 - lb["mae"] / base_mae
        if "wql" in lb:
            base_wql = lb.loc[lb["model"] == baseline, "wql"].iloc[0]
            lb["wql_skill"] = 1 - lb["wql"] / base_wql
            base_c = lb.loc[lb["model"] == baseline, "crps_rel"].iloc[0]
            lb["crps_skill"] = 1 - lb["crps_rel"] / base_c        # probabilistic skill vs random walk (scale-free)
    key = "crps_rel" if "crps_rel" in lb and lb["crps_rel"].notna().all() else "mase"
    return lb.sort_values(key).reset_index(drop=True)
