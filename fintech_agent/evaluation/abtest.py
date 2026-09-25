"""Statistical A/B testing between forecasting models (champion vs challenger).

Offline: paired tests on backtest windows (Diebold-Mariano w/ HLN correction, paired bootstrap, sign test).
Online : the same tests on shadow-mode predictions once actuals arrive (see experiments.py).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy import stats


def diebold_mariano(loss_a: np.ndarray, loss_b: np.ndarray, h: int = 1) -> tuple[float, float]:
    """DM test of equal predictive accuracy. d = loss_a - loss_b; positive stat => B is more accurate.
    Harvey-Leybourne-Newbold small-sample correction, Newey-West variance with h-1 lags.
    Returns (statistic, two-sided p-value)."""
    d = np.asarray(loss_a, float) - np.asarray(loss_b, float)
    d = d[np.isfinite(d)]
    n = len(d)
    if n < 5:
        return 0.0, 1.0
    if np.allclose(d, d[0]):  # zero variance: identical models, or one uniformly better
        return (0.0, 1.0) if abs(d[0]) < 1e-12 else (float(np.sign(d[0]) * 1e3), 0.0)
    dbar = d.mean()
    h = int(max(1, min(h, n - 1)))
    gamma = [np.sum((d[k:] - dbar) * (d[: n - k] - dbar)) / n for k in range(h)]
    var = (gamma[0] + 2 * sum(gamma[1:])) / n
    if var <= 0:
        var = gamma[0] / n
    dm = dbar / np.sqrt(var)
    hln = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    stat = float(dm * hln)
    p = float(2 * stats.t.sf(abs(stat), df=n - 1))
    return stat, p


def paired_bootstrap(loss_a: np.ndarray, loss_b: np.ndarray, n_boot: int = 2000, block: int = 5,
                     seed: int = 0) -> dict:
    """Moving-block bootstrap of mean(loss_a - loss_b). Returns CI and P(B better)."""
    d = np.asarray(loss_a, float) - np.asarray(loss_b, float)
    d = d[np.isfinite(d)]
    n = len(d)
    if n < 5:
        return {"mean_diff": float(d.mean()) if n else 0.0, "ci95": (np.nan, np.nan), "p_b_better": 0.5}
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(n / block))
    starts = rng.integers(0, max(1, n - block + 1), size=(n_boot, nb))
    idx = (starts[:, :, None] + np.arange(block)).reshape(n_boot, -1)[:, :n] % n
    means = d[idx].mean(axis=1)
    return {"mean_diff": float(d.mean()), "ci95": (float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))),
            "p_b_better": float((means > 0).mean())}


@dataclass
class ABResult:
    champion: str
    challenger: str
    metric: str
    n: int
    champion_loss: float
    challenger_loss: float
    improvement_pct: float
    dm_stat: float
    dm_p: float
    boot_ci95: tuple
    p_challenger_better: float
    ticker_win_rate: float
    dir_acc_champion: float
    dir_acc_challenger: float
    decision: str               # promote | keep | inconclusive
    reason: str

    def to_dict(self) -> dict:
        return asdict(self)


def compare(windows: pd.DataFrame, champion: str, challenger: str, metric: str = "crps_rel", horizon: int = 5,
            alpha: float = 0.05, min_win_rate: float = 0.55, step: int | None = None) -> ABResult:
    """Paired comparison on common (ticker, origin) windows.

    Forecast errors of consecutive windows are autocorrelated only when windows overlap, so the DM
    Newey-West lag order is ceil(horizon / step) (= 1, i.e. no extra lags, for non-overlapping windows)."""
    if step is None:                      # infer the window spacing (business days) per ticker
        gaps = []
        for _, g in windows[windows["model"] == champion].groupby("ticker"):
            try:
                o = np.array(sorted(pd.to_datetime(g["origin"].unique())), dtype="datetime64[D]")
            except (TypeError, ValueError):
                continue
            if len(o) > 1:
                gaps.append(float(np.median(np.busday_count(o[:-1], o[1:]))))
        step = int(round(np.median(gaps))) if gaps else horizon
    lag_h = int(np.ceil(horizon / max(step or horizon, 1)))
    a = windows[windows["model"] == champion].set_index(["ticker", "origin"])
    b = windows[windows["model"] == challenger].set_index(["ticker", "origin"])
    common = a.index.intersection(b.index)
    if len(common) < 5:
        raise ValueError(f"not enough paired windows between {champion} and {challenger}")
    a, b = a.loc[common].sort_index(), b.loc[common].sort_index()
    la, lb_ = a[metric].to_numpy(), b[metric].to_numpy()
    # DM per ticker (keeps the time ordering intact), combined with Stouffer's method
    zs, wins = [], []
    for t in common.get_level_values(0).unique():
        s, p = diebold_mariano(a.loc[t][metric].to_numpy(), b.loc[t][metric].to_numpy(), h=lag_h)
        zs.append(np.sign(s) * stats.norm.isf(p / 2) if p < 1 else 0.0)
        wins.append(float(b.loc[t][metric].mean() < a.loc[t][metric].mean()))
    z = float(np.sum(zs) / np.sqrt(len(zs)))
    p_comb = float(2 * stats.norm.sf(abs(z)))
    boot = paired_bootstrap(la, lb_)
    impr = float((la.mean() - lb_.mean()) / la.mean() * 100) if la.mean() else 0.0
    win_rate = float(np.mean(wins))
    if z > 0 and p_comb < alpha and win_rate >= min_win_rate:
        decision, reason = "promote", f"{challenger} 顯著較佳 (p={p_comb:.3f}, 勝率 {win_rate:.0%})"
    elif z < 0 and p_comb < alpha:
        decision, reason = "keep", f"{champion} 顯著較佳 (p={p_comb:.3f})"
    else:
        decision, reason = "inconclusive", f"差異不顯著 (p={p_comb:.3f})，維持 {champion}"
    return ABResult(champion, challenger, metric, len(common), float(la.mean()), float(lb_.mean()), impr, z, p_comb,
                    boot["ci95"], boot["p_b_better"], win_rate, float(a["dir_hit"].mean()), float(b["dir_hit"].mean()),
                    decision, reason)


def direction_significance(hits: np.ndarray) -> dict:
    """Binomial test: is directional accuracy better than a coin flip?"""
    h = np.asarray(hits, float)
    h = h[np.isfinite(h)]
    if len(h) == 0:
        return {"n": 0, "acc": np.nan, "p_value": 1.0}
    k = int(h.sum())
    return {"n": len(h), "acc": k / len(h), "p_value": float(stats.binomtest(k, len(h), 0.5, alternative="greater").pvalue)}
