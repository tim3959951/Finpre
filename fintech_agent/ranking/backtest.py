"""Walk-forward backtest of stock-ranking scorers.

Every `horizon` trading days (from `start`) each scorer ranks the universe using features known at that close.
Portfolios are entered at the next close (lag=1, built into the panel labels) and held `horizon` days, so the
holding periods are contiguous and non-overlapping. Trainable rankers are refit every month (or quarter) on
labels whose exit date is before the month starts — the same no-look-ahead rule as the forecasting backtest.

Per rebalance we record: rank IC (Spearman), mean return of each score quantile, top-N long-only return
(after turnover costs), equal-weight universe return, and top-N minus bottom-N long/short return.
"""
from __future__ import annotations

import logging
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy import stats

from ..evaluation.regimes import episode_of, regime_at
from .model import Scorer

log = logging.getLogger(__name__)


@dataclass
class RankConfig:
    horizon: int = 5
    start: str = "2018-01-01"
    end: str | None = None
    retrain: str = "M"
    train_years: float | None = None     # None = expanding window
    top_n: int = 10
    buffer_rank: int | None = 20         # buffered portfolio: keep a holding while it ranks within this many
    n_quantiles: int = 5
    cost_bps: float = 58.5               # round-trip cost charged on the traded fraction of the book
    train_stride: int = 2                # use every k-th date for training (labels overlap anyway)
    smooth: tuple = ()                   # EMA weights on the previous rebalance's rank, e.g. (0.5, 0.75):
                                         # adds "<scorer>~s0.5" variants that trade less


def rebalance_dates(panel: pd.DataFrame, cfg: RankConfig) -> pd.DatetimeIndex:
    lab = panel["fwd_ret"].notna().groupby(level="date").mean()
    dates = lab.index[lab > 0.8]                               # dates whose forward returns are known
    dates = dates[dates >= pd.Timestamp(cfg.start)]
    if cfg.end:
        dates = dates[dates <= pd.Timestamp(cfg.end)]
    all_dates = panel.index.get_level_values("date").unique().sort_values()
    pos = all_dates.get_indexer(dates)
    keep, last = [], -10 ** 9
    for d, p in zip(dates, pos):                               # every `horizon` trading days
        if p - last >= cfg.horizon:
            keep.append(d)
            last = p
    return pd.DatetimeIndex(keep)


def _score_trainable(scorer, panel: pd.DataFrame, rebal: pd.DatetimeIndex, cfg: RankConfig, progress=None):
    """Monthly refits -> scores for the rows of every rebalance date."""
    rows_date = panel.index.get_level_values("date")
    fwd_end = pd.to_datetime(panel["fwd_end"]).to_numpy(dtype="datetime64[ns]")
    all_dates = rows_date.unique().sort_values()
    stride_dates = set(all_dates[::max(1, cfg.train_stride)])
    in_stride = np.asarray(rows_date.isin(stride_dates))
    labelled = panel["fwd_rank"].notna().to_numpy()
    scores = {}
    periods = rebal.to_period(cfg.retrain)
    uniq = sorted(set(periods))
    for j, p in enumerate(uniq):
        cutoff = np.datetime64(p.start_time, "ns")
        sel = labelled & in_stride & (fwd_end < cutoff)
        if cfg.train_years:
            sel &= np.asarray(rows_date >= p.start_time - pd.Timedelta(days=int(cfg.train_years * 365.25)))
        scorer.fit(panel[sel])
        for d in rebal[np.asarray(periods == p)]:
            x = panel.loc[d]
            scores[d] = pd.Series(scorer.score(x), index=x.index)
        if progress and (j % 12 == 0 or j == len(uniq) - 1):
            progress(f"{scorer.name}: refit {j + 1}/{len(uniq)} ({p}) on {int(sel.sum()):,} rows")
    return scores


def rank_backtest(panel: pd.DataFrame, scorers: list[Scorer], cfg: RankConfig, regime: pd.Series | None = None,
                  market: str = "TW", progress=None) -> pd.DataFrame:
    """Per (scorer, rebalance date) rows."""
    rebal = rebalance_dates(panel, cfg)
    if len(rebal) < 3:
        raise ValueError("not enough rebalance dates")
    reg = regime_at(regime, rebal) if regime is not None else np.full(len(rebal), None)
    epi = episode_of(rebal, market)
    cost = cfg.cost_bps / 1e4
    out = []
    variants: list[tuple[str, dict]] = []
    for sc in scorers:
        if getattr(sc, "trainable", False):
            scores = _score_trainable(sc, panel, rebal, cfg, progress)
        else:
            scores = {d: pd.Series(sc.score(panel.loc[d]), index=panel.loc[d].index) for d in rebal}
        variants.append((sc.name, scores))
        for a in cfg.smooth:
            variants.append((f"{sc.name}~s{a:g}", smooth_scores(scores, rebal, a)))
    for name, scores in variants:
        prev_top, prev_bot, held = set(), set(), []
        for i, d in enumerate(rebal):
            x = panel.loc[d, ["fwd_ret"]].copy()
            x["score"] = scores[d]
            x = x.dropna()
            if len(x) < cfg.top_n * 2:
                continue
            simple = np.expm1(x["fwd_ret"])
            ic = float(stats.spearmanr(x["score"], x["fwd_ret"])[0])
            q = pd.qcut(x["score"].rank(method="first"), cfg.n_quantiles, labels=False)
            qret = simple.groupby(q).mean()
            order = x["score"].sort_values(ascending=False).index
            top, bot = set(order[:cfg.top_n]), set(order[-cfg.top_n:])
            t_top = 1.0 if not prev_top else len(top - prev_top) / cfg.top_n
            t_bot = 1.0 if not prev_bot else len(bot - prev_bot) / cfg.top_n
            r_top, r_bot, r_ew = simple[list(top)].mean(), simple[list(bot)].mean(), simple.mean()
            row = {"scorer": name, "date": d, "n": len(x), "ic": ic, "top": r_top, "bottom": r_bot, "ew": r_ew,
                   "turnover": t_top, "top_net": r_top - t_top * cost,
                   "ls_net": (r_top - r_bot) - (t_top + t_bot) * cost, "regime": reg[i], "episode": epi[i],
                   "top_names": ",".join(sorted(map(str, top)))}
            for k in range(cfg.n_quantiles):
                row[f"q{k + 1}"] = qret.get(k, np.nan)
            if cfg.buffer_rank:
                # turnover control: keep names still ranked within `buffer_rank`, refill from the top
                rank = {t: k for k, t in enumerate(order)}
                keep = [t for t in held if rank.get(t, 10 ** 9) < cfg.buffer_rank]
                new = [t for t in order if t not in keep][:cfg.top_n - len(keep)]
                book = keep + new
                t_buf = 1.0 if not held else len(set(book) - set(held)) / cfg.top_n
                r_buf = simple[book].mean()
                row.update(buf=r_buf, buf_turnover=t_buf, buf_net=r_buf - t_buf * cost)
                held = book
            out.append(row)
            prev_top, prev_bot = top, bot
    return pd.DataFrame(out)


def smooth_scores(scores: dict, rebal, alpha: float) -> dict:
    """EMA of each stock's cross-sectional rank across rebalances: yesterday's view keeps weight `alpha`.
    Ranks (not raw scores) are smoothed because monthly refits change the score scale."""
    out, prev = {}, None
    for d in rebal:
        r = scores[d].rank(pct=True)
        if prev is not None:
            p = prev.reindex(r.index)
            r = (1 - alpha) * r + alpha * p.fillna(r)
        out[d] = r
        prev = r
    return out


def _drawdown(r: pd.Series) -> float:
    eq = (1 + r).cumprod()
    return float((eq / eq.cummax() - 1).min())


def summarize(per: pd.DataFrame, cfg: RankConfig) -> pd.DataFrame:
    ppy = 252 / cfg.horizon
    rows = []
    for name, g in per.groupby("scorer", sort=False):
        g = g.sort_values("date")
        n = len(g)
        ic = g["ic"]
        ex = g["top_net"] - g["ew"]
        ann = lambda r: float((1 + r).prod() ** (ppy / len(r)) - 1)        # noqa: E731
        sharpe = lambda r: float(r.mean() / (r.std() + 1e-12) * np.sqrt(ppy))  # noqa: E731
        rows.append({
            "scorer": name, "periods": n, "ic_mean": ic.mean(), "ic_std": ic.std(),
            "icir_ann": ic.mean() / (ic.std() + 1e-12) * np.sqrt(ppy), "ic_t": ic.mean() / (ic.std() + 1e-12) * np.sqrt(n),
            "ic_pos": float((ic > 0).mean()),
            "top_ann_net": ann(g["top_net"]), "top_sharpe_net": sharpe(g["top_net"]), "top_maxdd": _drawdown(g["top_net"]),
            "ew_ann": ann(g["ew"]), "ew_sharpe": sharpe(g["ew"]), "ew_maxdd": _drawdown(g["ew"]),
            "excess_ann": float(ex.mean() * ppy), "info_ratio": sharpe(ex), "hit_vs_ew": float((ex > 0).mean()),
            "ls_ann_net": ann(g["ls_net"]), "ls_sharpe_net": sharpe(g["ls_net"]), "turnover": g["turnover"].mean(),
            "top_ann_gross": ann(g["top"]), "excess_ann_gross": float((g["top"] - g["ew"]).mean() * ppy),
            **({"buf_ann_net": ann(g["buf_net"]), "buf_sharpe_net": sharpe(g["buf_net"]),
                "buf_excess_ann": float((g["buf_net"] - g["ew"]).mean() * ppy),
                "buf_info_ratio": sharpe(g["buf_net"] - g["ew"]), "buf_maxdd": _drawdown(g["buf_net"]),
                "buf_turnover": g["buf_turnover"].mean()} if "buf_net" in g else {}),
            **{f"q{k + 1}_ann": float(g[f"q{k + 1}"].mean() * ppy) for k in range(cfg.n_quantiles) if f"q{k + 1}" in g},
        })
    return pd.DataFrame(rows).sort_values("ic_mean", ascending=False)


def breakdown(per: pd.DataFrame, cfg: RankConfig, by: str) -> pd.DataFrame:
    """IC and excess return of the top-N portfolio by year / regime / episode."""
    ppy = 252 / cfg.horizon
    d = per.copy()
    if by == "year":
        d["year"] = pd.to_datetime(d["date"]).dt.year
    d = d.dropna(subset=[by])
    d["ex"] = d["top_net"] - d["ew"]
    d["bex"] = (d["buf_net"] - d["ew"]) if "buf_net" in d else np.nan
    g = d.groupby(["scorer", by])
    return pd.DataFrame({
        "periods": g.size(), "ic_mean": g["ic"].mean(),
        "ic_t": g["ic"].mean() / (g["ic"].std() + 1e-12) * np.sqrt(g.size()),
        "excess_ann": g["ex"].mean() * ppy, "buf_excess_ann": g["bex"].mean() * ppy, "ew_ann": g["ew"].mean() * ppy,
    }).reset_index()


def config_dict(cfg: RankConfig) -> dict:
    return asdict(cfg)
