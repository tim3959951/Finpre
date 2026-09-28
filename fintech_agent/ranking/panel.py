"""Cross-sectional feature panel for stock ranking (選股排序).

One row per (date, ticker). Every feature on date d uses information available at the close of d; the label is
the forward return from the close of d+lag to the close of d+lag+h (lag=1: a signal computed after the close is
traded at the next close, so the backtest never assumes you can trade at the price that produced the signal).

Stock-level features are converted to cross-sectional percentiles per date (robust to regime shifts in scale);
market-level context (index trend / volatility) is kept raw so trees can learn regime interactions.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

PRICE_FEATURES = ["r1", "r5", "r20", "r60", "r120", "mom_120_20", "vol20", "vol60", "vol_ratio", "dist_ma20",
                  "dist_ma60", "dist_ma200", "rsi14", "pos60", "max_r20", "vol_z", "vol_trend", "beta60", "idio_vol60"]
CHIP_FEATURES = ["foreign_5", "foreign_20", "foreign_60", "trust_5", "trust_20", "trust_60", "dealer_5",
                 "dealer_20", "margin_z", "margin_chg20"]
MARKET_FEATURES = ["mkt_r20", "mkt_r60", "mkt_vol20", "mkt_dist_ma200"]


def _ticker_features(px: pd.DataFrame, mkt_r1: pd.Series, cov: pd.DataFrame | None) -> pd.DataFrame:
    c = px["close"].astype(float)
    lc = np.log(c.clip(lower=1e-9))
    r1 = lc.diff()
    f = pd.DataFrame(index=px.index)
    for k in (1, 5, 20, 60, 120):
        f[f"r{k}"] = lc.diff(k)
    f["mom_120_20"] = lc.shift(20) - lc.shift(120)
    f["vol20"] = r1.rolling(20, min_periods=15).std()
    f["vol60"] = r1.rolling(60, min_periods=40).std()
    f["vol_ratio"] = f["vol20"] / f["vol60"]
    for k in (20, 60, 200):
        f[f"dist_ma{k}"] = lc - lc.rolling(k, min_periods=int(k * 0.75)).mean()
    up, dn = r1.clip(lower=0).rolling(14).mean(), (-r1.clip(upper=0)).rolling(14).mean()
    f["rsi14"] = up / (up + dn + 1e-12)
    lo, hi = lc.rolling(60).min(), lc.rolling(60).max()
    f["pos60"] = (lc - lo) / (hi - lo + 1e-12)
    f["max_r20"] = r1.rolling(20).max()
    lv = np.log1p(px["volume"].astype(float))
    f["vol_z"] = (lv - lv.rolling(60, min_periods=20).mean()) / (lv.rolling(60, min_periods=20).std() + 1e-9)
    f["vol_trend"] = lv.rolling(5).mean() - lv.rolling(60, min_periods=20).mean()
    m = mkt_r1.reindex(px.index)
    cov_rm = r1.rolling(60, min_periods=40).cov(m)
    var_m = m.rolling(60, min_periods=40).var()
    f["beta60"] = cov_rm / (var_m + 1e-12)
    f["idio_vol60"] = (r1 - f["beta60"] * m).rolling(60, min_periods=40).std()
    if cov is not None and len(cov):
        cv = cov.reindex(px.index).ffill()
        for g in ("foreign", "trust", "dealer"):
            col = f"{g}_flow_lvl"
            if col in cv:
                for k in ((5, 20, 60) if g != "dealer" else (5, 20)):
                    f[f"{g}_{k}"] = cv[col].diff(k)
        if "margin_z" in cv:
            f["margin_z"] = cv["margin_z"]
            f["margin_chg20"] = cv["margin_z"].diff(20)
    return f


def build_panel(prices: dict[str, pd.DataFrame], market_close: pd.Series, horizon: int, lag: int = 1,
                covariates: dict[str, pd.DataFrame] | None = None, min_names: float = 0.8,
                members: pd.DataFrame | None = None) -> pd.DataFrame:
    """Long panel indexed by (date, ticker) with percentile features, raw market context and labels.

    Columns: features, `fwd_ret` (log return close[d+lag] -> close[d+lag+h]), `fwd_end` (date of that close),
    `fwd_rank` (cross-sectional percentile of fwd_ret; NaN where the future is unknown).
    Dates on which fewer than `min_names` of the universe trade are dropped.
    members: optional point-in-time universe (date x ticker bool, see data.universe.point_in_time_members);
    rows outside it are dropped *before* the cross-sectional percentiles are taken.
    """
    mc = market_close.astype(float).sort_index()
    mc = mc[~mc.index.duplicated(keep="last")]
    lm = np.log(mc)
    mkt = pd.DataFrame(index=mc.index)
    mkt["mkt_r20"] = lm.diff(20)
    mkt["mkt_r60"] = lm.diff(60)
    mkt["mkt_vol20"] = lm.diff().rolling(20).std()
    mkt["mkt_dist_ma200"] = lm - lm.rolling(200, min_periods=150).mean()
    mkt_r1 = lm.diff()
    frames = []
    for t, px in prices.items():
        px = px.sort_index()
        px = px[~px.index.duplicated(keep="last")]
        f = _ticker_features(px, mkt_r1, None if covariates is None else covariates.get(t))
        lc = np.log(px["close"].astype(float).clip(lower=1e-9))
        entry, exit_ = lc.shift(-lag), lc.shift(-(lag + horizon))
        f["fwd_ret"] = exit_ - entry
        dates = pd.Series(px.index, index=px.index)
        f["fwd_end"] = dates.shift(-(lag + horizon))
        f["ticker"] = t
        if members is not None:
            if t not in members:
                continue
            f = f[members[t].reindex(f.index).fillna(False).astype(bool)]
        frames.append(f)
    panel = pd.concat(frames)
    panel.index.name = "date"
    panel = panel.reset_index()
    counts = panel.groupby("date")["ticker"].transform("size")
    size = int(members.sum(axis=1).max()) if members is not None else len(prices)
    panel = panel[counts >= min_names * size]
    stock_cols = [c for c in PRICE_FEATURES + CHIP_FEATURES if c in panel]
    panel[stock_cols] = panel.groupby("date")[stock_cols].rank(pct=True)
    panel = panel.merge(mkt, left_on="date", right_index=True, how="left")
    panel["fwd_rank"] = panel.groupby("date")["fwd_ret"].rank(pct=True)
    return panel.set_index(["date", "ticker"]).sort_index()


def feature_columns(panel: pd.DataFrame, chips: bool) -> list[str]:
    cols = [c for c in PRICE_FEATURES if c in panel] + [c for c in MARKET_FEATURES if c in panel]
    if chips:
        cols += [c for c in CHIP_FEATURES if c in panel and panel[c].notna().any()]
    return cols
