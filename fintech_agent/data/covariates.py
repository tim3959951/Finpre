"""Daily covariate panels for covariate-aware forecasting (籌碼 / market / FX).

Every column is *past-only*: its value on date d is known at (or before) the close of d, so it can be used
for a forecast issued at the close of d. Look-ahead is avoided explicitly:
  * US series (S&P 500, SOX) for a Taiwan date d use the last US close strictly *before* d
    (the US session of calendar day d ends after Taiwan's close).
  * FX uses the last quote strictly before d.
  * FinMind 籌碼 for day d is published after that day's close -> known at the close of d.

Two kinds of columns:
  *_lvl : level-like (log index, cumulative flows). Foundation models take these as extra variates.
  *_z   : already stationary (z-scores, ratios).
`covariate_features()` turns a panel into stationary features (k-day changes) for tree models.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MARKET_SERIES = {
    "TW": {"mkt_lvl": ("^TWII", False), "us_lvl": ("^GSPC", True), "sox_lvl": ("^SOX", True),
           "fx_lvl": ("TWD=X", True)},
    "US": {"mkt_lvl": ("^GSPC", False), "vix_lvl": ("^VIX", False), "rate_lvl": ("^TNX", False),
           "dxy_lvl": ("DX-Y.NYB", True)},
}
INSTITUTION_GROUP = {"Foreign_Investor": "foreign", "Foreign_Dealer_Self": "foreign", "Investment_Trust": "trust",
                     "Dealer_self": "dealer", "Dealer_Hedging": "dealer", "Dealer": "dealer"}  # "Dealer" before 2014-12


def align_series(s: pd.Series, index: pd.DatetimeIndex, strictly_before: bool) -> pd.Series:
    """Value known at each date of `index` (last observation on/before, or strictly before, that date)."""
    s = s.dropna().sort_index()
    s = s[~s.index.duplicated(keep="last")]
    if s.empty:
        return pd.Series(np.nan, index=index)
    lookup = index - pd.Timedelta(days=1) if strictly_before else index
    out = s.reindex(s.index.union(lookup)).ffill().reindex(lookup)
    out.index = index
    return out


def volume_z(volume: pd.Series, n: int = 60) -> pd.Series:
    lv = np.log1p(volume.astype(float))
    return ((lv - lv.rolling(n, min_periods=20).mean()) / lv.rolling(n, min_periods=20).std()).clip(-4, 4)


def chip_columns(index: pd.DatetimeIndex, volume: pd.Series, inst: pd.DataFrame | None,
                 margin: pd.DataFrame | None) -> pd.DataFrame:
    """Cumulative institutional net buying (in units of average daily volume) + margin-balance z-score."""
    out = pd.DataFrame(index=index)
    avg_vol = volume.rolling(60, min_periods=1).mean().replace(0, np.nan).ffill()
    if inst is not None and len(inst):
        d = inst.copy()
        d["net"] = d["buy"].astype(float) - d["sell"].astype(float)
        d["group"] = d["name"].map(INSTITUTION_GROUP).fillna("other")
        pv = d.pivot_table(index="date", columns="group", values="net", aggfunc="sum").sort_index()
        pv = pv.reindex(index).fillna(0.0)                      # same-day, published after close
        for g in ("foreign", "trust", "dealer"):
            if g in pv:
                out[f"{g}_flow_lvl"] = (pv[g] / avg_vol).clip(-5, 5).cumsum()
    if margin is not None and len(margin) and "MarginPurchaseTodayBalance" in margin:
        m = margin.set_index("date")["MarginPurchaseTodayBalance"].astype(float).sort_index()
        m = m[~m.index.duplicated(keep="last")].reindex(index).ffill()
        out["margin_z"] = ((m / m.rolling(60, min_periods=10).mean()) - 1).clip(-1, 1)
    return out


def build_covariates(prices: pd.DataFrame, market: str, market_closes: dict[str, pd.Series],
                     inst: pd.DataFrame | None = None, margin: pd.DataFrame | None = None) -> pd.DataFrame:
    """Covariate panel aligned 1:1 with `prices.index` (NaNs forward-filled, then 0)."""
    idx = pd.DatetimeIndex(prices.index)
    cov = pd.DataFrame(index=idx)
    cov["vol_z"] = volume_z(prices["volume"])
    for col, (ticker, strictly_before) in MARKET_SERIES.get(market, {}).items():
        s = market_closes.get(ticker)
        if s is None or len(s) == 0:
            continue
        a = align_series(s.astype(float), idx, strictly_before)
        cov[col] = a / 10.0 if ticker == "^TNX" else np.log(a)
    if market == "TW":
        cov = cov.join(chip_columns(idx, prices["volume"], inst, margin))
    return cov.ffill().fillna(0.0).astype(float)


def covariate_features(cov: pd.DataFrame, t: int, lags=(1, 5, 20)) -> np.ndarray:
    """Stationary features at row t: k-day changes of *_lvl columns, raw values of the rest."""
    feats = []
    for c in cov.columns:
        col = cov[c].to_numpy()
        if c.endswith("_lvl"):
            feats += [col[t] - col[max(0, t - k)] for k in lags]
        else:
            feats.append(col[t])
    return np.asarray(feats, dtype=np.float64)


def covariate_feature_matrix(cov: pd.DataFrame, lags=(1, 5, 20)) -> np.ndarray:
    """Vectorised `covariate_features` for every row (row t == covariate_features(cov, t))."""
    n = len(cov)
    cols = []
    for c in cov.columns:
        col = cov[c].to_numpy(dtype=np.float64)
        if c.endswith("_lvl"):
            for k in lags:
                prev = col[np.maximum(0, np.arange(n) - k)]
                cols.append(col - prev)
        else:
            cols.append(col)
    return np.column_stack(cols) if cols else np.zeros((n, 0))


def lagged_window(cov: pd.DataFrame, n_rows: int) -> pd.DataFrame:
    """Last `n_rows` rows, left-padded with the first row if history is shorter."""
    if len(cov) >= n_rows:
        return cov.iloc[-n_rows:]
    pad = pd.DataFrame([cov.iloc[0].to_numpy()] * (n_rows - len(cov)), columns=cov.columns)
    return pd.concat([pad, cov.reset_index(drop=True)], ignore_index=True)
