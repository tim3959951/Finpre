"""Market regime labels (known at each date, no look-ahead) for splitting backtest results.

bull  : index above its 200-day average and up over the last 60 trading days
bear  : index below its 200-day average and down over the last 60 trading days
mixed : everything else (range-bound markets, turning points)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

REGIME_LABELS = {"bull": "多頭", "bear": "空頭", "mixed": "盤整/轉折"}

# Named stress episodes used in the reports (inclusive date ranges)
EPISODES = {
    "TW": {"2018-Q4 全球修正": ("2018-10-01", "2018-12-31"), "2020 COVID 崩盤": ("2020-02-20", "2020-04-30"),
           "2022 升息空頭": ("2022-01-01", "2022-10-31"), "2025-04 關稅衝擊": ("2025-03-15", "2025-05-15")},
    "US": {"2018-Q4 修正": ("2018-10-01", "2018-12-31"), "2020 COVID 崩盤": ("2020-02-20", "2020-04-30"),
           "2022 升息空頭": ("2022-01-01", "2022-10-31"), "2025-04 關稅衝擊": ("2025-03-15", "2025-05-15")},
}


def market_regime(close: pd.Series, ma: int = 200, lookback: int = 60) -> pd.Series:
    c = close.astype(float).dropna().sort_index()
    c = c[~c.index.duplicated(keep="last")]
    above = c > c.rolling(ma, min_periods=int(ma * 0.75)).mean()
    mom = c.pct_change(lookback)
    out = pd.Series("mixed", index=c.index)
    out[above & (mom > 0)] = "bull"
    out[~above & (mom < 0)] = "bear"
    out[c.rolling(ma, min_periods=int(ma * 0.75)).mean().isna()] = np.nan
    return out


def regime_at(regime: pd.Series, dates) -> np.ndarray:
    """Regime known at each date (last label on/before it)."""
    r = regime.dropna()
    r = r[~r.index.duplicated(keep="last")]
    idx = pd.DatetimeIndex(dates)
    u = idx.unique()
    known = r.reindex(r.index.union(u)).ffill().reindex(u)
    return known.reindex(idx).to_numpy()


def episode_of(dates, market: str) -> np.ndarray:
    idx = pd.DatetimeIndex(dates)
    out = np.full(len(idx), None, dtype=object)
    for name, (a, b) in EPISODES.get(market, {}).items():
        out[(idx >= pd.Timestamp(a)) & (idx <= pd.Timestamp(b))] = name
    return out


def skill_by_period(w: pd.DataFrame, regime: pd.Series | None, market: str) -> dict:
    """CRPS skill vs naive, direction accuracy and share of tickers beating naive — by year / regime / episode."""
    if "naive" not in set(w["model"]):
        return {}
    d = w.copy()
    d["origin"] = pd.to_datetime(d["origin"])
    d["year"] = d["origin"].dt.year
    d["regime"] = regime_at(regime, d["origin"]) if regime is not None else None
    d["episode"] = episode_of(d["origin"], market)
    out = {}
    for by in ("year", "regime", "episode"):
        g = d.dropna(subset=[by])
        if g.empty:
            continue
        m = g.groupby([by, "model"])["crps_rel"].mean().unstack("model")
        skill = (1 - m.div(m["naive"], axis=0)) * 100
        pt = g.groupby([by, "ticker", "model"])["crps_rel"].mean().unstack("model")
        share = (pt.lt(pt["naive"], axis=0)).groupby(level=0).mean()
        acc = g.groupby([by, "model"])["dir_hit"].mean().unstack("model")
        n = g[g["model"] == "naive"].groupby(by).size()
        rows = []
        for key in skill.index:
            for model in skill.columns:
                if model == "naive":
                    continue
                rows.append({by: key if by != "year" else int(key), "model": model, "windows": int(n.get(key, 0)),
                             "crps_skill_pct": round(float(skill.loc[key, model]), 3),
                             "share_beating_naive": round(float(share.loc[key, model]), 3),
                             "dir_acc": round(float(acc.loc[key, model]), 4)})
        out[by] = rows
    return out
