"""Technical indicators in plain pandas (no TA-Lib dependency).

Conventions follow common Taiwan broker software where they differ from US defaults
(e.g. KD uses 9-day RSV with 1/3 smoothing; MA set 5/10/20/60/120/240 = 週/雙週/月/季/半年/年線).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MA_WINDOWS = (5, 10, 20, 60, 120, 240)


def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n).mean()


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False, min_periods=n).mean()


def wilder(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    up, down = wilder(d.clip(lower=0), n), wilder(-d.clip(upper=0), n)
    rs = up / down.replace(0, np.nan)
    out = 100 - 100 / (1 + rs)
    return out.where(down != 0, 100.0).where(up.notna())


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    dif = ema(close, fast) - ema(close, slow)
    dea = dif.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return pd.DataFrame({"dif": dif, "dea": dea, "osc": dif - dea})


def kd(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 9) -> pd.DataFrame:
    """Taiwan-style stochastic: K = 2/3 K_prev + 1/3 RSV, D = 2/3 D_prev + 1/3 K, seeded at 50."""
    ll, hh = low.rolling(n, min_periods=n).min(), high.rolling(n, min_periods=n).max()
    rsv = ((close - ll) / (hh - ll).replace(0, np.nan) * 100).fillna(50.0)
    k = np.full(len(close), np.nan)
    d = np.full(len(close), np.nan)
    pk = pd_ = 50.0
    valid = ll.notna().to_numpy()
    for i, r in enumerate(rsv.to_numpy()):
        if not valid[i]:
            continue
        pk = pk * 2 / 3 + r / 3
        pd_ = pd_ * 2 / 3 + pk / 3
        k[i], d[i] = pk, pd_
    return pd.DataFrame({"k": k, "d": d}, index=close.index)


def bollinger(close: pd.Series, n: int = 20, k: float = 2.0) -> pd.DataFrame:
    mid = sma(close, n)
    sd = close.rolling(n, min_periods=n).std(ddof=0)
    upper, lower = mid + k * sd, mid - k * sd
    return pd.DataFrame({"bb_mid": mid, "bb_upper": upper, "bb_lower": lower,
                         "bb_pctb": (close - lower) / (upper - lower).replace(0, np.nan),
                         "bb_width": (upper - lower) / mid})


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    pc = close.shift()
    return pd.concat([high - low, (high - pc).abs(), (low - pc).abs()], axis=1).max(axis=1)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    return wilder(true_range(high, low, close), n)


def adx(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.DataFrame:
    up, down = high.diff(), -low.diff()
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=high.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=high.index)
    tr = wilder(true_range(high, low, close), n)
    pdi = 100 * wilder(plus_dm, n) / tr
    mdi = 100 * wilder(minus_dm, n) / tr
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    return pd.DataFrame({"plus_di": pdi, "minus_di": mdi, "adx": wilder(dx, n)})


def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    return (np.sign(close.diff()).fillna(0) * volume).cumsum()


def williams_r(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    hh, ll = high.rolling(n).max(), low.rolling(n).min()
    return -100 * (hh - close) / (hh - ll).replace(0, np.nan)


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of an OHLCV frame with all indicators appended."""
    out = df.copy()
    c, h, l, v = out["close"], out["high"], out["low"], out["volume"]
    for n in MA_WINDOWS:
        out[f"ma{n}"] = sma(c, n)
        out[f"bias{n}"] = (c / out[f"ma{n}"] - 1) * 100          # 乖離率 %
    out["ema12"], out["ema26"] = ema(c, 12), ema(c, 26)
    out = out.join(macd(c)).join(kd(h, l, c)).join(bollinger(c)).join(adx(h, l, c))
    out["rsi6"], out["rsi14"] = rsi(c, 6), rsi(c, 14)
    out["atr14"] = atr(h, l, c, 14)
    out["atr_pct"] = out["atr14"] / c * 100
    out["obv"] = obv(c, v)
    out["willr14"] = williams_r(h, l, c)
    out["vol_ma20"] = sma(v, 20)
    out["vol_ratio"] = v / out["vol_ma20"]
    out["ret_1d"] = c.pct_change() * 100
    out["hv20"] = np.log(c).diff().rolling(20).std() * np.sqrt(252) * 100   # 歷史波動率 %
    out["high_252"], out["low_252"] = h.rolling(252, min_periods=60).max(), l.rolling(252, min_periods=60).min()
    return out
