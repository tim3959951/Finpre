"""Pure functions that turn raw FinMind / yfinance frames into compact evidence dicts.

Kept free of I/O so they can be unit-tested and reused by any data provider.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

INSTITUTION_GROUP = {
    "Foreign_Investor": "foreign",
    "Foreign_Dealer_Self": "foreign",
    "Investment_Trust": "trust",
    "Dealer_self": "dealer",
    "Dealer_Hedging": "dealer",
}


def _f(x: Any, nd: int = 4) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(v) or math.isinf(v) else round(v, nd)


def signed_streak(s: pd.Series) -> int:
    """+n = n consecutive positive values at the end, -n = n consecutive negatives."""
    vals = [v for v in s.to_numpy() if not pd.isna(v)]
    if not vals or vals[-1] == 0:
        return 0
    sign = 1 if vals[-1] > 0 else -1
    n = 0
    for v in reversed(vals):
        if (v > 0 and sign > 0) or (v < 0 and sign < 0):
            n += 1
        else:
            break
    return sign * n


def summarize_institutional(df: pd.DataFrame, avg_daily_volume_shares: float | None = None) -> dict:
    """三大法人買賣超 summary (units: 張 = 1,000 shares)."""
    if df is None or df.empty:
        return {}
    d = df.copy()
    d["net"] = d["buy"].astype(float) - d["sell"].astype(float)
    d["group"] = d["name"].map(INSTITUTION_GROUP).fillna("other")
    pv = d.pivot_table(index="date", columns="group", values="net", aggfunc="sum").sort_index().fillna(0.0)
    out: dict[str, Any] = {"as_of": str(pv.index[-1].date())}
    for g in ("foreign", "trust", "dealer"):
        if g not in pv:
            continue
        s = pv[g]
        out[g] = {
            "net_1d_lots": _f(s.iloc[-1] / 1000, 1),
            "net_5d_lots": _f(s.tail(5).sum() / 1000, 1),
            "net_20d_lots": _f(s.tail(20).sum() / 1000, 1),
            "streak_days": signed_streak(s),
        }
    total = pv[[c for c in ("foreign", "trust", "dealer") if c in pv]].sum(axis=1)
    out["total_net_5d_lots"] = _f(total.tail(5).sum() / 1000, 1)
    out["total_net_20d_lots"] = _f(total.tail(20).sum() / 1000, 1)
    if avg_daily_volume_shares:
        out["net_20d_pct_of_avg_volume"] = _f(total.tail(20).sum() / (avg_daily_volume_shares * 20) * 100, 2)
    return out


def summarize_margin(df: pd.DataFrame) -> dict:
    """融資融券 summary (balances in 張)."""
    if df is None or df.empty:
        return {}
    d = df.sort_values("date")
    mb = d["MarginPurchaseTodayBalance"].astype(float)
    sb = d["ShortSaleTodayBalance"].astype(float)

    def chg(s: pd.Series, n: int) -> float | None:
        if len(s) <= n or s.iloc[-n - 1] == 0:
            return None
        return _f((s.iloc[-1] / s.iloc[-n - 1] - 1) * 100, 2)

    return {
        "as_of": str(d["date"].iloc[-1].date()),
        "margin_balance": _f(mb.iloc[-1], 0),
        "margin_chg_5d_pct": chg(mb, 5),
        "margin_chg_20d_pct": chg(mb, 20),
        "short_balance": _f(sb.iloc[-1], 0),
        "short_chg_20d_pct": chg(sb, 20),
        "short_margin_ratio_pct": _f(sb.iloc[-1] / mb.iloc[-1] * 100, 2) if mb.iloc[-1] else None,  # 券資比
    }


def summarize_shareholding(df: pd.DataFrame) -> dict:
    """外資持股比例."""
    if df is None or df.empty or "ForeignInvestmentSharesRatio" not in df:
        return {}
    d = df.sort_values("date")
    r = d["ForeignInvestmentSharesRatio"].astype(float)
    return {
        "as_of": str(d["date"].iloc[-1].date()),
        "foreign_ratio_pct": _f(r.iloc[-1], 2),
        "foreign_ratio_chg_20d": _f(r.iloc[-1] - r.iloc[max(0, len(r) - 21)], 2),
    }


def summarize_month_revenue(df: pd.DataFrame) -> dict:
    """月營收 MoM / YoY / 累計 YoY."""
    if df is None or df.empty:
        return {}
    d = df.copy()
    d["ym"] = d["revenue_year"].astype(int) * 100 + d["revenue_month"].astype(int)
    d = d.drop_duplicates("ym").sort_values("ym").set_index("ym")
    rev = d["revenue"].astype(float)
    last = int(rev.index[-1])
    y, m = divmod(last, 100)
    prev_m = (y * 100 + m - 1) if m > 1 else ((y - 1) * 100 + 12)
    yoy_key = (y - 1) * 100 + m

    def pct(a, b):
        return _f((a / b - 1) * 100, 2) if b else None

    ytd = rev[(rev.index // 100 == y)].sum()
    ytd_prev = rev[(rev.index // 100 == y - 1) & (rev.index % 100 <= m)].sum()
    yoys = []
    for k in rev.index[-3:]:
        ky, km = divmod(int(k), 100)
        base = rev.get((ky - 1) * 100 + km)
        if base:
            yoys.append(rev[k] / base - 1)
    return {
        "latest_month": f"{y}-{m:02d}",
        "revenue": _f(rev.iloc[-1], 0),
        "mom_pct": pct(rev.iloc[-1], rev.get(prev_m)),
        "yoy_pct": pct(rev.iloc[-1], rev.get(yoy_key)),
        "ytd_yoy_pct": pct(ytd, ytd_prev) if ytd_prev else None,
        "yoy_3m_avg_pct": _f(np.mean(yoys) * 100, 2) if yoys else None,
        "is_24m_high": bool(rev.iloc[-1] >= rev.tail(24).max()),
    }


def summarize_per(df: pd.DataFrame) -> dict:
    """本益比 / 股價淨值比 / 殖利率 and PER percentile vs own history (本益比河流圖概念)."""
    if df is None or df.empty:
        return {}
    d = df.sort_values("date")
    per = d["PER"].astype(float).replace(0, np.nan)
    last = d.iloc[-1]
    pct_rank = None
    if per.notna().sum() > 60 and not pd.isna(per.iloc[-1]):
        pct_rank = _f((per.dropna() < per.iloc[-1]).mean() * 100, 1)
    return {
        "as_of": str(last["date"].date()),
        "pe": _f(last.get("PER"), 2),
        "pb": _f(last.get("PBR"), 2),
        "dividend_yield_pct": _f(last.get("dividend_yield"), 2),
        "pe_percentile_3y": pct_rank,
        "pe_median_3y": _f(per.median(), 2),
    }


def summarize_financials(df: pd.DataFrame) -> dict:
    """EPS TTM growth and gross margin from quarterly statements."""
    if df is None or df.empty:
        return {}
    d = df.pivot_table(index="date", columns="type", values="value", aggfunc="last").sort_index()
    out: dict[str, Any] = {"latest_quarter": str(d.index[-1].date())}
    if "EPS" in d:
        eps = d["EPS"].dropna()
        out["eps_latest_q"] = _f(eps.iloc[-1], 2)
        if len(eps) >= 5:
            out["eps_q_yoy_pct"] = _f((eps.iloc[-1] / eps.iloc[-5] - 1) * 100, 2) if eps.iloc[-5] > 0 else None
        if len(eps) >= 4:
            out["eps_ttm"] = _f(eps.tail(4).sum(), 2)
        if len(eps) >= 8:
            prev = eps.iloc[-8:-4].sum()
            out["eps_ttm_yoy_pct"] = _f((eps.tail(4).sum() / prev - 1) * 100, 2) if prev > 0 else None
    if "GrossProfit" in d and "Revenue" in d:
        gm = (d["GrossProfit"] / d["Revenue"]).dropna() * 100
        if len(gm):
            out["gross_margin_pct"] = _f(gm.iloc[-1], 2)
            if len(gm) >= 5:
                out["gross_margin_yoy_pp"] = _f(gm.iloc[-1] - gm.iloc[-5], 2)
    return out


def summarize_us_info(info: dict) -> dict:
    """Pick the useful fields from yfinance Ticker.info."""
    keys = {
        "longName": "name", "sector": "sector", "industry": "industry", "marketCap": "market_cap",
        "trailingPE": "pe", "forwardPE": "forward_pe", "priceToBook": "pb", "pegRatio": "peg",
        "dividendYield": "dividend_yield", "profitMargins": "profit_margin", "grossMargins": "gross_margin",
        "revenueGrowth": "revenue_growth_yoy", "earningsGrowth": "earnings_growth_yoy",
        "returnOnEquity": "roe", "debtToEquity": "debt_to_equity", "freeCashflow": "free_cash_flow",
        "trailingEps": "eps_ttm", "forwardEps": "eps_forward",
        "recommendationMean": "analyst_rating_mean", "recommendationKey": "analyst_rating",
        "targetMeanPrice": "analyst_target_mean", "numberOfAnalystOpinions": "analyst_count",
        "heldPercentInstitutions": "inst_holding", "heldPercentInsiders": "insider_holding",
        "shortPercentOfFloat": "short_pct_float", "shortRatio": "short_ratio_days",
        "sharesShort": "shares_short", "sharesShortPriorMonth": "shares_short_prior_month",
        "beta": "beta", "fiftyTwoWeekHigh": "high_52w", "fiftyTwoWeekLow": "low_52w",
    }
    out: dict[str, Any] = {}
    for k, v in keys.items():
        val = info.get(k)
        if val is None:
            continue
        out[v] = val if isinstance(val, str) else _f(val, 4)
    if out.get("shares_short") and out.get("shares_short_prior_month"):
        out["short_interest_chg_pct"] = _f((out["shares_short"] / out["shares_short_prior_month"] - 1) * 100, 2)
    return out


def summarize_index(close: pd.Series) -> dict:
    """Market regime of a benchmark index (大盤/S&P)."""
    if close is None or len(close) < 60:
        return {}
    c = close.dropna()
    ma20, ma60 = c.rolling(20).mean().iloc[-1], c.rolling(60).mean().iloc[-1]
    ma200 = c.rolling(200).mean().iloc[-1] if len(c) >= 200 else np.nan
    ret = c.pct_change()
    return {
        "last": _f(c.iloc[-1], 2),
        "ret_5d_pct": _f((c.iloc[-1] / c.iloc[-6] - 1) * 100, 2),
        "ret_20d_pct": _f((c.iloc[-1] / c.iloc[-21] - 1) * 100, 2),
        "above_ma20": bool(c.iloc[-1] > ma20),
        "above_ma60": bool(c.iloc[-1] > ma60),
        "above_ma200": None if pd.isna(ma200) else bool(c.iloc[-1] > ma200),
        "vol_20d_ann_pct": _f(ret.tail(20).std() * np.sqrt(252) * 100, 2),
        "drawdown_from_1y_high_pct": _f((c.iloc[-1] / c.tail(252).max() - 1) * 100, 2),
    }
