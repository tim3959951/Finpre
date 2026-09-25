"""Market data providers.

LiveDataProvider  : yfinance (TW + US prices, US fundamentals/news) + FinMind (TW 籌碼/基本面/新聞), disk-cached.
SyntheticProvider : deterministic fake data for tests and offline demos.
"""
from __future__ import annotations

import hashlib
import logging
import math
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd

from ..config import Settings, get_settings
from .cache import DiskCache
from .finmind import FinMindClient
from .symbols import Symbol, parse_symbol
from . import summaries as S

log = logging.getLogger(__name__)
OHLCV = ["open", "high", "low", "close", "volume"]


class DataProvider:
    """Interface. All methods must degrade gracefully (return {} / empty) rather than raise."""

    def prices(self, sym: Symbol, years: float | None = None) -> pd.DataFrame: ...
    def profile(self, sym: Symbol) -> dict: ...
    def fundamentals(self, sym: Symbol) -> dict: ...
    def chips(self, sym: Symbol, prices: pd.DataFrame | None = None) -> dict: ...
    def news(self, sym: Symbol, limit: int = 15) -> list[dict]: ...
    def market_context(self, market: str) -> dict: ...

    def symbol(self, text: str) -> Symbol:
        return parse_symbol(text)


def _clean_history(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=OHLCV)
    df = df.rename(columns=str.lower)
    idx = pd.to_datetime(df.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    df.index = idx.normalize()
    df.index.name = "date"
    df = df[[c for c in OHLCV if c in df.columns]].astype(float)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df[(df["close"] > 0) & df["close"].notna()]


def _parse_yf_news(item: dict) -> dict | None:
    c = item.get("content", item) or {}
    title = c.get("title")
    if not title:
        return None
    when = c.get("pubDate") or c.get("displayTime") or item.get("providerPublishTime")
    if isinstance(when, (int, float)):
        when = datetime.fromtimestamp(when).isoformat()
    provider = c.get("provider") or {}
    url = (c.get("canonicalUrl") or {}).get("url") or c.get("link") or item.get("link")
    return {
        "date": str(when)[:19] if when else None,
        "title": title,
        "source": provider.get("displayName") if isinstance(provider, dict) else item.get("publisher"),
        "url": url,
        "summary": (c.get("summary") or "")[:400],
    }


class LiveDataProvider(DataProvider):
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self.cache = DiskCache(self.settings.resolve_path("data.cache_dir"),
                               float(self.settings.get_path("data.cache_ttl_hours", 12)))
        self.fm = FinMindClient(self.settings.get_path("data.finmind_token") or None)
        self.years = float(self.settings.get_path("data.history_years", 6))

    # ---------------------------------------------------------------- prices
    def _yf_history(self, ticker: str, years: float) -> pd.DataFrame:
        import yfinance as yf
        try:
            df = yf.Ticker(ticker).history(period=f"{int(math.ceil(years))}y", auto_adjust=True)
        except Exception as e:
            log.warning("yfinance %s failed: %s", ticker, e)
            return pd.DataFrame()
        return _clean_history(df)

    def prices(self, sym: Symbol, years: float | None = None) -> pd.DataFrame:
        years = years or self.years
        today = datetime.now().strftime("%Y%m%d")

        def load() -> pd.DataFrame:
            for t in sym.yf_candidates:
                df = self._yf_history(t, years)
                if len(df) > 30:
                    df["ticker"] = t
                    return df
            if sym.market == "TW" and not sym.is_index:  # FinMind fallback (unadjusted)
                start = (datetime.now() - timedelta(days=int(365 * years))).date().isoformat()
                df = _clean_history(self.fm.prices(sym.code, start))
                if len(df):
                    df["ticker"] = f"{sym.code}.TW"
                return df
            return pd.DataFrame(columns=OHLCV)

        df = self.cache.frame(f"px_{sym.code}_{sym.market}_{years}_{today}", load)
        return df

    # ---------------------------------------------------------------- profile
    def _tw_info_table(self) -> pd.DataFrame:
        return self.cache.frame(f"tw_info_{datetime.now():%Y%m%d}", self.fm.stock_info)

    def _yf_info(self, sym: Symbol) -> dict:
        def load() -> dict:
            import yfinance as yf
            for t in sym.yf_candidates:
                try:
                    info = yf.Ticker(t).info or {}
                except Exception as e:
                    log.warning("yfinance info %s failed: %s", t, e)
                    info = {}
                if info.get("regularMarketPrice") or info.get("longName") or info.get("shortName"):
                    return {k: v for k, v in info.items() if isinstance(v, (str, int, float, bool)) or v is None}
            return {}
        return self.cache.obj(f"yfinfo_{sym.code}_{datetime.now():%Y%m%d}", load) or {}

    def profile(self, sym: Symbol) -> dict:
        out: dict[str, Any] = {"code": sym.code, "market": sym.market, "currency": sym.currency}
        if sym.market == "TW" and not sym.is_index:
            try:
                tbl = self._tw_info_table()
            except Exception as e:
                log.warning("TaiwanStockInfo failed: %s", e)
                tbl = None
            if tbl is not None and len(tbl):
                row = tbl[tbl["stock_id"].astype(str) == sym.code]
                if len(row):
                    r = row.iloc[-1]
                    out.update(name=r.get("stock_name"), industry=r.get("industry_category"),
                               exchange="TWSE" if r.get("type") == "twse" else "TPEx")
        info = self._yf_info(sym) if not sym.is_index else {}
        out.setdefault("name", info.get("longName") or info.get("shortName") or sym.code)
        if info.get("sector"):
            out["sector"] = info.get("sector")
        out.setdefault("industry", info.get("industry"))
        return out

    # ---------------------------------------------------------------- fundamentals
    def fundamentals(self, sym: Symbol) -> dict:
        if sym.is_index:
            return {}
        out: dict[str, Any] = {}
        info = self._yf_info(sym)
        if info:
            out["yahoo"] = S.summarize_us_info(info)
        if sym.market == "TW":
            key = f"{sym.code}_{datetime.now():%Y%m%d}"
            out["valuation"] = self.cache.obj(f"per_{key}", lambda: S.summarize_per(self.fm.per(sym.code)))
            out["monthly_revenue"] = self.cache.obj(f"rev_{key}", lambda: S.summarize_month_revenue(self.fm.month_revenue(sym.code)))
            out["financials"] = self.cache.obj(f"fin_{key}", lambda: S.summarize_financials(self.fm.financial_statements(sym.code)))
        return {k: v for k, v in out.items() if v}

    # ---------------------------------------------------------------- chips (籌碼)
    def chips(self, sym: Symbol, prices: pd.DataFrame | None = None) -> dict:
        if sym.is_index:
            return {}
        key = f"{sym.code}_{datetime.now():%Y%m%d}"
        if sym.market == "TW":
            avg_vol = None
            if prices is not None and len(prices) >= 20:
                avg_vol = float(prices["volume"].tail(20).mean())
            out = {
                "institutional": self.cache.obj(f"inst_{key}", lambda: S.summarize_institutional(self.fm.institutional(sym.code), avg_vol)),
                "margin": self.cache.obj(f"margin_{key}", lambda: S.summarize_margin(self.fm.margin(sym.code))),
                "shareholding": self.cache.obj(f"hold_{key}", lambda: S.summarize_shareholding(self.fm.shareholding(sym.code))),
            }
            return {k: v for k, v in out.items() if v}
        # US: ownership / short interest / insider activity
        info = S.summarize_us_info(self._yf_info(sym))
        out = {k: info[k] for k in ("inst_holding", "insider_holding", "short_pct_float", "short_ratio_days",
                                    "short_interest_chg_pct") if k in info}
        out["insider_6m"] = self.cache.obj(f"insider_{key}", lambda: self._us_insider(sym))
        return {k: v for k, v in out.items() if v}

    def _us_insider(self, sym: Symbol) -> dict:
        import yfinance as yf
        try:
            df = yf.Ticker(sym.yf_ticker).insider_transactions
        except Exception:
            return {}
        if df is None or df.empty:
            return {}
        date_col = next((c for c in df.columns if "date" in c.lower()), None)
        if date_col:
            df = df[pd.to_datetime(df[date_col], errors="coerce") > datetime.now() - timedelta(days=182)]
        text = df.get("Text", pd.Series("", index=df.index)).fillna("").str.lower()
        buys, sells = text.str.contains("purchase|buy"), text.str.contains("sale|sell")
        val = pd.to_numeric(df.get("Value", 0), errors="coerce").fillna(0)
        return {"buy_count": int(buys.sum()), "sell_count": int(sells.sum()),
                "buy_value": float(val[buys].sum()), "sell_value": float(val[sells].sum())}

    # ---------------------------------------------------------------- news
    def news(self, sym: Symbol, limit: int = 15) -> list[dict]:
        def load() -> list[dict]:
            items: list[dict] = []
            if sym.market == "TW" and not sym.is_index:
                df = self.fm.news(sym.code, days=7)
                if len(df):
                    df = df.sort_values("date", ascending=False)
                    for _, r in df.head(limit).iterrows():
                        items.append({"date": str(r.get("date"))[:19], "title": r.get("title"),
                                      "source": r.get("source"), "url": r.get("link"), "summary": ""})
            if len(items) < limit:
                import yfinance as yf
                try:
                    raw = yf.Ticker(sym.yf_ticker).news or []
                except Exception:
                    raw = []
                items += [n for n in (_parse_yf_news(x) for x in raw) if n]
            seen, uniq = set(), []
            for n in items:
                if n["title"] not in seen:
                    seen.add(n["title"])
                    uniq.append(n)
            return uniq[:limit]
        return self.cache.obj(f"news_{sym.code}_{datetime.now():%Y%m%d%H}", load) or []

    # ---------------------------------------------------------------- market context
    def market_context(self, market: str) -> dict:
        out: dict[str, Any] = {}
        idx = self.settings.get_path(f"data.market_index.{market}")
        if idx:
            px = self.prices(parse_symbol(idx), years=2)
            out["index"] = {"symbol": idx, **S.summarize_index(px["close"] if len(px) else None)}
        vix = self.settings.get_path("data.volatility_index.US")
        if vix:
            px = self.prices(parse_symbol(vix), years=1)
            if len(px):
                out["vix"] = round(float(px["close"].iloc[-1]), 2)
        if market == "TW":
            def load() -> dict:
                df = self.fm.market_institutional(days=40)
                if df.empty:
                    return {}
                df["net"] = df["buy"].astype(float) - df["sell"].astype(float)
                f = df[df["name"].str.contains("Foreign", case=False)].groupby("date")["net"].sum().sort_index()
                t = df[df["name"].str.contains("Trust", case=False)].groupby("date")["net"].sum().sort_index()
                return {"foreign_net_5d_100m_twd": round(f.tail(5).sum() / 1e8, 1),
                        "foreign_net_20d_100m_twd": round(f.tail(20).sum() / 1e8, 1),
                        "trust_net_5d_100m_twd": round(t.tail(5).sum() / 1e8, 1) if len(t) else None,
                        "foreign_streak_days": S.signed_streak(f)}
            flows = self.cache.obj(f"mkt_inst_{datetime.now():%Y%m%d}", load)
            if flows:
                out["institutional_flows"] = flows
        return out


class SyntheticProvider(DataProvider):
    """Geometric-random-walk prices + plausible fake fundamentals. Deterministic per ticker."""

    def __init__(self, n_days: int = 900, drift: float = 0.0004, vol: float = 0.018):
        self.n_days, self.drift, self.vol = n_days, drift, vol

    def _rng(self, sym: Symbol) -> np.random.Generator:
        return np.random.default_rng(int(hashlib.md5(sym.code.encode()).hexdigest()[:8], 16))

    def prices(self, sym: Symbol, years: float | None = None) -> pd.DataFrame:
        rng = self._rng(sym)
        n = self.n_days
        rets = rng.normal(self.drift, self.vol, n)
        close = 100 * np.exp(np.cumsum(rets))
        high = close * (1 + np.abs(rng.normal(0, self.vol / 2, n)))
        low = close * (1 - np.abs(rng.normal(0, self.vol / 2, n)))
        open_ = np.r_[close[0], close[:-1]] * (1 + rng.normal(0, self.vol / 4, n))
        high, low = np.maximum.reduce([high, open_, close]), np.minimum.reduce([low, open_, close])
        vol = rng.lognormal(15, 0.4, n)
        idx = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n, name="date")
        return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": vol}, index=idx)

    def profile(self, sym: Symbol) -> dict:
        return {"code": sym.code, "market": sym.market, "currency": sym.currency, "name": f"Synthetic {sym.code}",
                "industry": "Test"}

    def fundamentals(self, sym: Symbol) -> dict:
        rng = self._rng(sym)
        return {"valuation": {"pe": round(float(rng.uniform(8, 35)), 2), "pb": round(float(rng.uniform(1, 6)), 2),
                              "dividend_yield_pct": round(float(rng.uniform(0, 5)), 2),
                              "pe_percentile_3y": round(float(rng.uniform(0, 100)), 1)},
                "monthly_revenue": {"yoy_pct": round(float(rng.normal(8, 15)), 2),
                                    "mom_pct": round(float(rng.normal(0, 8)), 2),
                                    "yoy_3m_avg_pct": round(float(rng.normal(6, 12)), 2)},
                "financials": {"eps_ttm_yoy_pct": round(float(rng.normal(10, 20)), 2),
                               "gross_margin_yoy_pp": round(float(rng.normal(0, 2)), 2)}}

    def chips(self, sym: Symbol, prices: pd.DataFrame | None = None) -> dict:
        rng = self._rng(sym)
        if sym.market != "TW":
            return {"inst_holding": 0.7, "short_pct_float": round(float(rng.uniform(0.005, 0.08)), 4)}
        f5 = float(rng.normal(0, 3000))
        return {"institutional": {"foreign": {"net_5d_lots": round(f5, 1), "net_20d_lots": round(f5 * 3, 1),
                                              "streak_days": int(np.sign(f5) * rng.integers(1, 6))},
                                  "trust": {"net_5d_lots": round(float(rng.normal(0, 500)), 1),
                                            "net_20d_lots": round(float(rng.normal(0, 1500)), 1),
                                            "streak_days": int(rng.integers(-4, 5))},
                                  "total_net_20d_lots": round(f5 * 3, 1)},
                "margin": {"margin_chg_20d_pct": round(float(rng.normal(0, 6)), 2),
                           "short_margin_ratio_pct": round(float(rng.uniform(0, 15)), 2)}}

    def news(self, sym: Symbol, limit: int = 15) -> list[dict]:
        return [{"date": "2026-01-01", "title": f"{sym.code} 營收創新高，法人看好後市", "source": "synthetic"},
                {"date": "2026-01-02", "title": f"{sym.code} faces margin pressure amid weak demand", "source": "synthetic"}][:limit]

    def market_context(self, market: str) -> dict:
        return {"index": {"symbol": "^TWII" if market == "TW" else "^GSPC", "ret_20d_pct": 1.2,
                          "above_ma60": True, "vol_20d_ann_pct": 14.0}}
