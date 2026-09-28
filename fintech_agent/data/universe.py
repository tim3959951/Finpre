"""Benchmark universes. Current large caps -> survivorship bias: results describe today's winners, not a
point-in-time index. Good enough for model comparison (all models see the same stocks), not for strategy P&L."""

TW50 = ["2330", "2317", "2454", "2308", "2382", "2881", "2882", "2412", "2891", "3711",
        "2886", "2303", "2884", "2885", "1216", "2357", "3231", "2892", "2345", "5880",
        "2002", "2880", "3008", "1303", "2887", "2883", "6669", "2890", "1301", "2207",
        "3034", "3037", "2327", "4938", "2395", "5871", "2379", "2912", "1326", "3045",
        "2301", "6505", "4904", "2603", "2609", "2615", "5876", "3661", "3017", "1101"]

US50 = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "AVGO", "TSLA", "BRK-B", "JPM",
        "LLY", "V", "UNH", "XOM", "MA", "JNJ", "PG", "HD", "COST", "ABBV",
        "WMT", "NFLX", "BAC", "CRM", "ORCL", "CVX", "KO", "MRK", "AMD", "PEP",
        "ADBE", "TMO", "LIN", "ACN", "MCD", "CSCO", "ABT", "WFC", "DHR", "QCOM",
        "TXN", "INTU", "IBM", "AMGN", "CAT", "GE", "PM", "NOW", "GS", "ISRG"]

UNIVERSES = {"tw50": ("TW", TW50), "us50": ("US", US50),
             "tw6": ("TW", ["2330", "2317", "2454", "2881", "2412", "0050"]),
             "us6": ("US", ["AAPL", "MSFT", "NVDA", "JPM", "XOM", "SPY"])}


# ---------------------------------------------------------------------------- point-in-time universe
# The fixed lists above are *today's* large caps. For strategy backtests (stock ranking) use a point-in-time
# universe instead: every day, the `top` most liquid listed stocks by trailing traded value, chosen from all
# currently listed TWSE common stocks. Remaining bias: stocks delisted / merged away before today are missing.

EXCLUDE_INDUSTRIES = {"ETF", "受益證券", "存託憑證", "ETN", "指數投資證券(ETN)"}


def tw_listed_codes(prov) -> list[str]:
    """All TWSE-listed common stock codes (4 digits, not starting with 0) from FinMind's stock info table."""
    info = prov._tw_info_table()
    d = info[(info["type"] == "twse") & info["stock_id"].astype(str).str.fullmatch(r"[1-9]\d{3}")]
    d = d[~d["industry_category"].isin(EXCLUDE_INDUSTRIES)]
    return sorted(d["stock_id"].astype(str).unique())


def bulk_prices(prov, codes: list[str], market: str = "TW", chunk: int = 80, progress=None) -> dict:
    """yfinance batch download (no FinMind fallback, so it never spends FinMind quota), cached under the same
    keys as `LiveDataProvider.prices` so later per-ticker calls hit the cache."""
    import math
    from datetime import datetime

    import pandas as pd
    import yfinance as yf

    from .provider import _clean_history
    years = prov.years
    today = datetime.now().strftime("%Y%m%d")
    suffix = ".TW" if market == "TW" else ""
    out, todo = {}, []
    for c in codes:
        p = prov.cache._path(f"px_{c}_{market}_{years}_{today}", "parquet")
        if prov.cache._fresh(p):
            try:
                out[c] = pd.read_parquet(p)
                continue
            except Exception:
                pass
        todo.append(c)
    for i in range(0, len(todo), chunk):
        batch = todo[i:i + chunk]
        tick = [c + suffix for c in batch]
        raw = yf.download(tick, period=f"{int(math.ceil(years))}y", auto_adjust=True, group_by="ticker",
                          threads=True, progress=False)
        for c, t in zip(batch, tick):
            try:
                df = _clean_history(raw[t].dropna(how="all")) if len(tick) > 1 else _clean_history(raw)
            except KeyError:
                continue
            if len(df) > 30:
                df["ticker"] = t
                prov.cache.frame(f"px_{c}_{market}_{years}_{today}", lambda df=df: df)
                out[c] = df
        if progress:
            progress(f"prices {min(i + chunk, len(todo))}/{len(todo)} downloaded, {len(out)} usable")
    return out


def point_in_time_members(prices: dict, top: int = 50, window: int = 60, min_history: int = 250):
    """Boolean (date x ticker) frame: True where the stock is among the `top` by trailing `window`-day average
    traded value (close x volume) on that date, using only data up to that date."""
    import numpy as np
    import pandas as pd
    tv, hist = {}, {}
    for t, px in prices.items():
        px = px[~px.index.duplicated(keep="last")].sort_index()
        tv[t] = (px["close"] * px["volume"]).rolling(window, min_periods=window // 2).mean()
        hist[t] = pd.Series(np.arange(1, len(px) + 1), index=px.index)
    tv, hist = pd.DataFrame(tv), pd.DataFrame(hist).ffill()
    tv = tv.where(hist >= min_history)
    return tv.rank(axis=1, ascending=False, method="first") <= top
