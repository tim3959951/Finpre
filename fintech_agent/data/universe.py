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
