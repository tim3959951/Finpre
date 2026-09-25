"""FinMind REST client for Taiwan market data (籌碼 / 基本面 / 新聞).

Docs: https://finmind.github.io/  Endpoint: https://api.finmindtrade.com/api/v4/data
Works without a token (lower rate limit); set FINMIND_TOKEN for more quota.
"""
from __future__ import annotations

import logging
from datetime import date, timedelta

import httpx
import pandas as pd

log = logging.getLogger(__name__)
API_URL = "https://api.finmindtrade.com/api/v4/data"


class FinMindClient:
    def __init__(self, token: str | None = None, timeout: float = 20.0):
        self.token = token or None
        self.timeout = timeout

    def fetch(self, dataset: str, data_id: str | None = None, start: str | date | None = None,
              end: str | date | None = None) -> pd.DataFrame:
        params: dict[str, str] = {"dataset": dataset}
        if data_id:
            params["data_id"] = data_id
        if start:
            params["start_date"] = str(start)
        if end:
            params["end_date"] = str(end)
        headers = {"Authorization": f"Bearer {self.token}"} if self.token else {}
        try:
            r = httpx.get(API_URL, params=params, headers=headers, timeout=self.timeout)
            r.raise_for_status()
            payload = r.json()
        except Exception as e:  # network / quota errors should never crash an analysis
            log.warning("FinMind %s %s failed: %s", dataset, data_id, e)
            return pd.DataFrame()
        if payload.get("status") not in (200, None) or not payload.get("data"):
            if payload.get("msg") and payload.get("msg") != "success":
                log.warning("FinMind %s %s: %s", dataset, data_id, payload.get("msg"))
            return pd.DataFrame()
        df = pd.DataFrame(payload["data"])
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
        return df

    # ---- convenience wrappers -------------------------------------------------
    @staticmethod
    def _ago(days: int) -> str:
        return (date.today() - timedelta(days=days)).isoformat()

    def stock_info(self) -> pd.DataFrame:
        return self.fetch("TaiwanStockInfo")

    def prices(self, code: str, start: str) -> pd.DataFrame:
        df = self.fetch("TaiwanStockPrice", code, start)
        if df.empty:
            return df
        df = df.rename(columns={"max": "high", "min": "low", "Trading_Volume": "volume"})
        return df.set_index("date")[["open", "high", "low", "close", "volume"]].astype(float)

    def institutional(self, code: str, days: int = 120) -> pd.DataFrame:
        return self.fetch("TaiwanStockInstitutionalInvestorsBuySell", code, self._ago(days))

    def margin(self, code: str, days: int = 120) -> pd.DataFrame:
        return self.fetch("TaiwanStockMarginPurchaseShortSale", code, self._ago(days))

    def shareholding(self, code: str, days: int = 120) -> pd.DataFrame:
        return self.fetch("TaiwanStockShareholding", code, self._ago(days))

    def month_revenue(self, code: str, days: int = 800) -> pd.DataFrame:
        return self.fetch("TaiwanStockMonthRevenue", code, self._ago(days))

    def per(self, code: str, days: int = 1100) -> pd.DataFrame:
        return self.fetch("TaiwanStockPER", code, self._ago(days))

    def financial_statements(self, code: str, days: int = 1200) -> pd.DataFrame:
        return self.fetch("TaiwanStockFinancialStatements", code, self._ago(days))

    def news(self, code: str, days: int = 7) -> pd.DataFrame:
        return self.fetch("TaiwanStockNews", code, self._ago(days))

    def market_institutional(self, days: int = 60) -> pd.DataFrame:
        return self.fetch("TaiwanStockTotalInstitutionalInvestors", None, self._ago(days))
