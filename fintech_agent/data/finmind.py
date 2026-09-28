"""FinMind REST client for Taiwan market data (籌碼 / 基本面 / 新聞).

Docs: https://finmind.github.io/  Endpoint: https://api.finmindtrade.com/api/v4/data
Works without a token (lower rate limit); set FINMIND_TOKEN for more quota.
"""
from __future__ import annotations

import logging
import time
from datetime import date, timedelta

import httpx
import pandas as pd

log = logging.getLogger(__name__)
API_URL = "https://api.finmindtrade.com/api/v4/data"


class FinMindClient:
    def __init__(self, token: str | None = None, timeout: float = 20.0, max_wait_s: float = 0.0):
        self.token = token or None
        self.timeout = timeout
        self.max_wait_s = max_wait_s          # >0: wait out the hourly quota instead of giving up (bulk jobs)

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
        waited = 0.0
        while True:
            try:
                r = httpx.get(API_URL, params=params, headers=headers, timeout=self.timeout)
                if r.status_code == 402 or "upper limit" in r.text[:300].lower():
                    if waited + 60 <= self.max_wait_s:
                        log.warning("FinMind quota reached; waiting 60s (waited %.0fs)", waited)
                        time.sleep(60)
                        waited += 60
                        continue
                    log.warning("FinMind quota reached for %s %s — set FINMIND_TOKEN for a higher limit", dataset, data_id)
                    return pd.DataFrame()
                r.raise_for_status()
                payload = r.json()
                break
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

    def institutional_range(self, code: str, start: str, end: str | None = None) -> pd.DataFrame:
        return self.fetch("TaiwanStockInstitutionalInvestorsBuySell", code, start, end)

    def margin_range(self, code: str, start: str, end: str | None = None) -> pd.DataFrame:
        return self.fetch("TaiwanStockMarginPurchaseShortSale", code, start, end)

    def market_institutional(self, days: int = 60) -> pd.DataFrame:
        return self.fetch("TaiwanStockTotalInstitutionalInvestors", None, self._ago(days))
