"""Finpre B2B API (FastAPI).

  uvicorn fintech_agent.api.main:app --host 0.0.0.0 --port 8000       # OpenAPI docs at /docs
  python scripts/manage_tenants.py create --name "某某投顧" --plan enterprise --licensed --mode advisor

Every request is authenticated with an API key (header `X-API-Key` or `Authorization: Bearer ...`), metered
against the tenant's plan, run in the tenant's compliance mode, and written to the audit log.
"""
from __future__ import annotations

import copy
import json
import logging
import threading
import time
from datetime import datetime, timezone
from typing import Literal

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse, StreamingResponse
from contextlib import contextmanager

from pydantic import BaseModel, Field, field_validator

from .. import __version__
from ..agents import ClientProfile, InvestmentAdvisor
from ..config import Settings, get_settings
from ..data.provider import DataProvider
from ..llm.base import LLMClient
from ..product.audit import AuditLog
from ..product.compliance import MODE_LABEL, RESEARCH_DISCLAIMER, ComplianceGuard
from ..product.forecast import forecast_ticker
from ..product.plans import PLANS, QuotaExceeded, Tenant, TenantStore
from ..product.portfolio import portfolio_risk
from ..product.report import build_report, render_html, render_markdown
from ..product.scorecard import backtest_scorecard, live_scorecard

log = logging.getLogger(__name__)


class AnalyzeIn(BaseModel):
    ticker: str = Field(..., examples=["2330", "NVDA"], max_length=12)
    horizon: int = Field(5, ge=1, le=60)
    risk: Literal["保守", "穩健", "積極"] = "穩健"


class ForecastIn(BaseModel):
    ticker: str = Field(..., max_length=12)
    horizon: int = Field(5, ge=1, le=60)


class PortfolioIn(BaseModel):
    holdings: dict[str, float] = Field(..., examples=[{"2330": 1000, "2317": 2000, "NVDA": 50}], min_length=1)
    horizon: int = Field(5, ge=1, le=60)
    cash: float = Field(0.0, ge=0, allow_inf_nan=False)
    base_currency: Literal["TWD", "USD"] = "TWD"

    @field_validator("holdings")
    @classmethod
    def _positive_finite(cls, v: dict[str, float]) -> dict[str, float]:
        import math
        bad = [k for k, x in v.items() if not (isinstance(x, (int, float)) and math.isfinite(x) and x > 0)]
        if bad:
            raise ValueError(f"持股股數必須為正數：{', '.join(bad[:5])}")
        if any(len(k.strip()) == 0 or len(k) > 12 for k in v):
            raise ValueError("股票代號格式錯誤")
        return v


def create_app(settings: Settings | None = None, provider: DataProvider | None = None, llm: LLMClient | None = None,
               tenants: TenantStore | None = None, audit: AuditLog | None = None,
               quant_panel: list[str] | None = None) -> FastAPI:
    s = settings or get_settings()
    if provider is None:
        from ..data import LiveDataProvider
        provider = LiveDataProvider(s)
    tenants = tenants or TenantStore(s)
    audit = audit or AuditLog(s)
    app = FastAPI(title="Finpre API", version=__version__,
                  description="每個訊號都附成績單的 AI 投資研究 API。研究模式不提供個股買賣建議；顧問模式限持牌投顧機構。")

    def tenant_dep(x_api_key: str | None = Header(None), authorization: str | None = Header(None)) -> Tenant:
        key = x_api_key or (authorization[7:] if authorization and authorization.lower().startswith("bearer ") else None)
        t = tenants.authenticate(key)
        if t is None:
            raise HTTPException(401, "缺少或無效的 API 金鑰")
        return t

    @contextmanager
    def metered(t: Tenant, endpoint: str, analysis: bool = False):
        """Count the call against the plan; give it back if the request then fails (typos don't eat the quota)."""
        try:
            token = tenants.charge(t, endpoint, analysis)
        except QuotaExceeded as e:
            raise HTTPException(429, str(e))
        try:
            yield
        except BaseException:
            tenants.refund(token)
            raise

    def record(t: Tenant, kind: str, subject: str, request, response, compliance=None) -> None:
        try:
            audit.record(actor=f"tenant:{t.id}", mode=t.mode, kind=kind, subject=subject, request=request,
                         response=response, compliance=compliance)
        except Exception as e:  # never lose an answer because of the audit log, but make it loud
            log.error("audit write failed: %s", e)

    def guard(t: Tenant, obj):
        return ComplianceGuard(t.mode).apply(obj)

    @app.exception_handler(ValueError)
    async def bad_input(_: Request, exc: ValueError):
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    @app.middleware("http")
    async def headers(request: Request, call_next):
        resp = await call_next(request)
        resp.headers["X-Finpre-Version"] = __version__
        return resp

    @app.get("/v1/health", tags=["system"])
    def health():
        return {"status": "ok", "version": __version__, "time": datetime.now(timezone.utc).isoformat(timespec="seconds")}

    @app.get("/v1/plans", tags=["system"])
    def plans():
        return [{"key": p.key, "name": p.name, "price_twd_month": p.price_twd_month,
                 "analyses_per_day": p.analyses_per_day, "api_calls_per_month": p.api_calls_per_month,
                 "watchlist_max": p.watchlist_max, "portfolio_positions_max": p.portfolio_positions_max,
                 "advisor_mode": p.advisor_mode} for p in PLANS.values()]

    @app.get("/v1/me", tags=["account"])
    def me(t: Tenant = Depends(tenant_dep)):
        return t.to_dict() | {"mode_label": MODE_LABEL[t.mode], "usage": tenants.usage(t.id)}

    # A full analysis takes 10–30 s (data + three analysts + LLM). Identical requests in the same compliance mode
    # within `product.analysis_cache_minutes` are served from memory: still metered and audited, marked cached.
    ttl = float(s.get_path("product.analysis_cache_minutes", 30) or 0) * 60
    cache: dict[tuple, tuple[float, dict]] = {}
    cache_lock = threading.Lock()

    @app.post("/v1/analyze", tags=["research"])
    def analyze(body: AnalyzeIn, t: Tenant = Depends(tenant_dep)):
        key = (body.ticker.strip().upper(), body.horizon, body.risk, t.mode)
        with metered(t, "analyze", analysis=True):
            now = time.time()
            with cache_lock:
                hit = cache.get(key)
            if ttl > 0 and hit and now - hit[0] < ttl:
                out = copy.deepcopy(hit[1]) | {"cached": True, "cache_age_s": round(now - hit[0], 1)}
                record(t, "analyze", out.get("ticker", key[0]), body.model_dump() | {"cached": True}, out, hit[2])
                return out
            adv = InvestmentAdvisor(provider, s, llm=llm, mode=t.mode, actor=f"tenant:{t.id}", audit=audit,
                                    quant_panel=quant_panel)
            res = adv.analyze(body.ticker, body.horizon, ClientProfile(risk=body.risk))
            out = res.brief() | {"compliance": res.compliance, "elapsed_s": res.elapsed_s, "cached": False}
            full = res.audit_compliance
            if ttl > 0:
                with cache_lock:
                    for k in [k for k, v in cache.items() if now - v[0] >= ttl]:
                        del cache[k]
                    cache[key] = (now, copy.deepcopy(out), full)
            return out

    @app.post("/v1/forecast", tags=["research"])
    def forecast(body: ForecastIn, t: Tenant = Depends(tenant_dep)):
        with metered(t, "forecast"):
            fc = forecast_ticker(provider, body.ticker, body.horizon, s)
            out = fc.to_dict() | {"quantile_returns_pct": {str(q): round(v * 100, 3) for q, v in fc.quantile_returns.items()},
                                  "disclaimer": RESEARCH_DISCLAIMER}
            out, rep = guard(t, out)
            record(t, "forecast", fc.symbol.code, body.model_dump(), out, rep.to_dict())
            return out

    @app.get("/v1/ranking/{market}", tags=["research"])
    def ranking(market: Literal["TW", "US"], horizon: int = Query(5, ge=1, le=60), t: Tenant = Depends(tenant_dep)):
        runs = s.resolve_path("evaluation.runs_dir")
        path = next((runs / f"ranking_{market}_h{h}.json" for h in (horizon, 5, 20)
                     if (runs / f"ranking_{market}_h{h}.json").exists()), None)
        if path is None:
            raise HTTPException(404, "尚未產生排名（請先執行 scripts/rank_stocks.py 或每日排程）")
        with metered(t, "ranking"):
            snap = json.loads(path.read_text())
            if t.plan.ranking_rows:
                snap["ranks"] = snap["ranks"][:t.plan.ranking_rows]
                snap["note"] = f"{t.plan.name}方案僅顯示前 {t.plan.ranking_rows} 名"
            snap["disclaimer"] = "排名為統計模型的研究結果，不是買賣建議。" + RESEARCH_DISCLAIMER
            out, rep = guard(t, snap)
            record(t, "ranking", market, {"horizon": horizon}, out, rep.to_dict())
            return out

    @app.post("/v1/portfolio/risk", tags=["research"])
    def portfolio(body: PortfolioIn, t: Tenant = Depends(tenant_dep)):
        if len(body.holdings) > t.plan.portfolio_positions_max:
            raise HTTPException(403, f"{t.plan.name}最多 {t.plan.portfolio_positions_max} 檔持股")
        with metered(t, "portfolio"):
            out = portfolio_risk(provider, body.holdings, body.horizon, body.base_currency, s, cash=body.cash,
                                 max_positions=t.plan.portfolio_positions_max)
            out, rep = guard(t, out)
            record(t, "portfolio", ",".join(list(body.holdings)[:20]), body.model_dump(), out, rep.to_dict())
            return out

    @app.get("/v1/report/{market}", tags=["research"])
    def report(market: Literal["TW", "US"], horizon: int = Query(5, ge=1, le=60),
               watchlist: str = Query("", max_length=2000, description="逗號分隔，例如 2330,2317"),
               format: Literal["json", "html", "md"] = "json", t: Tenant = Depends(tenant_dep)):
        wl = list(dict.fromkeys(x.strip().upper() for x in watchlist.split(",") if x.strip()))
        if len(wl) > t.plan.watchlist_max:
            raise HTTPException(403, f"{t.plan.name}觀察清單最多 {t.plan.watchlist_max} 檔")
        with metered(t, "report"):
            rows = t.plan.ranking_rows or 10
            rep = build_report(provider, market, horizon, wl, s, top_n=rows)
            if t.plan.ranking_rows and rep.get("ranking"):
                rep["ranking"]["bottom"] = []
                rep["ranking"]["note"] = f"{t.plan.name}方案僅顯示前 {t.plan.ranking_rows} 名"
            full = rep.pop("compliance_full", None)
            record(t, "report", market, {"horizon": horizon, "watchlist": wl, "format": format}, rep, full)
            if format == "html":
                return HTMLResponse(render_html(rep))
            if format == "md":
                return PlainTextResponse(render_markdown(rep), media_type="text/markdown; charset=utf-8")
            return rep

    @app.get("/v1/scorecard", tags=["transparency"])
    def scorecard(t: Tenant = Depends(tenant_dep)):
        with metered(t, "scorecard"):
            out = {"backtest": backtest_scorecard(), "live": live_scorecard(s)}
            record(t, "scorecard", "", {}, out)
            return out

    @app.get("/v1/usage", tags=["account"])
    def usage(month: str | None = Query(None, pattern=r"^\d{4}-\d{2}$"), t: Tenant = Depends(tenant_dep)):
        return tenants.usage(t.id, month)

    @app.get("/v1/audit/export", tags=["compliance"])
    def audit_export(t: Tenant = Depends(tenant_dep)):
        if not t.plan.audit_export:
            raise HTTPException(403, "稽核紀錄匯出僅限企業版")
        return StreamingResponse((line + "\n" for line in audit.export(actor=f"tenant:{t.id}")),
                                 media_type="application/x-ndjson")

    @app.get("/", include_in_schema=False)
    def root():
        return HTMLResponse("<h1>Finpre API</h1><p>文件：<a href='/docs'>/docs</a></p>")

    return app
