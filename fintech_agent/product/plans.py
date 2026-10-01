"""Plans, tenants (API customers), API keys and usage metering.

Tenants and usage live in SQLite (runs/tenants.sqlite). API keys are shown once at creation and stored only as
SHA-256 hashes. Advisor mode (buy/sell wording) can only be enabled for the enterprise plan AND a tenant flagged
as a licensed investment-consulting firm — the check lives here so no endpoint can bypass it.
"""
from __future__ import annotations

import hashlib
import secrets
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from ..config import Settings, get_settings


@dataclass(frozen=True)
class Plan:
    key: str
    name: str
    price_twd_month: int
    analyses_per_day: int          # full multi-agent analyses
    api_calls_per_month: int       # all metered API calls
    watchlist_max: int
    portfolio_positions_max: int
    advisor_mode: bool = False     # may use advisor mode (still requires a licensed tenant)
    audit_export: bool = False
    ranking_rows: int | None = None  # None = full ranking; free plan sees the top few only


PLANS: dict[str, Plan] = {
    "free": Plan("free", "免費", 0, 3, 300, 5, 5, ranking_rows=5),
    "research": Plan("research", "研究版", 690, 50, 3_000, 30, 30),
    "pro": Plan("pro", "專業版", 1_990, 200, 5_000, 200, 60),
    "enterprise": Plan("enterprise", "企業版", 60_000, 10_000, 200_000, 2_000, 500, advisor_mode=True, audit_export=True),
}

SCHEMA = """
CREATE TABLE IF NOT EXISTS tenants (
  id TEXT PRIMARY KEY, name TEXT NOT NULL, plan TEXT NOT NULL, licensed INTEGER NOT NULL DEFAULT 0,
  mode TEXT NOT NULL DEFAULT 'research', key_hash TEXT UNIQUE NOT NULL, key_prefix TEXT NOT NULL,
  active INTEGER NOT NULL DEFAULT 1, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS usage (
  tenant_id TEXT NOT NULL, day TEXT NOT NULL, endpoint TEXT NOT NULL, n INTEGER NOT NULL DEFAULT 0,
  PRIMARY KEY (tenant_id, day, endpoint));
"""
ANALYSIS_ENDPOINTS = ("analyze", "web:analyze", "web:chat")     # count against the daily full-analysis limit


class QuotaExceeded(Exception):
    pass


@dataclass
class Tenant:
    id: str
    name: str
    plan: Plan
    licensed: bool
    mode: str
    key_prefix: str
    active: bool

    def to_dict(self) -> dict:
        p = self.plan
        return {"id": self.id, "name": self.name, "plan": p.key, "plan_name": p.name, "licensed": self.licensed,
                "mode": self.mode, "key_prefix": self.key_prefix, "active": self.active,
                "limits": {"analyses_per_day": p.analyses_per_day, "api_calls_per_month": p.api_calls_per_month,
                           "watchlist_max": p.watchlist_max, "portfolio_positions_max": p.portfolio_positions_max}}


def _hash(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


class TenantStore:
    def __init__(self, settings: Settings | None = None, path: Path | str | None = None):
        s = settings or get_settings()
        self.path = Path(path) if path else s.resolve_path("evaluation.runs_dir") / "tenants.sqlite"
        with self._conn() as c:
            c.executescript(SCHEMA)

    def _conn(self):
        return sqlite3.connect(self.path, timeout=30)

    def create(self, name: str, plan: str = "free", licensed: bool = False, mode: str = "research") -> tuple[Tenant, str]:
        if plan not in PLANS:
            raise ValueError(f"unknown plan {plan!r}")
        self._check_mode(PLANS[plan], licensed, mode)
        key = "fp_" + secrets.token_urlsafe(32)
        tid = "t_" + secrets.token_hex(6)
        with self._conn() as c:
            c.execute("INSERT INTO tenants (id, name, plan, licensed, mode, key_hash, key_prefix, created_at) "
                      "VALUES (?,?,?,?,?,?,?,?)", (tid, name, plan, int(licensed), mode, _hash(key), key[:8],
                                                  datetime.now(timezone.utc).isoformat(timespec="seconds")))
        return self.get(tid), key

    @staticmethod
    def _check_mode(plan: Plan, licensed: bool, mode: str) -> None:
        if mode not in ("research", "advisor"):
            raise ValueError("mode must be research or advisor")
        if mode == "advisor" and not (plan.advisor_mode and licensed):
            raise ValueError("advisor mode requires the enterprise plan and a licensed investment-consulting firm")

    def set_plan(self, tenant_id: str, plan: str, licensed: bool | None = None, mode: str | None = None) -> Tenant:
        t = self.get(tenant_id)
        lic = t.licensed if licensed is None else licensed
        md = mode or ("research" if not (PLANS[plan].advisor_mode and lic) else t.mode)
        self._check_mode(PLANS[plan], lic, md)
        with self._conn() as c:
            c.execute("UPDATE tenants SET plan=?, licensed=?, mode=? WHERE id=?", (plan, int(lic), md, tenant_id))
        return self.get(tenant_id)

    def deactivate(self, tenant_id: str) -> None:
        with self._conn() as c:
            c.execute("UPDATE tenants SET active=0 WHERE id=?", (tenant_id,))

    def _row(self, where: str, arg) -> Tenant | None:
        with self._conn() as c:
            r = c.execute(f"SELECT id, name, plan, licensed, mode, key_prefix, active FROM tenants WHERE {where}", (arg,)).fetchone()
        if not r:
            return None
        return Tenant(r[0], r[1], PLANS[r[2]], bool(r[3]), r[4], r[5], bool(r[6]))

    def get(self, tenant_id: str) -> Tenant:
        t = self._row("id = ?", tenant_id)
        if t is None:
            raise KeyError(tenant_id)
        return t

    def authenticate(self, key: str | None) -> Tenant | None:
        if not key:
            return None
        t = self._row("key_hash = ?", _hash(key))
        return t if t and t.active else None

    def list(self) -> list[dict]:
        with self._conn() as c:
            ids = [r[0] for r in c.execute("SELECT id FROM tenants ORDER BY created_at")]
        return [self.get(i).to_dict() for i in ids]

    def rotate_key(self, tenant_id: str) -> str:
        """Issue a new API key; the old one stops working immediately."""
        self.get(tenant_id)
        key = "fp_" + secrets.token_urlsafe(32)
        with self._conn() as c:
            c.execute("UPDATE tenants SET key_hash=?, key_prefix=? WHERE id=?", (_hash(key), key[:8], tenant_id))
        return key

    # ------------------------------------------------------------------ metering
    @contextmanager
    def _write(self):
        c = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        try:
            c.execute("BEGIN IMMEDIATE")          # check-and-count is atomic across API worker processes
            yield c
            c.execute("COMMIT")
        except Exception:
            c.execute("ROLLBACK")
            raise
        finally:
            c.close()

    def charge(self, tenant: Tenant, endpoint: str, analysis: bool = False) -> tuple[str, str, str]:
        """Count one call; raise QuotaExceeded before counting if a limit would be exceeded. Returns a token for
        `refund()` (used when the request then fails on bad input, so typos don't eat the quota)."""
        now = datetime.now(timezone.utc)
        day, month = now.strftime("%Y-%m-%d"), now.strftime("%Y-%m")
        with self._write() as c:
            month_n = c.execute("SELECT COALESCE(SUM(n),0) FROM usage WHERE tenant_id=? AND day LIKE ?",
                                (tenant.id, month + "%")).fetchone()[0]
            if month_n >= tenant.plan.api_calls_per_month:
                raise QuotaExceeded(f"本月 API 呼叫已達 {tenant.plan.name} 上限 {tenant.plan.api_calls_per_month:,} 次")
            if analysis:
                q = ",".join("?" * len(ANALYSIS_ENDPOINTS))
                day_n = c.execute(f"SELECT COALESCE(SUM(n),0) FROM usage WHERE tenant_id=? AND day=? AND endpoint IN ({q})",
                                  (tenant.id, day, *ANALYSIS_ENDPOINTS)).fetchone()[0]
                if day_n >= tenant.plan.analyses_per_day:
                    raise QuotaExceeded(f"今日完整分析已達 {tenant.plan.name} 上限 {tenant.plan.analyses_per_day} 次")
            c.execute("INSERT INTO usage (tenant_id, day, endpoint, n) VALUES (?,?,?,1) "
                      "ON CONFLICT(tenant_id, day, endpoint) DO UPDATE SET n = n + 1", (tenant.id, day, endpoint))
        return tenant.id, day, endpoint

    def refund(self, token: tuple[str, str, str]) -> None:
        with self._write() as c:
            c.execute("UPDATE usage SET n = MAX(n - 1, 0) WHERE tenant_id=? AND day=? AND endpoint=?", token)

    def usage(self, tenant_id: str, month: str | None = None) -> dict:
        month = month or datetime.now(timezone.utc).strftime("%Y-%m")
        with self._conn() as c:
            rows = c.execute("SELECT day, endpoint, n FROM usage WHERE tenant_id=? AND day LIKE ? ORDER BY day",
                             (tenant_id, month + "%")).fetchall()
        by_ep: dict[str, int] = {}
        for _, ep, n in rows:
            by_ep[ep] = by_ep.get(ep, 0) + n
        return {"month": month, "total": sum(by_ep.values()), "by_endpoint": by_ep,
                "daily": [{"day": d, "endpoint": ep, "n": n} for d, ep, n in rows]}
