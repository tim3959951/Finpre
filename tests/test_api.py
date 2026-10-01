import json

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from fintech_agent.api import create_app  # noqa: E402
from fintech_agent.data import SyntheticProvider  # noqa: E402
from fintech_agent.llm import NullLLM  # noqa: E402
from fintech_agent.product.audit import AuditLog  # noqa: E402
from fintech_agent.product.compliance import ComplianceGuard  # noqa: E402
from fintech_agent.product.plans import TenantStore  # noqa: E402


@pytest.fixture
def env(tmp_path):
    tenants = TenantStore(path=tmp_path / "t.sqlite")
    audit = AuditLog(path=tmp_path / "a.sqlite")
    app = create_app(provider=SyntheticProvider(), llm=NullLLM(), tenants=tenants, audit=audit, quant_panel=["naive", "drift"])
    client = TestClient(app)
    free, free_key = tenants.create("個人用戶", "free")
    ent, ent_key = tenants.create("示範投顧", "enterprise", licensed=True, mode="advisor")
    return client, tenants, audit, free_key, ent_key


def H(k):
    return {"X-API-Key": k}


def test_auth_required(env):
    client, *_ = env
    assert client.get("/v1/health").status_code == 200
    assert client.get("/v1/me").status_code == 401
    assert client.post("/v1/analyze", json={"ticker": "2330"}, headers=H("fp_wrong")).status_code == 401


def test_research_tenant_gets_no_advice_and_is_audited(env):
    client, tenants, audit, free_key, _ = env
    r = client.post("/v1/analyze", json={"ticker": "2330", "horizon": 5}, headers=H(free_key))
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["mode"] == "research" and "signal" in body["decision"]
    assert ComplianceGuard("research").check(body) == []          # the whole body, compliance block included
    assert "removed_text" not in body["compliance"]
    assert audit.rows()[0]["kind"] == "analysis" and audit.verify()[0]


def test_licensed_enterprise_gets_advisor_output(env):
    client, _, _, _, ent_key = env
    body = client.post("/v1/analyze", json={"ticker": "2330"}, headers=H(ent_key)).json()
    assert body["mode"] == "advisor" and "action" in body["decision"] and "stop_loss" in body["decision"]


def test_advisor_mode_needs_license_and_enterprise(env):
    _, tenants, *_ = env
    with pytest.raises(ValueError):
        tenants.create("x", "pro", licensed=True, mode="advisor")
    with pytest.raises(ValueError):
        tenants.create("y", "enterprise", licensed=False, mode="advisor")


def test_quota_enforced(env):
    client, _, _, free_key, _ = env
    codes = [client.post("/v1/analyze", json={"ticker": "2330"}, headers=H(free_key)).status_code for _ in range(4)]
    assert codes == [200, 200, 200, 429]
    use = client.get("/v1/usage", headers=H(free_key)).json()
    assert use["by_endpoint"]["analyze"] == 3


def test_forecast_portfolio_report_scorecard(env):
    client, _, _, free_key, ent_key = env
    f = client.post("/v1/forecast", json={"ticker": "NVDA", "horizon": 20}, headers=H(free_key)).json()
    assert f["p10_pct"] < f["median_pct"] < f["p90_pct"] and 0 <= f["p_up"] <= 1
    p = client.post("/v1/portfolio/risk", json={"holdings": {"2330": 1000, "NVDA": 20}}, headers=H(free_key))
    assert p.status_code == 200, p.text
    pj = p.json()
    assert pj["var95_pct"] > 0 and abs(sum(x["weight"] for x in pj["positions"]) - 1) < 1e-3
    too_many = {str(1000 + i): 1 for i in range(6)}
    assert client.post("/v1/portfolio/risk", json={"holdings": too_many}, headers=H(free_key)).status_code == 403
    rep = client.get("/v1/report/TW?watchlist=2330,2317&format=json", headers=H(ent_key)).json()
    assert rep["market"] == "TW" and len(rep["watchlist"]) == 2
    html = client.get("/v1/report/TW?format=html", headers=H(ent_key))
    assert html.status_code == 200 and "<html" in html.text
    md = client.get("/v1/report/TW?format=md", headers=H(ent_key))
    assert md.text.startswith("# ")
    sc = client.get("/v1/scorecard", headers=H(ent_key)).json()
    assert "backtest" in sc and "live" in sc


def test_ranking_endpoint_free_plan_truncated(env, tmp_path):
    client, _, _, free_key, ent_key = env
    from fintech_agent.config import get_settings
    runs = get_settings().resolve_path("evaluation.runs_dir")
    snap = {"as_of": "2026-09-30", "model": "xs-lgbm-chips", "universe": "twse-pit50", "market": "TW", "horizon": 5,
            "n": 50, "backtest": {"ic_mean": 0.03, "ic_t": 3.4},
            "ranks": [{"ticker": str(1000 + i), "name": f"S{i}", "rank": i + 1, "score": 0.5 - i / 100} for i in range(50)]}
    (runs / "ranking_TW_h5.json").write_text(json.dumps(snap))
    assert len(client.get("/v1/ranking/TW", headers=H(free_key)).json()["ranks"]) == 5
    assert len(client.get("/v1/ranking/TW", headers=H(ent_key)).json()["ranks"]) == 50


def test_audit_export_enterprise_only(env):
    client, _, _, free_key, ent_key = env
    client.post("/v1/forecast", json={"ticker": "2330"}, headers=H(ent_key))
    assert client.get("/v1/audit/export", headers=H(free_key)).status_code == 403
    lines = client.get("/v1/audit/export", headers=H(ent_key)).text.strip().splitlines()
    assert lines and all(json.loads(x)["actor"].startswith("tenant:") for x in lines)


def test_analysis_cache_is_metered_and_audited(env):
    client, tenants, audit, free_key, _ = env
    a = client.post("/v1/analyze", json={"ticker": "2330"}, headers=H(free_key)).json()
    b = client.post("/v1/analyze", json={"ticker": "2330 "}, headers=H(free_key)).json()
    assert a["cached"] is False and b["cached"] is True
    assert a["decision"] == b["decision"]
    assert client.get("/v1/usage", headers=H(free_key)).json()["by_endpoint"]["analyze"] == 2
    kinds = [r["kind"] for r in audit.rows()]
    assert kinds.count("analyze") == 1 and kinds.count("analysis") == 1
    assert audit.verify()[0]


def _strict_env(tmp_path):
    from tests.test_hardening import StrictProvider
    tenants = TenantStore(path=tmp_path / "t2.sqlite")
    audit = AuditLog(path=tmp_path / "a2.sqlite")
    app = create_app(provider=StrictProvider(), llm=NullLLM(), tenants=tenants, audit=audit, quant_panel=["naive", "drift"])
    return TestClient(app), tenants


def test_failed_requests_do_not_use_quota(tmp_path):
    client, tenants = _strict_env(tmp_path)
    _, key = tenants.create("u", "free")
    assert client.post("/v1/analyze", json={"ticker": "23300"}, headers=H(key)).status_code == 400
    assert client.post("/v1/analyze", json={"ticker": "233O"}, headers=H(key)).status_code == 400
    assert client.get("/v1/usage", headers=H(key)).json()["total"] == 0
    assert client.post("/v1/analyze", json={"ticker": "2330"}, headers=H(key)).status_code == 200


def test_portfolio_input_validation(env):
    client, _, _, free_key, _ = env
    for bad in ({"2330": 0}, {"2330": -1}, {}):
        assert client.post("/v1/portfolio/risk", json={"holdings": bad}, headers=H(free_key)).status_code == 422


def test_report_json_with_ranking_and_free_plan_limits(env):
    client, _, _, free_key, ent_key = env
    from fintech_agent.config import get_settings
    runs = get_settings().resolve_path("evaluation.runs_dir")
    snap = {"as_of": "2020-01-02", "model": "xs-lgbm-chips", "universe": "twse-pit50", "market": "TW", "horizon": 5,
            "n": 12, "backtest": {"ic_mean": 0.03, "ic_t": 3.4},
            "ranks": [{"ticker": t, "name": t, "rank": i + 1, "score": 0.5 - i / 100}
                      for i, t in enumerate(["2330", "2317", "2454"] + [str(1100 + i) for i in range(9)])]}
    (runs / "ranking_TW_h5.json").write_text(json.dumps(snap))
    full = client.get("/v1/report/TW?format=json", headers=H(ent_key))
    assert full.status_code == 200, full.text
    rk = full.json()["ranking"]
    assert rk["stale"] is True and len(rk["top"]) == 10 and len(rk["bottom"]) == 2      # no duplicates
    free = client.get("/v1/report/TW?format=json", headers=H(free_key)).json()["ranking"]
    assert len(free["top"]) == 5 and free["bottom"] == []
    md = client.get("/v1/report/TW?format=md", headers=H(free_key)).text
    assert "注意：排名資料已" in md and "前 5 名" in md
