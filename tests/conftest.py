import pytest


@pytest.fixture(autouse=True)
def _isolated_state(tmp_path, monkeypatch):
    """Tests never read or write the real runs/ or checkpoints/ folders (audit logs, experiments, trained models)."""
    from fintech_agent.config import get_settings
    from fintech_agent.forecasting import registry
    s = get_settings()
    monkeypatch.setitem(s["evaluation"], "runs_dir", str(tmp_path / "runs"))
    monkeypatch.setitem(s["forecasting"], "checkpoints_dir", str(tmp_path / "checkpoints"))
    monkeypatch.setattr(registry, "_CACHE", {})
    yield
