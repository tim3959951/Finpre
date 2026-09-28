import numpy as np
import pandas as pd
import pytest

from fintech_agent.data import SyntheticProvider, parse_symbol
from fintech_agent.data.covariates import align_series, build_covariates, chip_columns, covariate_features
from fintech_agent.evaluation.backtest import BacktestConfig, run_backtest
from fintech_agent.forecasting.base import QUANTILES
from fintech_agent.forecasting.baselines import NaiveForecaster


def test_align_strictly_before_avoids_same_day_us_close():
    tw = pd.DatetimeIndex(["2026-01-05", "2026-01-06"])
    us = pd.Series([100.0, 200.0, 300.0], index=pd.DatetimeIndex(["2026-01-02", "2026-01-05", "2026-01-06"]))
    same = align_series(us, tw, strictly_before=False)
    before = align_series(us, tw, strictly_before=True)
    assert list(same) == [200.0, 300.0]
    assert list(before) == [100.0, 200.0]          # TW 1/6 only knows the US close of 1/5


def test_chip_columns_cumulate_and_scale():
    idx = pd.bdate_range("2026-01-01", periods=5)
    vol = pd.Series(1000.0, index=idx)
    inst = pd.DataFrame([{"date": d, "name": "Foreign_Investor", "buy": 600, "sell": 100} for d in idx])
    marg = pd.DataFrame([{"date": d, "MarginPurchaseTodayBalance": 100 + i} for i, d in enumerate(idx)])
    out = chip_columns(idx, vol, inst, marg)
    assert out["foreign_flow_lvl"].iloc[-1] == pytest.approx(2.5)   # 5 days x 500 / 1000
    assert "margin_z" in out


def test_build_covariates_shape_and_features():
    p = SyntheticProvider()
    s = parse_symbol("2330")
    px = p.prices(s)
    cov = p.covariates(s, px)
    assert len(cov) == len(px) and not cov.isna().any().any()
    f = covariate_features(cov, len(cov) - 1)
    assert len(f) == 1 + 3 * (len(cov.columns) - 1)                  # vol_z raw + 3 lags per *_lvl


class _PeekCov(NaiveForecaster):
    name, uses_covariates = "peek-cov", True

    def __init__(self):
        self.last_rows = []

    def predict(self, contexts, horizon, covariates=None):
        self.last_rows += [c.index[-1] for c in covariates]
        return super().predict(contexts, horizon)


def test_backtest_passes_only_past_covariates():
    p = SyntheticProvider()
    prices, covs = {}, {}
    for t in ["2330", "2317"]:
        px = p.prices(parse_symbol(t))
        prices[t], covs[t] = px["close"], p.covariates(parse_symbol(t), px)
    peek = _PeekCov()
    w, _ = run_backtest(prices, {"naive": NaiveForecaster(), "peek": peek}, BacktestConfig(horizon=5, n_windows=10),
                        covariates=covs)
    rows = w[w["model"] == "peek"].reset_index(drop=True)
    for (_, r), last in zip(rows.iterrows(), peek.last_rows):
        assert last < r["origin"]                                     # strictly before the first forecast day


def test_lgbm_cov_trains_and_predicts():
    pytest.importorskip("lightgbm")
    from fintech_agent.forecasting.ml import LGBMCovForecaster
    p = SyntheticProvider()
    ser, covs = [], []
    for t in ["A1", "B2", "C3"]:
        px = p.prices(parse_symbol(t))
        ser.append(px["close"].to_numpy())
        covs.append(p.covariates(parse_symbol(t), px))
    m = LGBMCovForecaster(horizon=5, n_estimators=20).fit([s[:-50] for s in ser], [c.iloc[:-50] for c in covs])
    r = m.predict([s[-300:] for s in ser], 5, covariates=[c.iloc[-305:] for c in covs])
    assert len(r) == 3 and set(r[0].quantiles) == set(QUANTILES)


def test_lgbm_checkpoint_roundtrip_and_load_trained(tmp_path, monkeypatch):
    pytest.importorskip("lightgbm")
    from fintech_agent.config import get_settings
    from fintech_agent.forecasting import registry
    from fintech_agent.forecasting.ml import LGBMCovForecaster
    s = get_settings()
    monkeypatch.setitem(s["forecasting"], "checkpoints_dir", str(tmp_path))
    p = SyntheticProvider()
    ser, covs = [], []
    for t in ["A1", "B2"]:
        px = p.prices(parse_symbol(t))
        ser.append(px["close"].to_numpy())
        covs.append(p.covariates(parse_symbol(t), px))
    m = LGBMCovForecaster(horizon=5, n_estimators=10).fit(ser, covs)
    m.save(registry.checkpoint_path(s, "lgbm-cov", "TW", 5))
    assert registry.load_trained("lgbm-cov", "US", 5, s) is None
    loaded = registry.load_trained("lgbm-cov", "TW", 5, s)
    a = m.predict([ser[0][-300:]], 5, covariates=[covs[0].iloc[-305:]])[0]
    b = loaded.predict([ser[0][-300:]], 5, covariates=[covs[0].iloc[-305:]])[0]
    assert np.allclose(a.point, b.point)


def test_champion_lookup_order(tmp_path, monkeypatch):
    from fintech_agent.config import get_settings
    from fintech_agent.evaluation.experiments import ExperimentStore
    s = get_settings()
    monkeypatch.setitem(s["evaluation"], "runs_dir", str(tmp_path))
    monkeypatch.setitem(s["forecasting"], "champions", {"TW": {5: "lgbm-cov"}})
    st = ExperimentStore(s)
    assert st.champion("TW", 5) == "lgbm-cov"             # settings table
    assert st.champion("TW", 10) == st.default_champion   # horizon not listed
    assert st.champion("US", 5) == st.default_champion    # market not listed
    st.set_champion("TW", 5, "chronos-2", "test")
    assert st.champion("TW", 5) == "chronos-2"            # local promotion wins


def test_untrained_champion_falls_back(tmp_path, monkeypatch):
    from fintech_agent.agents.base import AnalysisContext
    from fintech_agent.agents.specialists import QuantAgent
    from fintech_agent.config import get_settings
    s = get_settings()
    monkeypatch.setitem(s["evaluation"], "runs_dir", str(tmp_path / "runs"))
    monkeypatch.setitem(s["forecasting"], "checkpoints_dir", str(tmp_path / "ck"))
    monkeypatch.setitem(s["forecasting"], "champions", {"TW": {5: "lgbm"}})
    monkeypatch.setitem(s["forecasting"], "panel", ["naive"])
    prov = SyntheticProvider()
    q = QuantAgent(prov, None, s, skill_windows=3)
    q.store.default_champion = "drift"                    # keep the test offline (no model download)
    sym = parse_symbol("2330")
    ctx = AnalysisContext(sym, prov.prices(sym), prov.profile(sym), 5)
    ev = q.gather(ctx)[0]
    assert ev["champion"] == "drift"
    assert "lgbm" not in ev["forecasts"]
