"""tests/python/test_pipeline.py — pytest suite for alpha_research Python modules.

Tests cover:
  - data.py: synthetic generation, numpy extraction, Polars DataFrame shapes
  - signals.py: PDRRM signal computation, z-score properties, DecayMonitor
  - backtest.py: BacktestOrchestrator full run, KPI sanity checks

All tests use use_synthetic=True and small n_days for CI speed.

Author: Alpha Research Pod — 2026
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest

# Ensure package is importable without installation (editable mode / Bazel PYTHONPATH)
sys.path.insert(0, str(Path(__file__).parents[2] / "src" / "python"))

from alpha_research.data import (
    MarketDataConfig,
    generate_synthetic_data,
    extract_numpy_panels,
    G10_FX_SYMBOLS,
    ENERGY_SYMBOLS,
    CROSS_ASSET_SYMBOLS,
)
from alpha_research.signals import (
    compute_pdrrm_signals,
    compute_master_signal,
    DecayMonitor,
    _zscore_panel,
    _rolling_vol,
    _compute_rrdm_py,
    _compute_pss_py,
    _compute_rac_py,
)
from alpha_research.backtest import BacktestOrchestrator


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

N_DAYS      = 300
N_CURRENCIES= 5


@pytest.fixture(scope="session")
def synthetic_data():
    return generate_synthetic_data(n_days=N_DAYS, n_currencies=N_CURRENCIES, seed=0)


@pytest.fixture(scope="session")
def panels(synthetic_data):
    return extract_numpy_panels(synthetic_data, n_currencies=N_CURRENCIES)


@pytest.fixture(scope="session")
def dates():
    from alpha_research.data import _make_trading_days
    return _make_trading_days(N_DAYS)


# ─────────────────────────────────────────────────────────────────────────────
# data.py tests
# ─────────────────────────────────────────────────────────────────────────────


class TestGenerateSyntheticData:
    def test_returns_all_keys(self, synthetic_data):
        required = {"fx_returns", "nominal_rates", "breakevens", "cb_surprises",
                    "forward_premia", "yield_curve", "vol_surface",
                    "energy", "equity", "cross_asset"}
        assert required.issubset(synthetic_data.keys())

    def test_fx_returns_shape(self, synthetic_data):
        df = synthetic_data["fx_returns"]
        assert isinstance(df, pl.DataFrame)
        assert len(df) == N_DAYS
        assert df.shape[1] == N_CURRENCIES + 1  # +1 for date

    def test_no_nulls_in_returns(self, synthetic_data):
        df = synthetic_data["fx_returns"]
        for col in G10_FX_SYMBOLS[:N_CURRENCIES]:
            assert df[col].null_count() == 0

    def test_nominal_rates_positive(self, synthetic_data):
        df = synthetic_data["nominal_rates"]
        for col in G10_FX_SYMBOLS[:N_CURRENCIES]:
            assert (df[col] > 0).all()

    def test_vix_in_plausible_range(self, synthetic_data):
        vix = synthetic_data["vol_surface"]["vix_spot"]
        assert vix.min() >= 5
        assert vix.max() <= 200

    def test_seed_reproducibility(self):
        d1 = generate_synthetic_data(n_days=100, seed=42)
        d2 = generate_synthetic_data(n_days=100, seed=42)
        diff = (d1["fx_returns"]["6J"] - d2["fx_returns"]["6J"]).abs().sum()
        assert diff == pytest.approx(0.0)

    def test_different_seeds_differ(self):
        d1 = generate_synthetic_data(n_days=100, seed=1)
        d2 = generate_synthetic_data(n_days=100, seed=2)
        diff = (d1["fx_returns"]["6J"] - d2["fx_returns"]["6J"]).abs().sum()
        assert diff > 0


class TestExtractNumpyPanels:
    def test_shapes(self, panels):
        assert panels["fx_returns"].shape    == (N_DAYS, N_CURRENCIES)
        assert panels["nominal_rates"].shape == (N_DAYS, N_CURRENCIES)
        assert panels["breakevens"].shape    == (N_DAYS, N_CURRENCIES)
        assert panels["tp_acm"].shape        == (N_DAYS,)
        assert panels["energy_returns"].shape== (N_DAYS, 3)
        assert panels["ca_returns"].shape    == (N_DAYS, 5)

    def test_dtype_float64(self, panels):
        for key, arr in panels.items():
            assert arr.dtype == np.float64, f"{key} is not float64"

    def test_c_contiguous(self, panels):
        for key, arr in panels.items():
            assert arr.flags["C_CONTIGUOUS"], f"{key} not C-contiguous"

    def test_no_nans(self, panels):
        for key, arr in panels.items():
            assert not np.isnan(arr).any(), f"NaN found in {key}"


# ─────────────────────────────────────────────────────────────────────────────
# signals.py utility function tests
# ─────────────────────────────────────────────────────────────────────────────


class TestZScorePanel:
    def test_zero_mean_per_row(self):
        arr = np.random.default_rng(0).normal(5, 2, (50, 8))
        z = _zscore_panel(arr)
        row_means = z.mean(axis=1)
        np.testing.assert_allclose(row_means, 0.0, atol=1e-10)

    def test_unit_variance_per_row(self):
        arr = np.random.default_rng(1).normal(0, 3, (50, 8))
        z = _zscore_panel(arr)
        row_vars = z.var(axis=1)
        np.testing.assert_allclose(row_vars, 1.0, atol=1e-8)

    def test_constant_row_returns_zero(self):
        arr = np.ones((10, 5)) * 7.0
        z = _zscore_panel(arr)
        np.testing.assert_allclose(z, 0.0, atol=1e-10)


class TestRollingVol:
    def test_shape_preserved(self, panels):
        rv = _rolling_vol(panels["fx_returns"], window=21)
        assert rv.shape == panels["fx_returns"].shape

    def test_no_nans_after_warmup(self, panels):
        rv = _rolling_vol(panels["fx_returns"], window=21)
        assert not np.isnan(rv[21:]).any()

    def test_positive_vol(self, panels):
        rv = _rolling_vol(panels["fx_returns"], window=21)
        assert (rv[21:] > 0).all()


class TestPureRRDM:
    def test_shape(self, panels):
        rrdm = _compute_rrdm_py(panels["nominal_rates"], panels["breakevens"])
        assert rrdm.shape == panels["nominal_rates"].shape

    def test_warmup_zeros(self, panels):
        rrdm = _compute_rrdm_py(panels["nominal_rates"], panels["breakevens"], window=20)
        np.testing.assert_allclose(rrdm[:20], 0.0, atol=1e-15)

    def test_usd_anchor_near_zero(self, panels):
        rrdm = _compute_rrdm_py(panels["nominal_rates"], panels["breakevens"])
        # Column 0 is USD vs itself; differential = 0 → z-score of 0 is near 0
        # (z-scoring across N cols means it won't be exactly 0, but should be small)
        assert np.abs(rrdm[20:, 0]).mean() < 3.0


class TestPurePSS:
    def test_shape(self, panels):
        pss = _compute_pss_py(panels["cb_surprises"])
        assert pss.shape == panels["cb_surprises"].shape

    def test_zero_surprises_near_zero_output(self):
        surprises = np.zeros((100, 5))
        pss = _compute_pss_py(surprises)
        np.testing.assert_allclose(pss, 0.0, atol=1e-15)

    def test_surprise_propagates(self):
        surprises = np.zeros((100, 3))
        surprises[10, 1] = 0.005   # CB surprise at day 10, currency 1
        pss = _compute_pss_py(surprises, halflife=10.0)
        # PSS should be non-zero after day 10 for currency 1
        assert pss[15, 1] != 0.0
        # And decaying
        assert abs(pss[20, 1]) < abs(pss[11, 1]) + 1e-10


class TestPureRAC:
    def test_shape(self, panels):
        rac = _compute_rac_py(panels["forward_premia"], panels["fx_returns"])
        assert rac.shape == panels["fx_returns"].shape

    def test_warmup_zeros(self, panels):
        rac = _compute_rac_py(panels["forward_premia"], panels["fx_returns"], vol_window=21)
        np.testing.assert_allclose(rac[:21], 0.0, atol=1e-15)


# ─────────────────────────────────────────────────────────────────────────────
# PDRRM full signal tests
# ─────────────────────────────────────────────────────────────────────────────


class TestComputePDRRMSignals:
    def test_output_shapes(self, panels):
        sigs, pos = compute_pdrrm_signals(
            nominal_rates  = panels["nominal_rates"],
            breakevens     = panels["breakevens"],
            cb_surprises   = panels["cb_surprises"],
            forward_premia = panels["forward_premia"],
            fx_returns     = panels["fx_returns"],
        )
        T, N = panels["fx_returns"].shape
        assert sigs.shape == (T, N)
        assert pos.shape  == (T, N)

    def test_positions_clamped(self, panels):
        _, pos = compute_pdrrm_signals(
            nominal_rates  = panels["nominal_rates"],
            breakevens     = panels["breakevens"],
            cb_surprises   = panels["cb_surprises"],
            forward_premia = panels["forward_premia"],
            fx_returns     = panels["fx_returns"],
        )
        assert (np.abs(pos) <= 0.25 + 1e-10).all()

    def test_no_nans_in_signals(self, panels):
        sigs, pos = compute_pdrrm_signals(
            nominal_rates  = panels["nominal_rates"],
            breakevens     = panels["breakevens"],
            cb_surprises   = panels["cb_surprises"],
            forward_premia = panels["forward_premia"],
            fx_returns     = panels["fx_returns"],
        )
        assert not np.isnan(sigs).any()
        assert not np.isnan(pos).any()


# ─────────────────────────────────────────────────────────────────────────────
# Master signal & DecayMonitor tests
# ─────────────────────────────────────────────────────────────────────────────


class TestComputeMasterSignal:
    @pytest.fixture(scope="class")
    def bundles(self, panels, dates):
        return compute_master_signal(panels, dates)

    def test_all_strategies_present(self, bundles):
        for strat in ("PDRRM", "TPMCR", "MAERM", "ISRC", "VSRA", "FDSP"):
            assert strat in bundles, f"{strat} missing from bundles"

    def test_signals_not_all_zero(self, bundles):
        for name, bundle in bundles.items():
            assert np.abs(bundle.signals).sum() > 0, f"{name} signals all zero"

    def test_positions_finite(self, bundles):
        for name, bundle in bundles.items():
            assert np.isfinite(bundle.positions).all(), f"{name} positions not finite"


class TestDecayMonitor:
    def test_no_alert_on_healthy_ic(self):
        dm = DecayMonitor("TEST")
        alerts = [dm.update(0.07) for _ in range(60)]
        assert not alerts[-1]

    def test_alert_on_near_zero_ic(self):
        dm = DecayMonitor("TEST")
        for _ in range(60):
            dm.update(0.06)
        # Now decay
        alert = False
        for _ in range(30):
            alert = dm.update(0.005)
        assert alert

    def test_half_life_returns_float(self):
        dm = DecayMonitor("TEST")
        for i in range(40):
            dm.update(0.06 * (0.97 ** i))
        hl = dm.half_life()
        assert isinstance(hl, float)

    def test_slack_webhook_none_does_not_raise(self):
        dm = DecayMonitor("TEST", slack_webhook=None)
        dm.update(0.0)   # Should not raise even with no webhook


# ─────────────────────────────────────────────────────────────────────────────
# BacktestOrchestrator tests
# ─────────────────────────────────────────────────────────────────────────────


class TestBacktestOrchestrator:
    @pytest.fixture(scope="class")
    def report(self):
        orc = BacktestOrchestrator(n_days=300, n_currencies=5)
        return orc.run()

    def test_report_is_dataframe(self, report):
        assert isinstance(report, pl.DataFrame)

    def test_report_has_all_strategies(self, report):
        strategies = set(report["strategy"].to_list())
        for s in ("PDRRM", "TPMCR", "MAERM", "ISRC", "VSRA", "FDSP"):
            assert s in strategies

    def test_sharpe_is_finite(self, report):
        if "sharpe" in report.columns:
            for v in report["sharpe"].to_list():
                assert math.isfinite(float(v))

    def test_mdd_between_0_and_100(self, report):
        if "mdd_%" in report.columns:
            for v in report["mdd_%"].to_list():
                assert 0.0 <= float(v) <= 100.0
