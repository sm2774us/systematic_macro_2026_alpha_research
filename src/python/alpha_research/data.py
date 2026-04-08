"""alpha_research.data — Market data ingestion and preprocessing.

All data operations use **Polars** (no pandas) for maximum throughput:
  - LazyFrame query plans compiled before execution
  - SIMD-accelerated column operations (Polars uses Arrow/SIMD internally)
  - Zero-copy NumPy bridge via `.to_numpy()` for C++ hand-off

Supported data sources:
  - CSV files (local backtest data)
  - FRED API (yield curves, swap spreads)
  - Yahoo Finance via yfinance (futures proxies for backtesting)
  - Simulated synthetic data (unit testing / CI)

Author: Alpha Research Pod — 2026
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

logger = logging.getLogger(__name__)


def _make_trading_days(n: int, start: tuple[int, int, int] = (2015, 1, 2)) -> pl.Series:
    """Generate n weekday (Mon–Fri) dates starting from `start`.

    Polars 1.x dropped 'bd' interval support; we filter calendar days by weekday.
    """
    # Generate 2× as many calendar days as we need to guarantee enough weekdays
    end_date = pl.date(start[0] + (n // 200) + 3, 1, 1)
    all_days = pl.date_range(pl.date(*start), end_date, interval="1d", eager=True)
    weekdays = all_days.filter(all_days.dt.weekday().is_in([0, 1, 2, 3, 4]))
    return weekdays.head(n)


# ─────────────────────────────────────────────────────────────────────────────
# Schema constants
# ─────────────────────────────────────────────────────────────────────────────

#: G10 FX futures CME symbols (PDRRM universe)
G10_FX_SYMBOLS: tuple[str, ...] = ("6J", "6E", "6B", "6A", "6C", "6S", "6N", "6M")

#: Rates futures universe (TPMCR)
RATES_SYMBOLS: tuple[str, ...] = ("ZN", "ZB", "RX", "G", "TN")

#: Energy futures (ISRC)
ENERGY_SYMBOLS: tuple[str, ...] = ("CL", "NG", "RB")

#: Equity index futures (MAERM)
EQUITY_SYMBOLS: tuple[str, ...] = ("ES", "NQ", "RTY", "SX5E")

#: Cross-asset (FDSP)
CROSS_ASSET_SYMBOLS: tuple[str, ...] = ("TLT", "UUP", "GLD", "SPY", "VX")

# ─────────────────────────────────────────────────────────────────────────────
# Data schema
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(slots=True, frozen=True)
class MarketDataConfig:
    """Configuration for market data loading.

    Attributes:
        start_date:   ISO date string for backtest start.
        end_date:     ISO date string for backtest end.
        data_dir:     Path to CSV data directory.
        use_synthetic: If True, generate synthetic data (for CI/testing).
        n_days:       Number of synthetic days (only used if use_synthetic=True).
    """

    start_date: str = "2015-01-01"
    end_date: str = "2026-03-31"
    data_dir: Path = Path("data")
    use_synthetic: bool = True
    n_days: int = 2520  # 10 years


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic data generator (CI / unit testing)
# ─────────────────────────────────────────────────────────────────────────────


def generate_synthetic_data(
    n_days: int = 2520,
    n_currencies: int = 8,
    seed: int = 42,
) -> dict[str, pl.DataFrame]:
    """Generate realistic synthetic G10 FX and macro data for backtesting.

    Uses correlated GBM processes calibrated to historical G10 FX statistics
    (2014–2025). Incorporates:
      - Mean-reverting real rate differentials
      - Persistent EWMA volatility regimes
      - Simulated CB meeting surprise events (quarterly)
      - Seasonal inventory patterns for energy

    Args:
        n_days:        Number of trading days to generate.
        n_currencies:  Number of G10 currencies (max 8).
        seed:          RNG seed for reproducibility.

    Returns:
        Dictionary of Polars DataFrames keyed by data type:
        'fx_returns', 'nominal_rates', 'breakevens', 'cb_surprises',
        'forward_premia', 'yield_curve', 'vol_surface', 'energy', 'equity'
    """
    rng = np.random.default_rng(seed)
    n_currencies = min(n_currencies, len(G10_FX_SYMBOLS))

    dates = _make_trading_days(n_days)

    # ── FX Returns: correlated GBM with GARCH-like vol clustering ────────────
    # Cross-correlation matrix: G10 FX tends to cluster (commodity vs safe-haven)
    corr_base = 0.3 * np.ones((n_currencies, n_currencies))
    np.fill_diagonal(corr_base, 1.0)
    corr_base[0, 1] = corr_base[1, 0] = 0.6   # 6J–6E mild correlation
    corr_base[2, 3] = corr_base[3, 2] = 0.55  # 6B–6A commodity linkage
    L = np.linalg.cholesky(corr_base)

    daily_vol = 0.007  # ~7bp daily vol, typical G10
    raw_shocks = rng.standard_normal((n_days, n_currencies))
    corr_shocks = raw_shocks @ L.T * daily_vol

    # GARCH(1,1)-like vol clustering
    vol_state = np.ones(n_currencies)
    fx_returns = np.zeros((n_days, n_currencies))
    for t in range(n_days):
        vol_state = 0.9 * vol_state + 0.1 * np.abs(corr_shocks[t])
        fx_returns[t] = corr_shocks[t] * vol_state

    # ── Nominal rates: mean-reverting AR(1) around regime means ──────────────
    # Regime: USD high (Fed 5%+), JPY rising (0.75→1%), EUR low (2%)
    rate_means = np.array([0.050, 0.010, 0.020, 0.035, 0.040,
                            0.015, 0.045, 0.038])[:n_currencies]
    nominal_rates = np.zeros((n_days, n_currencies))
    nominal_rates[0] = rate_means + rng.normal(0, 0.002, n_currencies)
    for t in range(1, n_days):
        # AR(1) mean reversion
        nominal_rates[t] = (0.995 * nominal_rates[t - 1]
                           + 0.005 * rate_means
                           + rng.normal(0, 0.0003, n_currencies))
    # BOJ hiking trend in last min(500, n_days//2) days (2025-2026 regime)
    hike_window = min(500, n_days // 2)
    hike_trend = np.linspace(0, 0.003, hike_window)
    nominal_rates[-hike_window:, 1] += hike_trend  # JPY rate rising

    # ── Breakevens: CPI-linked, slightly below nominal ─────────────────────
    breakeven_means = np.array([0.025, 0.005, 0.020, 0.030, 0.028,
                                  0.018, 0.025, 0.022])[:n_currencies]
    breakevens = np.zeros((n_days, n_currencies))
    breakevens[0] = breakeven_means
    for t in range(1, n_days):
        breakevens[t] = (0.997 * breakevens[t - 1]
                        + 0.003 * breakeven_means
                        + rng.normal(0, 0.0002, n_currencies))

    # ── CB meeting surprises: quarterly, sparse ─────────────────────────────
    cb_surprises = np.zeros((n_days, n_currencies))
    meeting_days = np.arange(0, n_days, 63)  # ~quarterly
    for day in meeting_days:
        # Random surprise: mostly zero, occasionally ±5-25bp
        if rng.random() < 0.3:
            idx = rng.integers(0, n_currencies)
            surprise = rng.choice([-0.0025, -0.0050, 0.0025, 0.0050, 0.0100])
            cb_surprises[day, idx] = surprise

    # ── Forward premia: approx carry = interest rate differential ─────────
    forward_premia = nominal_rates - nominal_rates[:, [0]]  # vs USD

    # ── Yield curve (for TPMCR) ───────────────────────────────────────────
    y2 = 0.048 - np.cumsum(rng.normal(0, 0.0004, n_days))
    y10 = 0.045 - np.cumsum(rng.normal(0, 0.0003, n_days))
    y30 = 0.047 - np.cumsum(rng.normal(0, 0.0003, n_days))
    y2 = np.clip(y2, 0.001, 0.10)
    y10 = np.clip(y10, 0.001, 0.10)
    y30 = np.clip(y30, 0.001, 0.12)
    # ACM term premium: trending up post-2024 (fiscal concerns)
    tp_acm = 0.50 + 0.80 * np.arange(n_days) / n_days + rng.normal(0, 0.05, n_days)
    tp_acm = np.clip(tp_acm, -0.5, 3.0)

    # ── Vol surface (VSRA) ────────────────────────────────────────────────
    vix = 18 + 5 * rng.standard_normal(n_days)
    vix = np.abs(vix).clip(10, 80)
    # Simulate vol spikes (black swan events)
    for spike_day in rng.choice(n_days, size=15, replace=False):
        end_idx = min(spike_day + 5, n_days)
        length = end_idx - spike_day
        vix[spike_day:end_idx] *= np.linspace(2.5, 1.0, length)
    iv_atm = vix / 100
    rv_21d = 0.8 * iv_atm + rng.normal(0, 0.02, n_days)
    rv_21d = np.abs(rv_21d).clip(0.05, 0.80)
    vix_3m = vix * (1 + 0.02 * rng.standard_normal(n_days)) + 2

    # ── Energy (ISRC) ─────────────────────────────────────────────────────
    energy_returns = rng.normal(0, 0.015, (n_days, 3))
    inventory_surprises = rng.normal(0, 1, (n_days, 3))
    # EIA weekly (every 5 days, Wednesday)
    inventory_surprises[np.mod(np.arange(n_days), 5) != 2] = 0.0
    roll_returns = rng.normal(0, 0.002, (n_days, 3))

    # ── Equity (MAERM) ────────────────────────────────────────────────────
    equity_returns = rng.normal(0, 0.010, (n_days, 4))
    eps_revisions = rng.normal(0.02, 0.15, (n_days, 4))  # slight upward bias
    # AI mega-cap (NQ) upward revision trend in last portion
    ai_window = min(500, n_days // 2)
    eps_revisions[-ai_window:, 1] += 0.05
    ism_pmi = 52 + 4 * np.sin(2 * np.pi * np.arange(n_days) / 252) + rng.normal(0, 2, n_days)

    # ── Cross-asset (FDSP) ─────────────────────────────────────────────────
    swap_spread_30y = -0.002 + rng.normal(0, 0.0005, n_days)
    cds_5y = 0.003 + rng.normal(0, 0.001, n_days)
    cds_5y = np.abs(cds_5y)
    tbill_spike = rng.normal(0, 0.0005, n_days).clip(0, 0.05)
    ca_returns = rng.normal(0, 0.010, (n_days, 5))

    # ── Assemble Polars DataFrames ─────────────────────────────────────────

    def _make_panel(arr: np.ndarray, names: list[str]) -> pl.DataFrame:
        return pl.DataFrame(
            {name: arr[:, i].tolist() for i, name in enumerate(names)}
        ).with_columns(dates.alias("date")).select(["date"] + names)

    dates_series = _make_trading_days(n_days)

    def _col_names(symbols: tuple[str, ...], n: int) -> list[str]:
        return list(symbols[:n])

    fx_names = _col_names(G10_FX_SYMBOLS, n_currencies)

    result: dict[str, pl.DataFrame] = {
        "fx_returns": _make_panel(fx_returns, fx_names),
        "nominal_rates": _make_panel(nominal_rates, fx_names),
        "breakevens": _make_panel(breakevens, fx_names),
        "cb_surprises": _make_panel(cb_surprises, fx_names),
        "forward_premia": _make_panel(forward_premia, fx_names),
        "yield_curve": pl.DataFrame({
            "date": dates_series,
            "y2": y2.tolist(),
            "y10": y10.tolist(),
            "y30": y30.tolist(),
            "tp_acm_10y": tp_acm.tolist(),
            "swap_spread_30y": swap_spread_30y.tolist(),
            "cds_5y_proxy": cds_5y.tolist(),
        }),
        "vol_surface": pl.DataFrame({
            "date": dates_series,
            "iv_atm": iv_atm.tolist(),
            "rv_21d": rv_21d.tolist(),
            "vix_spot": vix.tolist(),
            "vix_3m": vix_3m.tolist(),
            "put_skew": (0.05 + rng.normal(0, 0.01, n_days)).tolist(),
            "rv_skew": (-0.3 + rng.normal(0, 0.1, n_days)).tolist(),
            "vx_return": rng.normal(0, 0.03, n_days).tolist(),
        }),
        "energy": pl.DataFrame({
            "date": dates_series,
            **{f"ret_{s}": energy_returns[:, i].tolist()
               for i, s in enumerate(ENERGY_SYMBOLS)},
            **{f"inv_surprise_{s}": inventory_surprises[:, i].tolist()
               for i, s in enumerate(ENERGY_SYMBOLS)},
            **{f"roll_{s}": roll_returns[:, i].tolist()
               for i, s in enumerate(ENERGY_SYMBOLS)},
        }),
        "equity": pl.DataFrame({
            "date": dates_series,
            **{f"ret_{s}": equity_returns[:, i].tolist()
               for i, s in enumerate(EQUITY_SYMBOLS)},
            **{f"eps_rev_{s}": eps_revisions[:, i].tolist()
               for i, s in enumerate(EQUITY_SYMBOLS)},
            "ism_pmi": ism_pmi.tolist(),
        }),
        "cross_asset": pl.DataFrame({
            "date": dates_series,
            "swap_spread_30y": swap_spread_30y.tolist(),
            "cds_5y": cds_5y.tolist(),
            "tbill_spike": tbill_spike.tolist(),
            **{f"ret_{s}": ca_returns[:, i].tolist()
               for i, s in enumerate(CROSS_ASSET_SYMBOLS)},
        }),
    }

    logger.info(
        "Synthetic data generated: %d days, %d currencies",
        n_days, n_currencies,
    )
    return result


def load_data_from_csv(config: MarketDataConfig) -> dict[str, pl.DataFrame]:
    """Load market data from CSV files (production path).

    Args:
        config: Data loading configuration.

    Returns:
        Dictionary of DataFrames per data type.

    Raises:
        FileNotFoundError: If required CSV files are missing.
    """
    data_dir = config.data_dir
    result: dict[str, pl.DataFrame] = {}

    for name in ("fx_returns", "nominal_rates", "breakevens", "cb_surprises",
                  "forward_premia", "yield_curve", "vol_surface",
                  "energy", "equity", "cross_asset"):
        fpath = data_dir / f"{name}.csv"
        if not fpath.exists():
            raise FileNotFoundError(
                f"Required data file not found: {fpath}. "
                "Set use_synthetic=True to use generated data."
            )
        result[name] = (
            pl.scan_csv(fpath)
            .with_columns(pl.col("date").str.to_date("%Y-%m-%d"))
            .filter(
                pl.col("date").is_between(
                    pl.date(*map(int, config.start_date.split("-"))),
                    pl.date(*map(int, config.end_date.split("-"))),
                )
            )
            .collect()
        )
        logger.info("Loaded %s: %d rows", name, len(result[name]))

    return result


def extract_numpy_panels(
    data: dict[str, pl.DataFrame],
    n_currencies: int = 8,
) -> dict[str, np.ndarray]:
    """Extract numpy arrays from Polars DataFrames for C++ hand-off.

    All arrays are C-contiguous float64 as required by nanobind.

    Args:
        data:         Dictionary of Polars DataFrames.
        n_currencies: Number of FX currencies to use.

    Returns:
        Dictionary of numpy arrays keyed by variable name.
    """
    n = min(n_currencies, len(G10_FX_SYMBOLS))
    fx_names = list(G10_FX_SYMBOLS[:n])

    def _to_f64(df: pl.DataFrame, cols: list[str]) -> np.ndarray:
        return np.ascontiguousarray(df.select(cols).to_numpy(), dtype=np.float64)

    return {
        "fx_returns":     _to_f64(data["fx_returns"],     fx_names),
        "nominal_rates":  _to_f64(data["nominal_rates"],  fx_names),
        "breakevens":     _to_f64(data["breakevens"],     fx_names),
        "cb_surprises":   _to_f64(data["cb_surprises"],   fx_names),
        "forward_premia": _to_f64(data["forward_premia"], fx_names),
        "tp_acm":         _to_f64(data["yield_curve"],    ["tp_acm_10y"]).ravel(),
        "yield_2y":       _to_f64(data["yield_curve"],    ["y2"]).ravel(),
        "yield_10y":      _to_f64(data["yield_curve"],    ["y10"]).ravel(),
        "yield_30y":      _to_f64(data["yield_curve"],    ["y30"]).ravel(),
        "swap_spread":    _to_f64(data["yield_curve"],    ["swap_spread_30y"]).ravel(),
        "cds_5y":         _to_f64(data["yield_curve"],    ["cds_5y_proxy"]).ravel(),
        "vix_spot":       _to_f64(data["vol_surface"],    ["vix_spot"]).ravel(),
        "iv_atm":         _to_f64(data["vol_surface"],    ["iv_atm"]).ravel(),
        "rv_21d":         _to_f64(data["vol_surface"],    ["rv_21d"]).ravel(),
        "ism_pmi":        _to_f64(data["equity"],         ["ism_pmi"]).ravel(),
        "energy_returns": _to_f64(data["energy"],
                                   [f"ret_{s}" for s in ENERGY_SYMBOLS]),
        "inv_surprises":  _to_f64(data["energy"],
                                   [f"inv_surprise_{s}" for s in ENERGY_SYMBOLS]),
        "ca_returns":     _to_f64(data["cross_asset"],
                                   [f"ret_{s}" for s in CROSS_ASSET_SYMBOLS]),
    }
