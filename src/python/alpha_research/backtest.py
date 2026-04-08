"""alpha_research.backtest — Backtest orchestrator with KPI and stress analysis.

Orchestrates the full research pipeline:
  1. Load / generate market data (Polars)
  2. Compute all six strategy signals (C++26 engines via nanobind)
  3. Run full backtest with KPI computation (C++26 BacktestEngine)
  4. Stress-test against historical black-swan events
  5. Produce structured KPI report (Polars DataFrame)

Typical usage:
    from alpha_research.backtest import BacktestOrchestrator
    orc = BacktestOrchestrator()
    report = orc.run()
    orc.print_report(report)

Author: Alpha Research Pod — 2026
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl

from alpha_research.data import MarketDataConfig, generate_synthetic_data, extract_numpy_panels
from alpha_research.signals import compute_master_signal, MASTER_WEIGHTS, SignalBundle

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Black-swan event windows (historical)
# ─────────────────────────────────────────────────────────────────────────────

BLACK_SWAN_WINDOWS: dict[str, tuple[int, int]] = {
    "COVID-19 Crash (Mar 2020)":       (1300, 1350),
    "2022 Rate Shock":                  (1800, 1900),
    "Aug 2024 JPY Carry Unwind":        (2300, 2340),
    "2023 US Banking Stress":           (2050, 2090),
}


# ─────────────────────────────────────────────────────────────────────────────
# KPI report row
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(slots=True)
class StrategyKPIRow:
    """Single-row KPI summary for one strategy."""

    strategy:         str
    sharpe:           float
    sortino:          float
    calmar:           float
    cagr_pct:         float
    mdd_pct:          float
    annual_vol_pct:   float
    mean_ic:          float
    icir:             float
    hit_rate_pct:     float
    tc_drag_bps:      float
    net_alpha_bps:    float
    half_life_days:   float
    decay_alert:      bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy":       self.strategy,
            "sharpe":         round(self.sharpe, 3),
            "sortino":        round(self.sortino, 3),
            "calmar":         round(self.calmar, 3),
            "cagr_%":         round(self.cagr_pct, 2),
            "mdd_%":          round(self.mdd_pct, 2),
            "ann_vol_%":      round(self.annual_vol_pct, 2),
            "mean_ic":        round(self.mean_ic, 4),
            "icir":           round(self.icir, 3),
            "hit_rate_%":     round(self.hit_rate_pct, 2),
            "tc_drag_bps":    round(self.tc_drag_bps, 1),
            "net_alpha_bps":  round(self.net_alpha_bps, 1),
            "hl_days":        round(self.half_life_days, 1),
            "decay_alert":    self.decay_alert,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Backtest Orchestrator
# ─────────────────────────────────────────────────────────────────────────────


class BacktestOrchestrator:
    """Full-pipeline backtest orchestrator.

    Args:
        config:    Market data configuration.
        n_days:    Override for number of synthetic days.
        n_currencies: G10 currencies to include.
    """

    def __init__(
        self,
        config: MarketDataConfig | None = None,
        n_days: int = 2520,
        n_currencies: int = 8,
    ) -> None:
        self.config       = config or MarketDataConfig(use_synthetic=True, n_days=n_days)
        self.n_currencies = n_currencies
        self._bundles: dict[str, SignalBundle] | None = None

    def run(self) -> pl.DataFrame:
        """Run the full backtest pipeline and return a KPI report DataFrame.

        Returns:
            Polars DataFrame with one row per strategy + 'MASTER' row.
        """
        logger.info("=== Alpha Research Pipeline — Full Backtest ===")

        # ── 1. Data loading ───────────────────────────────────────────────────
        logger.info("Loading market data (synthetic=%s)...", self.config.use_synthetic)
        if self.config.use_synthetic:
            raw_data = generate_synthetic_data(
                n_days=self.config.n_days, n_currencies=self.n_currencies)
        else:
            from alpha_research.data import load_data_from_csv
            raw_data = load_data_from_csv(self.config)

        panels = extract_numpy_panels(raw_data, self.n_currencies)
        T = panels["fx_returns"].shape[0]

        # Reconstruct date series
        dates = pl.date_range(
            start=pl.date(2015, 1, 2),
            end=None, interval="1bd", eager=True,
        ).head(T)

        # ── 2. Signal computation ─────────────────────────────────────────────
        self._bundles = compute_master_signal(panels, dates)

        # ── 3. Stress-test analysis ───────────────────────────────────────────
        self._run_stress_tests(panels)

        # ── 4. Assemble KPI report ────────────────────────────────────────────
        rows: list[dict] = []
        for name, bundle in self._bundles.items():
            if bundle.kpis is not None:
                k = bundle.kpis
                rows.append(StrategyKPIRow(
                    strategy       = name,
                    sharpe         = k.sharpe_ratio,
                    sortino        = k.sortino_ratio,
                    calmar         = k.calmar_ratio,
                    cagr_pct       = k.cagr * 100,
                    mdd_pct        = k.max_drawdown * 100,
                    annual_vol_pct = k.annual_vol * 100,
                    mean_ic        = k.mean_ic,
                    icir           = k.icir,
                    hit_rate_pct   = k.hit_rate * 100,
                    tc_drag_bps    = k.tc_drag_bps,
                    net_alpha_bps  = k.net_alpha_bps,
                    half_life_days = k.ic_half_life_days,
                    decay_alert    = k.decay_alert,
                ).to_dict())
            else:
                rows.append({"strategy": name, "sharpe": 0.0,
                              "mean_ic": 0.0, "decay_alert": False})

        report = pl.DataFrame(rows)
        self._print_kpi_table(report)
        return report

    def _run_stress_tests(self, panels: dict[str, np.ndarray]) -> None:
        """Compute strategy PnL during historical black-swan windows."""
        logger.info("\n📉 Black-Swan Stress Test Results:")
        logger.info("%-35s | %8s | %8s | %8s", "Event", "PDRRM", "TPMCR", "FDSP")
        logger.info("-" * 70)

        if self._bundles is None:
            return

        for event_name, (t_start, t_end) in BLACK_SWAN_WINDOWS.items():
            row_vals: list[str] = []
            for strat in ("PDRRM", "TPMCR", "FDSP"):
                bundle = self._bundles.get(strat)
                if bundle is None:
                    row_vals.append("  N/A  ")
                    continue
                pos = bundle.positions
                # Use correct returns
                if strat == "PDRRM":
                    rets = panels["fx_returns"][:, :pos.shape[1]]
                elif strat == "TPMCR":
                    rets = np.tile(panels["yield_10y"].reshape(-1, 1) * -0.01,
                                   (1, pos.shape[1]))
                else:
                    rets = panels["ca_returns"][:, :pos.shape[1]]

                end = min(t_end, len(pos) - 1)
                start = min(t_start, end - 1)
                stress_pnl = (pos[start:end] * rets[start:end]).sum()
                row_vals.append(f"{stress_pnl:+8.3f}")

            logger.info("%-35s | %s | %s | %s",
                        event_name[:34], *row_vals)

    @staticmethod
    def _print_kpi_table(report: pl.DataFrame) -> None:
        """Log the full KPI report to console."""
        logger.info("\n📊 Full KPI Report:")
        logger.info(report.to_pandas().to_string() if hasattr(report, 'to_pandas')
                    else str(report))


def _cli_entry() -> None:
    """Entry point for `alpha-backtest` console script (pyproject.toml)."""
    import logging
    logging.basicConfig(level="INFO",
                        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s")
    orc = BacktestOrchestrator()
    report = orc.run()
    print(report)
