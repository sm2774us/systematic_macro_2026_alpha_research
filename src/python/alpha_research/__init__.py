"""alpha_research — Institutional systematic macro alpha research pipeline.

Six strategy signals implemented in C++26 with Python 3.13 research interface:
  - PDRRM: Policy Divergence × Real Rate Momentum (G10 FX Futures)
  - TPMCR: Term Premium Momentum & Curve Regime (Rates Futures)
  - MAERM: Macro-Adjusted Earnings Revision Momentum (Equity Futures)
  - ISRC:  Inventory Surprise × Roll Return Composite (Energy Futures)
  - VSRA:  Volatility Surface Regime Arbitrage (SPX Options + VIX)
  - FDSP:  Fiscal Dominance Shock Propagation (Cross-Asset)

Author: Alpha Research Pod — 2026
"""

from alpha_research.backtest import BLACK_SWAN_WINDOWS, BacktestOrchestrator, StrategyKPIRow
from alpha_research.data import (
    CROSS_ASSET_SYMBOLS,
    ENERGY_SYMBOLS,
    EQUITY_SYMBOLS,
    G10_FX_SYMBOLS,
    RATES_SYMBOLS,
    MarketDataConfig,
    extract_numpy_panels,
    generate_synthetic_data,
)
from alpha_research.signals import (
    MASTER_WEIGHTS,
    DecayMonitor,
    SignalBundle,
    compute_master_signal,
    compute_pdrrm_signals,
)

__all__ = [
    "MarketDataConfig",
    "generate_synthetic_data",
    "extract_numpy_panels",
    "G10_FX_SYMBOLS",
    "RATES_SYMBOLS",
    "ENERGY_SYMBOLS",
    "EQUITY_SYMBOLS",
    "CROSS_ASSET_SYMBOLS",
    "compute_pdrrm_signals",
    "compute_master_signal",
    "DecayMonitor",
    "MASTER_WEIGHTS",
    "SignalBundle",
    "BacktestOrchestrator",
    "StrategyKPIRow",
    "BLACK_SWAN_WINDOWS",
]
