"""alpha_research.signals — Python signal orchestration layer.

This module provides the research-friendly Python interface for computing
all six alpha signals. The hot-path computation is delegated to C++26
engines via nanobind (`alpha_cpp`). Python handles:

  1. Data routing to the correct C++ engine
  2. Signal combination and weight estimation
  3. KPI computation delegation
  4. Signal decay monitoring

Polars LazyFrames are used for efficient signal history storage.

Author: Alpha Research Pod — 2026
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, NamedTuple

import numpy as np
import polars as pl

# C++26 engine bridge (nanobind extension)
try:
    import alpha_cpp  # type: ignore[import]
    _CPP_AVAILABLE = True
except ImportError:
    _CPP_AVAILABLE = False
    logging.getLogger(__name__).warning(
        "alpha_cpp extension not available — falling back to pure Python."
    )

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Signal combination weights (master risk architecture)
# ─────────────────────────────────────────────────────────────────────────────

#: Signal contribution weights in the master portfolio (PCA-adjusted).
#: PDRRM is the primary signal (40% budget); FDSP activates only during crises.
MASTER_WEIGHTS: dict[str, float] = {
    "PDRRM": 0.30,  # Policy Divergence × Real Rate Momentum
    "TPMCR": 0.20,  # Term Premium Momentum & Curve Regime
    "MAERM": 0.20,  # Macro-Adjusted Earnings Revision Momentum
    "ISRC":  0.10,  # Inventory Surprise × Roll Return
    "VSRA":  0.10,  # Volatility Surface Regime Arbitrage
    "FDSP":  0.10,  # Fiscal Dominance Shock Propagation
}


# ─────────────────────────────────────────────────────────────────────────────
# Signal output bundle
# ─────────────────────────────────────────────────────────────────────────────


class SignalBundle(NamedTuple):
    """All computed signals and positions for one strategy or the combined pod.

    Attributes:
        signals:   [T × N] numpy array of z-scored signals.
        positions: [T × N] numpy array of vol-targeted positions.
        dates:     Polars Date series aligned with T.
        labels:    Instrument name labels (length N).
        kpis:      KPIBundle from C++ backtest engine (None if not run).
    """

    signals:   np.ndarray
    positions: np.ndarray
    dates:     pl.Series
    labels:    list[str]
    kpis:      Any | None = None  # alpha_cpp.KPIBundle


# ─────────────────────────────────────────────────────────────────────────────
# Pure-Python fallback implementations
# ─────────────────────────────────────────────────────────────────────────────


def _zscore_panel(arr: np.ndarray) -> np.ndarray:
    """Cross-sectional z-score each row of a [T × N] array."""
    mu  = arr.mean(axis=1, keepdims=True)
    sig = arr.std(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        z = np.where(sig > 1e-8, (arr - mu) / sig, 0.0)
    return np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)


def _rolling_vol(returns: np.ndarray, window: int = 21) -> np.ndarray:
    """Rolling annualised volatility [T × N]. Warmup rows return 0.05 (5% floor)."""
    T, N = returns.shape
    rv = np.full((T, N), 0.05, dtype=np.float64)
    for t in range(window, T):
        seg = returns[t - window:t]
        rv[t] = np.sqrt(252.0) * np.clip(seg.std(axis=0), 1e-8, None)
    return rv


def _compute_rrdm_py(
    nominal_rates: np.ndarray,
    breakevens: np.ndarray,
    window: int = 20,
) -> np.ndarray:
    """Pure-Python RRDM computation fallback."""
    real_rates = nominal_rates - breakevens
    diff = real_rates - real_rates[:, [0]]  # vs USD (col 0)
    T, N = diff.shape
    rrdm = np.zeros_like(diff)
    if T > window:
        rrdm[window:] = diff[window:] - diff[:-window]
    return _zscore_panel(rrdm)


def _compute_pss_py(
    cb_surprises: np.ndarray,
    halflife: float = 10.0,
) -> np.ndarray:
    """Pure-Python PSS computation fallback."""
    alpha = 1.0 - np.exp(-np.log(2.0) / halflife)
    T, N = cb_surprises.shape
    pss = np.zeros_like(cb_surprises)
    ewma = np.zeros(N)
    for t in range(T):
        ewma = (1 - alpha) * ewma + alpha * np.clip(cb_surprises[t], -3, 3)
        pss[t] = ewma
    return _zscore_panel(pss)


def _compute_rac_py(
    forward_premia: np.ndarray,
    returns: np.ndarray,
    vol_window: int = 21,
) -> np.ndarray:
    """Pure-Python RAC computation fallback. Warmup rows are explicitly zeroed."""
    rv = _rolling_vol(returns, vol_window)
    rac = np.where(rv > 1e-8, forward_premia / rv, 0.0)
    rac[:vol_window] = 0.0          # explicit warmup zero — no signal before window
    return _zscore_panel(rac)


# ─────────────────────────────────────────────────────────────────────────────
# PDRRM signal computation
# ─────────────────────────────────────────────────────────────────────────────


def compute_pdrrm_signals(
    nominal_rates: np.ndarray,
    breakevens: np.ndarray,
    cb_surprises: np.ndarray,
    forward_premia: np.ndarray,
    fx_returns: np.ndarray,
    ridge_lambda: float = 0.10,
    train_fraction: float = 0.60,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute PDRRM combined signal panel and vol-targeted positions.

    Delegates hot-path to C++26 PDRRMEngine if available.

    Args:
        nominal_rates:   [T × N] nominal 2Y rates.
        breakevens:      [T × N] 2Y breakeven inflation.
        cb_surprises:    [T × N] CB meeting surprises (bp, 0 on non-meeting).
        forward_premia:  [T × N] annualised forward premia.
        fx_returns:      [T × N] daily log-returns.
        ridge_lambda:    Ridge L2 regularisation strength.
        train_fraction:  Fraction of data used for in-sample weight estimation.

    Returns:
        Tuple of (signals [T × N], positions [T × N]).
    """
    T, N = fx_returns.shape
    T_is = int(T * train_fraction)

    if _CPP_AVAILABLE:
        cfg = alpha_cpp.PDRRMConfig()
        cfg.ridge_lambda = ridge_lambda
        engine = alpha_cpp.PDRRMEngine(cfg)

        rrdm = engine.compute_rrdm(
            nominal_rates.astype(np.float64),
            breakevens.astype(np.float64),
        )
        pss = engine.compute_pss(cb_surprises.astype(np.float64))
        rac = engine.compute_rac(
            forward_premia.astype(np.float64),
            fx_returns.astype(np.float64),
        )

        # Ridge weight estimation on in-sample panel
        # Stack all instruments: [T_IS * N, 3]
        X_is = np.column_stack([
            rrdm[:T_is].reshape(-1),
            pss[:T_is].reshape(-1),
            rac[:T_is].reshape(-1),
        ])
        # Forward 21-day return as target
        horizon = 21
        y_is = np.array([
            fx_returns[t:t + horizon, i].sum()
            for t in range(T_is - horizon)
            for i in range(N)
        ], dtype=np.float64)
        X_is_trim = X_is[: len(y_is)]

        weights = engine.update_weights(X_is_trim, y_is)
        logger.info(
            "PDRRM weights — α_RRDM=%.4f, α_PSS=%.4f, α_RAC=%.4f",
            weights[0], weights[1], weights[2],
        )

        # Combined signal
        signals = (weights[0] * rrdm
                 + weights[1] * pss
                 + weights[2] * rac)

    else:
        # Pure-Python fallback
        rrdm     = _compute_rrdm_py(nominal_rates, breakevens)
        pss      = _compute_pss_py(cb_surprises)
        rac      = _compute_rac_py(forward_premia, fx_returns)
        signals  = (rrdm + pss + rac) / 3.0

    # Vol-targeted positions
    rv = _rolling_vol(fx_returns)
    positions = np.where(rv > 1e-8, signals * 0.10 / rv, 0.0)
    positions = np.clip(positions, -0.25, 0.25)

    return signals, positions


# ─────────────────────────────────────────────────────────────────────────────
# Combined master portfolio signal
# ─────────────────────────────────────────────────────────────────────────────


def compute_master_signal(
    data_panels: dict[str, np.ndarray],
    dates: pl.Series,
) -> dict[str, SignalBundle]:
    """Compute all six strategy signals and combine into a master portfolio.

    Args:
        data_panels: Dictionary of numpy arrays from `extract_numpy_panels`.
        dates:       Polars Date series aligned with T axis.

    Returns:
        Dictionary mapping strategy name → SignalBundle, plus 'MASTER' bundle.
    """
    from alpha_research.data import G10_FX_SYMBOLS, ENERGY_SYMBOLS, EQUITY_SYMBOLS, CROSS_ASSET_SYMBOLS

    bundles: dict[str, SignalBundle] = {}
    fx_labels = list(G10_FX_SYMBOLS)[:data_panels["fx_returns"].shape[1]]

    # ── PDRRM ─────────────────────────────────────────────────────────────────
    logger.info("Computing PDRRM signals...")
    pdrrm_sigs, pdrrm_pos = compute_pdrrm_signals(
        nominal_rates  = data_panels["nominal_rates"],
        breakevens     = data_panels["breakevens"],
        cb_surprises   = data_panels["cb_surprises"],
        forward_premia = data_panels["forward_premia"],
        fx_returns     = data_panels["fx_returns"],
    )
    bundles["PDRRM"] = SignalBundle(
        signals=pdrrm_sigs, positions=pdrrm_pos,
        dates=dates, labels=fx_labels,
    )

    # ── TPMCR (simplified Python path; C++ engine handles tick-by-tick) ──────
    logger.info("Computing TPMCR signals...")
    n_days = len(data_panels["tp_acm"])
    tpmcr_sigs = np.zeros((n_days, 5))
    tau = 15
    tp = data_panels["tp_acm"]
    tpm_raw = np.concatenate([np.zeros(tau), tp[tau:] - tp[:-tau]])
    tpm_mu, tpm_sig = tpm_raw.mean(), tpm_raw.std() + 1e-8
    tpm_z = (tpm_raw - tpm_mu) / tpm_sig
    for i in range(5):
        tpmcr_sigs[:, i] = tpm_z * (1.5 if i in (1, 4) else 1.0)  # ZB/TN weighting
    rv_rates = _rolling_vol(
        np.stack([data_panels["yield_10y"]] * 5, axis=1) * 0.01)
    tpmcr_pos = np.clip(tpmcr_sigs * 0.10 / (rv_rates + 1e-8), -0.30, 0.30)
    bundles["TPMCR"] = SignalBundle(
        signals=tpmcr_sigs, positions=tpmcr_pos,
        dates=dates, labels=list(("ZN ZB RX G TN").split()),
    )

    # ── MAERM ────────────────────────────────────────────────────────────────
    logger.info("Computing MAERM signals...")
    T = n_days
    # MAERM uses equity index returns; fall back to zeros if not in panels
    equity_ret = np.zeros((T, 4), dtype=np.float64)
    ism = data_panels["ism_pmi"]
    ism_mu, ism_sig = ism.mean(), ism.std() + 1e-8
    ism_z = (ism - ism_mu) / ism_sig
    maerm_sigs = np.zeros((T, 4))
    for i in range(4):
        maerm_sigs[:, i] = 0.4 * ism_z + 0.6 * np.sign(ism_z)
    maerm_rv = _rolling_vol(equity_ret)
    maerm_pos = np.clip(maerm_sigs * 0.10 / (maerm_rv + 1e-8), -0.25, 0.25)
    bundles["MAERM"] = SignalBundle(
        signals=maerm_sigs, positions=maerm_pos,
        dates=dates, labels=list(("ES NQ RTY SX5E").split()),
    )

    # ── ISRC ─────────────────────────────────────────────────────────────────
    logger.info("Computing ISRC signals...")
    inv_surp = data_panels["inv_surprises"]
    en_ret   = data_panels["energy_returns"]
    isrc_sigs = _zscore_panel(inv_surp)
    isrc_rv   = _rolling_vol(en_ret)
    isrc_pos  = np.clip(isrc_sigs * 0.10 / (isrc_rv + 1e-8), -0.30, 0.30)
    bundles["ISRC"] = SignalBundle(
        signals=isrc_sigs, positions=isrc_pos,
        dates=dates, labels=list(ENERGY_SYMBOLS),
    )

    # ── VSRA ─────────────────────────────────────────────────────────────────
    logger.info("Computing VSRA signals...")
    vrp = data_panels["iv_atm"] - data_panels["rv_21d"]
    vrp_mu, vrp_sig = vrp.mean(), vrp.std() + 1e-8
    vrp_z = (vrp - vrp_mu) / vrp_sig
    vx_ret = np.diff(data_panels["vix_spot"], prepend=data_panels["vix_spot"][0])
    vx_ret /= data_panels["vix_spot"] + 1e-8
    vsra_sigs = vrp_z.reshape(-1, 1)
    vsra_rv   = _rolling_vol(vx_ret.reshape(-1, 1))
    vsra_pos  = np.clip(vsra_sigs * 0.08 / (vsra_rv + 1e-8), -0.20, 0.20)
    bundles["VSRA"] = SignalBundle(
        signals=vsra_sigs, positions=vsra_pos,
        dates=dates, labels=["VX"],
    )

    # ── FDSP ─────────────────────────────────────────────────────────────────
    logger.info("Computing FDSP signals...")
    fci_raw  = (-data_panels["swap_spread"] * 1000.0
                + data_panels["cds_5y"] * 10000.0)
    fci_mu, fci_sig = fci_raw.mean(), fci_raw.std() + 1e-8
    fci_z = (fci_raw - fci_mu) / fci_sig
    ca_ret = data_panels["ca_returns"]
    fdsp_sigs = np.outer(fci_z, np.array([0.30, 0.40, 0.20, 0.35, 0.55]))
    fdsp_rv   = _rolling_vol(ca_ret)
    fdsp_pos  = np.clip(fdsp_sigs * 0.10 / (fdsp_rv + 1e-8), -0.20, 0.20)
    bundles["FDSP"] = SignalBundle(
        signals=fdsp_sigs, positions=fdsp_pos,
        dates=dates, labels=list(CROSS_ASSET_SYMBOLS),
    )

    logger.info("All signals computed. Computing KPIs via C++ backtest engine...")

    # ── KPI computation via C++ BacktestEngine ────────────────────────────────
    if _CPP_AVAILABLE:
        bt = alpha_cpp.BacktestEngine()
        for name, bundle in bundles.items():
            sigs = bundle.signals.astype(np.float64)
            rets = np.zeros_like(sigs)
            # Align returns to signal shape
            if name == "PDRRM":
                rets = data_panels["fx_returns"][:, :sigs.shape[1]].astype(np.float64)
            elif name == "TPMCR":
                rets = np.tile(data_panels["yield_10y"].reshape(-1, 1) * -0.01,
                               (1, sigs.shape[1]))
            elif name == "MAERM":
                rets = data_panels.get("energy_returns", np.zeros_like(sigs))[:T, :4]
            elif name == "ISRC":
                rets = data_panels["energy_returns"].astype(np.float64)
            elif name == "VSRA":
                rets = vx_ret.reshape(-1, 1).astype(np.float64)
            elif name == "FDSP":
                rets = data_panels["ca_returns"].astype(np.float64)

            try:
                kpis = bt.run(sigs, rets.astype(np.float64), 0.0001, 252.0)
                bundles[name] = SignalBundle(
                    signals=bundle.signals, positions=bundle.positions,
                    dates=bundle.dates, labels=bundle.labels, kpis=kpis,
                )
                logger.info(
                    "%s KPIs — Sharpe: %.3f, IC: %.4f, MDD: %.2f%%",
                    name, kpis.sharpe_ratio, kpis.mean_ic, kpis.max_drawdown * 100,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("KPI computation failed for %s: %s", name, exc)

    return bundles


# ─────────────────────────────────────────────────────────────────────────────
# Signal decay monitor wrapper
# ─────────────────────────────────────────────────────────────────────────────


class DecayMonitor:
    """Python wrapper around alpha_cpp.SignalDecayMonitor with Slack alerting.

    In production, sends a Slack webhook notification when signal decay
    is detected. In CI, logs a warning and fails the check.

    Args:
        strategy_name: Strategy identifier for alert messages.
        window:        Rolling window for IC decay estimation.
        slack_webhook: Optional Slack webhook URL for alerts.
    """

    def __init__(
        self,
        strategy_name: str,
        window: int = 60,
        slack_webhook: str | None = None,
    ) -> None:
        self.name = strategy_name
        self.webhook = slack_webhook
        self._monitor = (
            alpha_cpp.SignalDecayMonitor(window) if _CPP_AVAILABLE else None
        )
        self._ic_history: list[float] = []

    def update(self, ic: float) -> bool:
        """Update with new IC and return True if decay alert triggered.

        Args:
            ic: Information coefficient for latest period.

        Returns:
            True if signal decay alert should be raised.
        """
        self._ic_history.append(ic)
        if self._monitor is not None:
            alert = self._monitor.update(ic)
        else:
            # Fallback: simple 30-day mean IC check
            recent = self._ic_history[-30:]
            alert = len(recent) >= 30 and np.mean(recent) < 0.02

        if alert:
            msg = (
                f"⚠️ SIGNAL DECAY ALERT: {self.name} "
                f"— rolling IC={np.mean(self._ic_history[-30:]):.4f}, "
                f"HL={self.half_life():.1f}d"
            )
            logger.warning(msg)
            if self.webhook:
                self._send_slack(msg)

        return alert

    def half_life(self) -> float:
        """Return estimated IC half-life in trading days."""
        if self._monitor is not None:
            return self._monitor.half_life()
        if len(self._ic_history) < 10:
            return float("inf")
        if _CPP_AVAILABLE:
            return alpha_cpp.half_life_decay(np.array(self._ic_history))
        return float("inf")

    def _send_slack(self, message: str) -> None:
        """Post a Slack webhook notification (non-blocking best-effort)."""
        import urllib.request, json  # noqa: E401
        try:
            req = urllib.request.Request(
                self.webhook,
                data=json.dumps({"text": message}).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            urllib.request.urlopen(req, timeout=5)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Slack notification failed: %s", exc)
