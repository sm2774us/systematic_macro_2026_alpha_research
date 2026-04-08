"""scripts/runner.py — Bazel run dispatcher for alpha-research-2026.

Each ``py_binary`` target in the root BUILD.bazel passes a single positional
argument (``backtest`` | ``monitor`` | ``jupyter`` | ``docker_build`` |
``docker_jupyter`` | ``docker_down``) as ``args = ["<action>"]``.

Environment variables:
    DATA_DIR        Path to market data CSV directory. (default: data)
    N_DAYS          Synthetic data days.               (default: 2520)
    N_CURRENCIES    G10 currencies to include.         (default: 8)
    SLACK_WEBHOOK   Slack webhook URL for decay alerts.(default: "")
    LOG_LEVEL       Python logging level.              (default: INFO)

Usage (via Bazel):
    bazel run //:backtest
    bazel run //:monitor
    bazel run //:research
    bazel run //:docker_build
    bazel run //:docker_research
    bazel run //:docker_down

Author: Alpha Research Pod — 2026
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Final

# Ensure src/python is importable when invoked via Bazel or directly
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src" / "python"))

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("runner")


def _env(key: str, default: str) -> str:
    return os.environ.get(key, default)


_DOCKER_COMMANDS: Final[dict[str, list[str]]] = {
    "docker_build":   ["docker", "compose", "build"],
    "docker_jupyter": ["docker", "compose", "up", "jupyter"],
    "docker_down":    ["docker", "compose", "down"],
}


def _run_backtest() -> int:
    """Run full alpha research backtest pipeline."""
    from alpha_research.backtest import BacktestOrchestrator
    from alpha_research.data import MarketDataConfig

    cfg = MarketDataConfig(
        data_dir     = Path(_env("DATA_DIR", "data")),
        use_synthetic= _env("DATA_DIR", "") == "",
        n_days       = int(_env("N_DAYS", "2520")),
    )
    orc = BacktestOrchestrator(
        config       = cfg,
        n_currencies = int(_env("N_CURRENCIES", "8")),
    )
    report = orc.run()
    logger.info("\n%s", report)
    return 0


def _run_monitor() -> int:
    """Run signal decay monitor over latest signals."""
    from alpha_research.data import MarketDataConfig, generate_synthetic_data, extract_numpy_panels
    from alpha_research.signals import compute_master_signal, DecayMonitor
    import polars as pl

    n_days = int(_env("N_DAYS", "2520"))
    raw = generate_synthetic_data(n_days=n_days, seed=99)
    panels = extract_numpy_panels(raw)
    from alpha_research.data import _make_trading_days
    dates = _make_trading_days(n_days)

    bundles = compute_master_signal(panels, dates)
    webhook = _env("SLACK_WEBHOOK", "")

    for name, bundle in bundles.items():
        dm = DecayMonitor(name, slack_webhook=webhook or None)
        # Feed last 60 days of IC estimates (approx from signal autocorrelation)
        import numpy as np
        sig = bundle.signals
        ret = np.zeros_like(sig)  # placeholder; production uses actual returns
        # Rolling 1-day IC from last 60 days
        for t in range(max(0, len(sig) - 60), len(sig) - 1):
            s = sig[t]
            r = sig[t + 1]   # surrogate: use next signal as proxy
            sig_std = s.std() + 1e-8
            ret_std = r.std() + 1e-8
            ic = float(np.corrcoef(s, r)[0, 1]) if len(s) > 1 else 0.0
            alert = dm.update(ic)
            if alert:
                logger.warning("DECAY ALERT — %s  HL=%.1fd", name, dm.half_life())

    logger.info("Monitor run complete.")
    return 0


def _dispatch(action: str) -> int:
    if action == "backtest":
        return _run_backtest()

    if action == "monitor":
        return _run_monitor()

    if action == "jupyter":
        nb_dir = _REPO_ROOT / "notebooks"
        return subprocess.call(
            ["jupyter", "lab", "--notebook-dir", str(nb_dir),
             "--ip=0.0.0.0", "--no-browser"],
        )

    if action in _DOCKER_COMMANDS:
        return subprocess.call(_DOCKER_COMMANDS[action],
                               cwd=str(_REPO_ROOT))

    logger.error("Unknown action '%s'. Valid: backtest monitor jupyter "
                 "docker_build docker_jupyter docker_down", action)
    return 1


def main() -> int:
    if len(sys.argv) < 2:
        logger.error("No action provided.")
        return 1
    return _dispatch(sys.argv[1])


if __name__ == "__main__":
    sys.exit(main())
