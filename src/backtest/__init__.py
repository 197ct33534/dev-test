"""Walk-forward backtesting master (quant replay + report)."""

from __future__ import annotations

from typing import Any

__all__ = [
    "BacktestConfig",
    "BacktestMatch",
    "QuantReport",
    "WalkForwardBacktester",
    "build_quant_report",
    "generate_synthetic_matches",
    "render_quant_report",
    "run_backtest",
]


def __getattr__(name: str) -> Any:
    """Lazy exports so ``python -m src.backtest.run_backtest`` stays clean."""
    if name in {
        "BacktestConfig",
        "BacktestMatch",
        "WalkForwardBacktester",
        "generate_synthetic_matches",
        "run_backtest",
    }:
        from src.backtest import run_backtest as _rb

        return getattr(_rb, name)
    if name in {"QuantReport", "build_quant_report", "render_quant_report"}:
        from src.backtest import report as _rep

        return getattr(_rep, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
