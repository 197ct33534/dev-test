"""Paper trading + auto-settlement (Worker 7)."""

from __future__ import annotations

from src.execution.paper_trader import PaperTrader, build_1x2_candidates
from src.execution.settlement_v2 import (
    SettlementEngineV2,
    compute_clv_value,
    map_settle_status,
    settle_market,
)

__all__ = [
    "PaperTrader",
    "SettlementEngineV2",
    "build_1x2_candidates",
    "compute_clv_value",
    "map_settle_status",
    "settle_market",
]
