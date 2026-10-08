"""Thin observability stubs for Quant Engine gates and data-quality alerts."""

from __future__ import annotations

import logging
from collections import deque
from typing import Any, Optional

logger = logging.getLogger("src.monitoring.data_monitor")

# Records kept per buffer; older entries are dropped (long-lived workers).
DEFAULT_MAX_RECORDS = 1000


class _RingBuffer(deque):
    """Bounded ``deque`` that still compares equal to a plain ``list``."""

    def __eq__(self, other: object) -> bool:
        if isinstance(other, list):
            return list(self) == other
        return super().__eq__(other)

    def __ne__(self, other: object) -> bool:
        return not self.__eq__(other)

    __hash__ = None  # type: ignore[assignment]


class DataMonitor:
    """Process-local monitor for warnings, hard gates, and NO-BET alerts.

    Replace / extend with metrics sinks later; methods are intentionally thin.
    """

    def __init__(
        self,
        *,
        logger_: Optional[logging.Logger] = None,
        max_records: int = DEFAULT_MAX_RECORDS,
    ) -> None:
        self._logger = logger_ or logger
        cap = max(1, int(max_records))
        # Bounded: ``default_monitor`` lives for the whole worker process.
        self.warnings: deque[dict[str, Any]] = _RingBuffer(maxlen=cap)
        self.alerts: deque[dict[str, Any]] = _RingBuffer(maxlen=cap)
        self.hard_gates: deque[dict[str, Any]] = _RingBuffer(maxlen=cap)

    def warn(self, message: str, *, code: Optional[str] = None, **context: Any) -> None:
        """Log a non-fatal data-quality warning (e.g. UNMAPPED entity)."""
        record = {"level": "WARN", "message": message, "code": code, **context}
        self.warnings.append(record)
        self._logger.warning("[%s] %s | %s", code or "WARN", message, context)

    def alert(self, message: str, *, code: Optional[str] = None, **context: Any) -> None:
        """Log an operational alert (elevated vs warn)."""
        record = {"level": "ALERT", "message": message, "code": code, **context}
        self.alerts.append(record)
        self._logger.error("[%s] %s | %s", code or "ALERT", message, context)

    def hard_gate(
        self,
        message: str,
        *,
        code: str = "MODEL_TIER_X",
        no_bet: bool = True,
        **context: Any,
    ) -> None:
        """Fire a HARD GATE — blocks betting (NO BET) for the affected path."""
        record = {
            "level": "HARD_GATE",
            "message": message,
            "code": code,
            "no_bet": no_bet,
            **context,
        }
        self.hard_gates.append(record)
        self.alerts.append(record)
        self._logger.critical(
            "[HARD_GATE %s] NO_BET=%s %s | %s", code, no_bet, message, context
        )


# Module-level default for convenience call sites.
default_monitor = DataMonitor()
