"""Point-in-time (PIT) raw-record filter with leakage HARD GATE (Worker 2).

Dict-based API — no SQLAlchemy / Worker 1 schema imports.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Callable, Optional

from src.data.quality_monitor import HARD_GATE_CODE, HardGate, default_hard_gate

logger = logging.getLogger(__name__)

# Re-export for tests / call sites that expect the constant here.
__all__ = ["HARD_GATE_CODE", "PITEngine"]


def _parse_observed_at(record: dict[str, Any]) -> Optional[datetime]:
    """Extract ``observed_at`` as datetime; None if missing/unparseable."""
    value = record.get("observed_at")
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        text = value.replace("Z", "+00:00")
        try:
            return datetime.fromisoformat(text)
        except ValueError:
            return None
    return None


def _align_for_compare(obs: datetime, as_of_time: datetime) -> datetime:
    """Align tz awareness so naive/aware mixes do not raise."""
    if as_of_time.tzinfo is not None and obs.tzinfo is None:
        return obs.replace(tzinfo=timezone.utc)
    if as_of_time.tzinfo is None and obs.tzinfo is not None:
        return obs.replace(tzinfo=None)
    return obs


class PITEngine:
    """Filter raw observation dicts to a point-in-time window.

    Integrity rule
    --------------
    Keep records with ``observed_at <= as_of_time``. If **any** record has
    ``observed_at > as_of_time``, integrity is False and a HARD GATE
    ``MODEL_TIER_X`` (NO BET) is fired via the local :class:`HardGate`
    (or an injected callback) — never via Worker 1 ``DataMonitor``.

    Records lacking a parseable ``observed_at`` are **excluded** from the
    valid set but do **not** count as future leakage.
    """

    def __init__(
        self,
        hard_gate: Optional[HardGate] = None,
        on_hard_gate: Optional[Callable[..., Any]] = None,
    ) -> None:
        self.hard_gate = hard_gate or default_hard_gate
        self.on_hard_gate = on_hard_gate

    def get_valid_raw_records(
        self,
        raw_records: list[dict[str, Any]],
        as_of_time: datetime,
    ) -> tuple[list[dict[str, Any]], bool]:
        """Return PIT-valid records and integrity flag.

        Parameters
        ----------
        raw_records:
            Observation dicts; each should include ``observed_at``
            (``datetime`` or ISO-8601 string).
        as_of_time:
            Inclusive upper bound — ``observed_at <= as_of_time`` is valid.
            Must be timezone-aware.

        Returns
        -------
        tuple[list[dict], bool]
            ``(valid_records, pit_integrity_passed)``.
            ``pit_integrity_passed`` is False iff any record leaked
            (``observed_at > as_of_time``).
        """
        if as_of_time.tzinfo is None:
            raise ValueError("as_of_time must be timezone-aware")

        valid: list[dict[str, Any]] = []
        leak_indices: list[int] = []

        for idx, record in enumerate(raw_records):
            obs = _parse_observed_at(record)
            if obs is None:
                continue
            obs = _align_for_compare(obs, as_of_time)
            if obs > as_of_time:
                leak_indices.append(idx)
            else:
                valid.append(record)

        pit_integrity_passed = len(leak_indices) == 0
        if not pit_integrity_passed:
            self._trigger_hard_gate(
                as_of_time=as_of_time,
                leak_count=len(leak_indices),
                leak_indices=leak_indices,
            )
            logger.critical(
                "PIT HARD GATE %s — %d leak record(s) after %s",
                HARD_GATE_CODE,
                len(leak_indices),
                as_of_time.isoformat(),
            )

        return valid, pit_integrity_passed

    def _trigger_hard_gate(
        self,
        *,
        as_of_time: datetime,
        leak_count: int,
        leak_indices: list[int],
    ) -> None:
        message = "PIT leakage detected: raw observations after as_of_time"
        ctx = {
            "as_of_time": as_of_time.isoformat(),
            "leak_count": leak_count,
            "leak_indices": leak_indices,
        }
        self.hard_gate.trigger(
            message, code=HARD_GATE_CODE, no_bet=True, **ctx
        )
        if self.on_hard_gate is not None:
            self.on_hard_gate(
                message, code=HARD_GATE_CODE, no_bet=True, **ctx
            )
