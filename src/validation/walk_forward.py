"""Expanding-window walk-forward time splits with no temporal leakage."""

from __future__ import annotations

import calendar
from dataclasses import dataclass
from datetime import datetime, timezone


@dataclass(frozen=True, slots=True)
class TimeSplit:
    """One train/test window using half-open UTC intervals ``[start, end)``.

    Attributes
    ----------
    train_start, train_end :
        Training window; ``train_end`` is exclusive.
    test_start, test_end :
        Test window; always satisfies ``train_end <= test_start < test_end``.
    """

    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime

    def __post_init__(self) -> None:
        if self.train_start.tzinfo is None or self.train_end.tzinfo is None:
            raise ValueError("train bounds must be timezone-aware")
        if self.test_start.tzinfo is None or self.test_end.tzinfo is None:
            raise ValueError("test bounds must be timezone-aware")
        if not (self.train_start < self.train_end):
            raise ValueError("train_start must be strictly before train_end")
        if not (self.test_start < self.test_end):
            raise ValueError("test_start must be strictly before test_end")
        if self.train_end > self.test_start:
            raise ValueError("temporal leakage: train_end must be <= test_start")


class WalkForwardValidator:
    """Generate expanding walk-forward calendar splits.

    Example (``start_year=2019``, ``end_year=2023``, ``train_window_years=3``,
    ``test_window_months=6``)::

        Train [2019-01-01, 2023-01-01) → Test [2023-01-01, 2023-07-01)  # 2023-H1
        Train [2019-01-01, 2023-07-01) → Test [2023-07-01, 2024-01-01)  # 2023-H2
    """

    @staticmethod
    def generate_time_splits(
        start_year: int,
        end_year: int,
        train_window_years: int = 3,
        test_window_months: int = 6,
    ) -> list[TimeSplit]:
        """Build expanding train / fixed-length test splits.

        Initial training covers calendar years
        ``start_year`` … ``start_year + train_window_years`` inclusive
        (half-open end at Jan 1 of the following year). Each subsequent split
        expands the train end to the previous test end. Test windows advance by
        ``test_window_months`` until the exclusive horizon ``end_year + 1``.

        Parameters
        ----------
        start_year :
            First calendar year included in training.
        end_year :
            Last calendar year that may appear in a test window.
        train_window_years :
            Span from ``start_year`` to the last fully-trained year
            (e.g. 3 → train through end of ``start_year + 3``).
        test_window_months :
            Length of each out-of-sample block (default 6 ⇒ half-years).

        Returns
        -------
        list[TimeSplit]
            Chronological splits; empty if the horizon cannot hold a test fold.
        """
        if train_window_years < 1:
            raise ValueError("train_window_years must be >= 1")
        if test_window_months < 1:
            raise ValueError("test_window_months must be >= 1")
        if end_year < start_year:
            raise ValueError("end_year must be >= start_year")

        tz = timezone.utc
        train_start = datetime(start_year, 1, 1, tzinfo=tz)
        # Inclusive train years: start .. start+window → exclusive end Jan 1 next
        first_test_start = datetime(
            start_year + train_window_years + 1, 1, 1, tzinfo=tz
        )
        horizon_end = datetime(end_year + 1, 1, 1, tzinfo=tz)

        splits: list[TimeSplit] = []
        test_start = first_test_start
        while test_start < horizon_end:
            test_end = _add_months(test_start, test_window_months)
            if test_end > horizon_end:
                test_end = horizon_end
            if test_end <= test_start:
                break

            train_end = test_start  # expanding: train up to (not into) test
            if train_end <= train_start:
                break

            splits.append(
                TimeSplit(
                    train_start=train_start,
                    train_end=train_end,
                    test_start=test_start,
                    test_end=test_end,
                )
            )
            test_start = test_end

        return splits


def _add_months(dt: datetime, months: int) -> datetime:
    """Add calendar months, clamping the day to the target month length."""
    year = dt.year + (dt.month - 1 + months) // 12
    month = (dt.month - 1 + months) % 12 + 1
    day = min(dt.day, calendar.monthrange(year, month)[1])
    return dt.replace(year=year, month=month, day=day)
