"""Regression tests: missing stats → NaN and point-in-time imputation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features import engineer_rolling_features, impute_rolling_with_league_mean


def _row(day: str, home: str, away: str, **stats: float) -> dict:
    base = {"Date": day, "HomeTeam": home, "AwayTeam": away}
    base.update(stats)
    return base


def test_missing_corners_not_stored_as_zero() -> None:
    rows = [
        # A has corners=10 here.
        _row("2024-01-01", "A", "B", HS=10, HST=4, AS=8, AST=3, HC=10, AC=4),
        # Corners missing, shots present → must NOT enter history as 0 corners.
        _row("2024-01-08", "A", "C", HS=12, HST=5, AS=6, AST=2, HC=np.nan, AC=np.nan),
        # Feature row for A after two matches.
        _row("2024-01-15", "A", "D", HS=9, HST=3, AS=7, AST=2, HC=6, AC=5),
    ]
    feat = engineer_rolling_features(pd.DataFrame(rows), windows=(5,))
    last = feat.iloc[-1]
    assert last["rolling_corners_home"] == pytest.approx(10.0)  # not 5.0
    assert last["rolling_sot_home"] == pytest.approx(4.5)


def test_missing_shots_not_stored_as_zero() -> None:
    rows = [
        _row("2024-01-01", "A", "B", HS=10, HST=4, AS=8, AST=3, HC=10, AC=4),
        # Shots missing, corners present → SOT / xG history must skip it.
        _row("2024-01-08", "A", "C", HC=8, AC=2),
        _row("2024-01-15", "A", "D", HS=9, HST=3, AS=7, AST=2, HC=6, AC=5),
    ]
    feat = engineer_rolling_features(pd.DataFrame(rows), windows=(5,))
    last = feat.iloc[-1]
    assert last["rolling_sot_home"] == pytest.approx(4.0)
    assert last["rolling_xg_home"] == pytest.approx(0.1 * 10 + 0.3 * 4)
    assert last["rolling_corners_home"] == pytest.approx(9.0)


def test_imputation_uses_only_earlier_rows() -> None:
    df = pd.DataFrame(
        {
            "Date": pd.to_datetime(
                ["2024-01-01", "2024-01-02", "2024-01-02", "2024-01-03", "2024-01-04"]
            ),
            "rolling_corners_home": [4.0, np.nan, 6.0, np.nan, 1000.0],
        }
    )
    out = impute_rolling_with_league_mean(df)
    # Row 1 (same day as a 6.0 row) only sees the 01-01 value.
    assert out.loc[1, "rolling_corners_home"] == pytest.approx(4.0)
    # Row 3 sees 4.0 and 6.0 — never the future 1000.0.
    assert out.loc[3, "rolling_corners_home"] == pytest.approx(5.0)
    # Legacy whole-frame mean still available explicitly.
    legacy = impute_rolling_with_league_mean(df, point_in_time=False)
    assert legacy.loc[3, "rolling_corners_home"] == pytest.approx((4 + 6 + 1000) / 3)


def test_engineered_features_stable_when_future_rows_appended() -> None:
    rng = np.random.default_rng(1)
    teams = [f"T{i}" for i in range(8)]
    rows = []
    for i in range(60):
        h, a = teams[i % 8], teams[(i + 3) % 8]
        rows.append(
            _row(
                str((pd.Timestamp("2024-01-01") + pd.Timedelta(days=i)).date()),
                h,
                a,
                HS=float(rng.integers(5, 15)),
                HST=float(rng.integers(1, 5)),
                AS=float(rng.integers(5, 15)),
                AST=float(rng.integers(1, 5)),
                HC=float(rng.integers(2, 10)),
                AC=float(rng.integers(2, 10)),
            )
        )
    base = pd.DataFrame(rows)
    # Future rows with extreme stats for brand-new clubs.
    future = pd.DataFrame(
        [
            _row("2024-06-01", "NEW1", "NEW2", HS=99, HST=99, AS=99, AST=99, HC=99, AC=99),
            # Second meeting → non-NaN extreme rolling values that a
            # whole-frame mean would leak back into early debut rows.
            _row("2024-06-02", "NEW1", "NEW2", HS=99, HST=99, AS=99, AST=99, HC=99, AC=99),
        ]
    )
    f1 = engineer_rolling_features(base)
    f2 = engineer_rolling_features(pd.concat([base, future], ignore_index=True))
    cols = [c for c in f1.columns if c.startswith("rolling_") or c.startswith("corner_total")]
    pd.testing.assert_frame_equal(
        f1[cols].reset_index(drop=True),
        f2[cols].iloc[: len(f1)].reset_index(drop=True),
    )
    assert not f2[cols].isna().any().any()
