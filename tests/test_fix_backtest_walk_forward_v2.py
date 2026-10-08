"""Regression tests: v2 walk-forward backtest evaluates a real model on real odds."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from types import SimpleNamespace
from uuid import uuid4

import pytest

from src.backtest.run_backtest import (
    BacktestConfig,
    BacktestDataError,
    BacktestMatch,
    WalkForwardBacktester,
    _fair_closing,
    _odds_and_records_from_lake,
    load_matches_from_fixtures,
    run_backtest,
)

UTC = timezone.utc


@dataclass
class _RecordingModel:
    """Fake model: records the training window, predicts a fixed home lean."""

    seen: list

    def fit(self, df):  # noqa: ANN001
        self.max_date = df["Date"].max()
        self.n = len(df)
        self.seen.append(self)
        return self

    def predict_match_probs(self, home: str, away: str) -> dict[str, float]:
        return {"H": 0.5, "D": 0.25, "A": 0.25}


def _season(league: str, start: datetime, n: int, *, odds: bool = True) -> list[BacktestMatch]:
    teams = [f"{league}_T{i}" for i in range(6)]
    out = []
    for i in range(n):
        out.append(
            BacktestMatch(
                canonical_match_id=uuid4(),
                kickoff_utc=start + timedelta(days=3 * i),
                home_team=teams[i % 6],
                away_team=teams[(i + 1 + i // 6) % 6] if (i + 1 + i // 6) % 6 != i % 6 else teams[(i + 2) % 6],
                league=league,
                ft_home_goals=i % 3,
                ft_away_goals=(i + 1) % 2,
                bookmaker_odds={"H": 2.2, "D": 3.4, "A": 3.5} if odds else {},
                closing_odds=None,
                dixon_coles_probs={},
                lightgbm_probs=None,
                raw_records=[],
                expected_sources=None,
            )
        )
    return out


def test_walk_forward_fits_on_training_window_only() -> None:
    hist = _season("J1", datetime(2022, 1, 5, tzinfo=UTC), 120)
    test = _season("J1", datetime(2024, 1, 5, tzinfo=UTC), 60)
    seen: list = []
    bt = WalkForwardBacktester(
        BacktestConfig(min_train_matches=20, test_window_months=3),
        model_factory=lambda: _RecordingModel(seen),
    )
    bt._fit_walk_forward(test, hist)

    assert bt.n_folds >= 2
    assert all(m.dixon_coles_probs for m in test)
    for fold in bt.fold_log:
        # No look-ahead: every training kickoff strictly before the test window.
        assert fold["max_train_kickoff"] < fold["test_start"]
        assert fold["league"] == "J1"
    # Later folds train on strictly more data (expanding window).
    sizes = [f["n_train"] for f in bt.fold_log]
    assert sizes == sorted(sizes) and sizes[-1] > sizes[0]


def test_per_league_models_and_league_not_hardcoded() -> None:
    hist = _season("J1", datetime(2023, 1, 5, tzinfo=UTC), 80) + _season(
        "LALIGA", datetime(2023, 1, 6, tzinfo=UTC), 80
    )
    test = _season("J1", datetime(2024, 3, 1, tzinfo=UTC), 10) + _season(
        "LALIGA", datetime(2024, 3, 2, tzinfo=UTC), 10
    )
    seen: list = []
    bt = WalkForwardBacktester(
        BacktestConfig(min_train_matches=20, train_window_years=1),
        model_factory=lambda: _RecordingModel(seen),
    )
    bt._fit_walk_forward(test, hist)
    assert {f["league"] for f in bt.fold_log} == {"J1", "LALIGA"}


@pytest.mark.asyncio
async def test_matches_without_odds_or_model_are_skipped_and_counted() -> None:
    hist = _season("J1", datetime(2023, 1, 5, tzinfo=UTC), 80)
    no_odds = _season("J1", datetime(2024, 2, 1, tzinfo=UTC), 4, odds=False)
    unknown_league = _season("NEWLEAGUE", datetime(2024, 2, 2, tzinfo=UTC), 3)
    seen: list = []
    bt = WalkForwardBacktester(
        BacktestConfig(min_train_matches=20, train_window_years=1),
        model_factory=lambda: _RecordingModel(seen),
    )
    report = await bt.run(no_odds + unknown_league, history=hist, mode="live", source="test")
    assert report.n_skipped_no_odds == 4
    assert report.n_skipped_no_model == 3  # no training data for that league
    assert report.total_bets == 0
    assert report.mode == "live"


def test_fixture_loader_has_no_fake_odds_or_copied_closing(tmp_path) -> None:
    path = tmp_path / "fx.json"
    path.write_text(
        json.dumps(
            [
                {"kickoff_utc": "2024-03-01T12:00:00Z", "home_team": "A", "away_team": "B"},
                {
                    "kickoff_utc": "2024-03-02T12:00:00Z",
                    "home_team": "C",
                    "away_team": "D",
                    "league": "J1",
                    "bookmaker_odds": {"H": 2.0, "D": 3.3, "A": 4.0},
                },
            ]
        ),
        encoding="utf-8",
    )
    a, b = load_matches_from_fixtures(path)
    assert a.bookmaker_odds == {}
    assert b.bookmaker_odds == {"H": 2.0, "D": 3.3, "A": 4.0}
    assert b.closing_odds is None  # never copied from taken odds (CLV ≡ 0 bug)
    assert b.league == "J1"


def _lake_row(payload, observed_at, source="football-data"):
    return SimpleNamespace(raw_id=uuid4(), payload=payload, observed_at=observed_at, source=source)


def test_lake_odds_use_pre_close_entry_and_real_closing() -> None:
    ko = datetime(2024, 5, 1, 18, 0, tzinfo=UTC)
    as_of = ko - timedelta(minutes=30)
    fd = {
        "B365H": 2.1, "B365D": 3.4, "B365A": 3.6,
        "PSCH": 2.0, "PSCD": 3.5, "PSCA": 3.9,
        "FTHG": 2, "FTAG": 1,
    }
    entry, closing, at_close, records = _odds_and_records_from_lake(
        [_lake_row(fd, ko)], kickoff=ko, as_of=as_of
    )
    assert entry == {"H": 2.1, "D": 3.4, "A": 3.6}
    assert closing == {"H": 2.0, "D": 3.5, "A": 3.9}
    assert not at_close
    # Derived record is PIT-clean and carries no result columns.
    assert all(r["observed_at"] <= as_of for r in records)
    assert all("FTHG" not in (r.get("payload") or {}) for r in records)

    # Only an entry snapshot → no closing (no CLV), never the entry copied.
    snap = {"odds": {"H": 2.5, "D": 3.2, "A": 3.0}}
    entry, closing, _, _ = _odds_and_records_from_lake(
        [_lake_row(snap, as_of - timedelta(hours=1), "odds_api")], kickoff=ko, as_of=as_of
    )
    assert entry == {"H": 2.5, "D": 3.2, "A": 3.0}
    assert closing is None

    # Later Pinnacle snapshot before kickoff → closing.
    late = {"odds": {"H": 2.3, "D": 3.3, "A": 3.2}}
    entry, closing, _, _ = _odds_and_records_from_lake(
        [
            _lake_row(snap, as_of - timedelta(hours=1), "odds_api"),
            _lake_row(late, ko - timedelta(minutes=2), "pinnacle"),
        ],
        kickoff=ko,
        as_of=as_of,
    )
    assert closing == {"H": 2.3, "D": 3.3, "A": 3.2}

    # Closing-only row → bet at close, CLV excluded.
    entry, closing, at_close, _ = _odds_and_records_from_lake(
        [_lake_row({"PSCH": 2.0, "PSCD": 3.5, "PSCA": 3.9}, ko)], kickoff=ko, as_of=as_of
    )
    assert at_close and entry == {"H": 2.0, "D": 3.5, "A": 3.9} and closing is None


def test_fair_closing_is_devigged() -> None:
    fair = _fair_closing({"H": 2.0, "D": 3.5, "A": 3.9})
    assert fair is not None
    assert sum(1.0 / v for v in fair.values()) == pytest.approx(1.0)
    assert fair["H"] > 2.0
    assert _fair_closing(None) is None


@pytest.mark.asyncio
async def test_live_mode_never_falls_back_to_synthetic(tmp_path) -> None:
    from src.db.schema_v2 import init_schema_v2_async
    from sqlalchemy.ext.asyncio import create_async_engine

    url = f"sqlite+aiosqlite:///{(tmp_path / 'empty.db').as_posix()}"
    eng = create_async_engine(url)
    await init_schema_v2_async(eng)
    await eng.dispose()

    cfg = BacktestConfig(
        start_date=date(2024, 1, 1),
        end_date=date(2024, 6, 1),
        database_url=url,
    )
    with pytest.raises(BacktestDataError, match="synthetic"):
        await run_backtest(cfg)


@pytest.mark.asyncio
async def test_synthetic_only_on_request_and_labelled() -> None:
    cfg = BacktestConfig(
        start_date=date(2026, 1, 1),
        end_date=date(2026, 9, 1),
        synthetic=True,
        seed=5,
    )
    report = await run_backtest(cfg)
    assert report.mode == "synthetic"
    assert report.n_matches == 50
    assert math.isfinite(report.final_bankroll)
