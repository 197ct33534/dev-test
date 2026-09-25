"""Worker 7 — PaperTrader + SettlementEngineV2 (AH quarters + CLV)."""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from typing import Any, AsyncIterator
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from src.db.schema_v2 import (
    BetSnapshot,
    CanonicalCompetition,
    CanonicalMatch,
    CanonicalTeam,
    PitFeatureSnapshot,
    PredictionSnapshot,
    RawDataLake,
    init_schema_v2_async,
)
from src.execution import (
    PaperTrader,
    SettlementEngineV2,
    compute_clv_value,
    map_settle_status,
    settle_market,
)
from src.execution.settlement_v2 import settle_ah_detailed
from src.risk import RiskEngine, RiskEngineConfig

AS_OF = datetime(2025, 9, 19, 12, 0, tzinfo=timezone.utc)
KICKOFF = datetime(2025, 9, 20, 15, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Pure settlement / CLV
# ---------------------------------------------------------------------------


def test_clv_formula_log_ratio() -> None:
    """clv_value = log(taken_odds / closing_odds)."""
    taken, closing = 2.20, 2.00
    clv = compute_clv_value(taken, closing)
    assert clv == pytest.approx(math.log(taken / closing))
    assert clv > 0.0  # took a better price than close

    # Worse price → negative CLV
    assert compute_clv_value(1.90, 2.10) == pytest.approx(math.log(1.90 / 2.10))
    assert compute_clv_value(1.90, 2.10) < 0.0

    # Invalid → NaN
    assert math.isnan(compute_clv_value(0.0, 2.0))
    assert math.isnan(compute_clv_value(2.0, -1.0))


def test_map_settle_status_win_loss_push() -> None:
    assert map_settle_status("WIN") == "WON"
    assert map_settle_status("LOSS") == "LOST"
    assert map_settle_status("PUSH") == "VOID"
    assert map_settle_status("HALF_WIN") == "HALF_WIN"
    assert map_settle_status("HALF_LOSS") == "HALF_LOSS"


def test_ah_minus_0_25_settlement() -> None:
    """Home AH -0.25: 1-1 → HALF_LOSS; 1-0 → WON."""
    # Reuse service math then map (same path settle_market uses).
    st, pnl, _ = settle_ah_detailed("AH Home", 1, 1, -0.25, 1.95, 100.0)
    assert map_settle_status(st) == "HALF_LOSS"
    assert pnl == pytest.approx(-50.0)

    status, pnl2, _ = settle_market(
        market_type="AH",
        selection="AH Home -0.25",
        taken_odds=1.95,
        stake=100.0,
        ft_home=1,
        ft_away=1,
    )
    assert status == "HALF_LOSS"
    assert pnl2 == pytest.approx(-50.0)

    status_w, pnl_w, _ = settle_market(
        market_type="AH",
        selection="Home",
        taken_odds=1.95,
        stake=100.0,
        ft_home=1,
        ft_away=0,
        line=-0.25,
    )
    assert status_w == "WON"
    assert pnl_w == pytest.approx(95.0)


def test_ah_minus_0_75_settlement() -> None:
    """Home AH -0.75 splits (-1.0, -0.5).

    Score 1-0 → win on -0.5 + push on -1.0 → HALF_WIN.
    Score 0-0 → lose both halves → LOST.
    Score 2-0 → win both → WON.
    """
    half_win, pnl_hw, _ = settle_market(
        market_type="AH",
        selection="AH Home -0.75",
        taken_odds=1.90,
        stake=100.0,
        ft_home=1,
        ft_away=0,
    )
    assert half_win == "HALF_WIN"
    assert pnl_hw == pytest.approx(0.5 * 100.0 * (1.90 - 1.0))

    lost, pnl_l, _ = settle_market(
        market_type="AH",
        selection="AH Home -0.75",
        taken_odds=1.90,
        stake=100.0,
        ft_home=0,
        ft_away=0,
    )
    assert lost == "LOST"
    assert pnl_l == pytest.approx(-100.0)

    won, pnl_w, _ = settle_market(
        market_type="AH",
        selection="AH Home",
        taken_odds=1.90,
        stake=100.0,
        ft_home=2,
        ft_away=0,
        line=-0.75,
    )
    assert won == "WON"
    assert pnl_w == pytest.approx(90.0)


# ---------------------------------------------------------------------------
# PaperTrader (mocked pipeline / risk)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_paper_trader_places_pending_when_stake_positive() -> None:
    match_id = uuid4()
    pred_id = uuid4()
    bet_uuid = uuid4()

    async def fake_pipeline(cid: UUID, as_of: datetime, **kwargs: Any) -> dict[str, Any]:
        assert cid == match_id
        return {
            "prediction_id": pred_id,
            "aggregate_data_score": 90.0,
            "no_bet": False,
            "model_tier": "MODEL_TIER_A",
            "fair_probabilities": {"H": 0.55, "D": 0.25, "A": 0.20},
            "fair_lines": {"H": 1.818, "D": 4.0, "A": 5.0},
            "ensemble": MagicMock(model_confidence_score=0.90),
        }

    session = AsyncMock()
    session.get = AsyncMock(return_value=None)

    # Bypass ExposureManager DB path: mock accept_bet to return a PENDING record.
    exposure = MagicMock()
    exposure.check_exposure.return_value = {"allowed": True, "reason": "ok"}

    async def fake_accept(stake: float, **kwargs: Any) -> dict[str, Any]:
        assert stake > 0
        return {
            "accepted": True,
            "record": {
                "bet_id": bet_uuid,
                "prediction_id": pred_id,
                "status": "PENDING",
                "final_stake_fraction": stake,
                "raw_kelly_fraction": 0.02,
                "data_weight": 0.9,
                "model_weight": 0.9,
            },
            "persist": {"saved": True, "bet_id": bet_uuid},
        }

    exposure.accept_bet = AsyncMock(side_effect=fake_accept)

    # Soft risk config so EV lands OK.
    risk = RiskEngine(
        RiskEngineConfig(
            min_ev_threshold=0.05,
            min_data_score=50.0,
            max_bet_cap_percent=0.05,
        )
    )

    trader = PaperTrader(
        session=session,
        risk_engine=risk,
        exposure_manager=exposure,
        pipeline_fn=fake_pipeline,
        bankroll=10_000.0,
        persist=True,
    )

    ids = await trader.execute_value_bets(
        match_id,
        AS_OF,
        bookmaker_odds={"H": 2.20, "D": 3.50, "A": 4.00},
    )
    assert ids == [bet_uuid]
    exposure.accept_bet.assert_awaited()
    call_kwargs = exposure.accept_bet.await_args.kwargs
    assert call_kwargs["prediction_id"] == pred_id
    assert call_kwargs["stake_result"]["final_stake"] > 0
    assert call_kwargs["stake_result"]["f_kelly"] > 0
    assert call_kwargs["stake_result"]["data_weight"] is not None


@pytest.mark.asyncio
async def test_paper_trader_skips_when_pipeline_no_bet() -> None:
    async def fake_pipeline(*_a: Any, **_k: Any) -> dict[str, Any]:
        return {
            "prediction_id": uuid4(),
            "aggregate_data_score": 0.0,
            "no_bet": True,
            "model_tier": "MODEL_TIER_X",
            "fair_probabilities": {"H": 1 / 3, "D": 1 / 3, "A": 1 / 3},
        }

    trader = PaperTrader(pipeline_fn=fake_pipeline, session=None)
    ids = await trader.execute_value_bets(uuid4(), AS_OF, bookmaker_odds={"H": 2.0})
    assert ids == []


@pytest.mark.asyncio
async def test_paper_trader_zero_stake_inserts_nothing() -> None:
    """Low data_score → RiskEngine NO_BET → no inserts."""

    async def fake_pipeline(*_a: Any, **_k: Any) -> dict[str, Any]:
        return {
            "prediction_id": uuid4(),
            "aggregate_data_score": 20.0,  # below min_data_score
            "no_bet": False,
            "fair_probabilities": {"H": 0.60, "D": 0.20, "A": 0.20},
            "ensemble": MagicMock(model_confidence_score=0.9),
        }

    exposure = MagicMock()
    exposure.accept_bet = AsyncMock()
    trader = PaperTrader(
        pipeline_fn=fake_pipeline,
        exposure_manager=exposure,
        session=None,
    )
    ids = await trader.execute_value_bets(
        uuid4(),
        AS_OF,
        bookmaker_odds={"H": 2.50, "D": 3.5, "A": 4.0},
    )
    assert ids == []
    exposure.accept_bet.assert_not_called()


# ---------------------------------------------------------------------------
# SettlementEngineV2 integration (in-memory SQLite)
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def session() -> AsyncIterator[AsyncSession]:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    await init_schema_v2_async(engine)
    factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    async with factory() as sess:
        yield sess
    await engine.dispose()


async def _seed_finished_match_with_bet(
    session: AsyncSession,
    *,
    market_type: str = "AH",
    selection: str = "AH Home -0.25",
    taken_odds: float = 1.95,
    stake: float = 100.0,
    ft_home: int = 1,
    ft_away: int = 1,
    closing_home: float = 1.85,
) -> tuple[BetSnapshot, CanonicalMatch]:
    home = CanonicalTeam(name="Arsenal", mapping_status="MAPPED")
    away = CanonicalTeam(name="Chelsea", mapping_status="MAPPED")
    comp = CanonicalCompetition(name="EPL", code="EPL", mapping_status="MAPPED")
    session.add_all([home, away, comp])
    await session.flush()

    match = CanonicalMatch(
        home_team_id=home.id,
        away_team_id=away.id,
        competition_id=comp.id,
        kickoff_utc=KICKOFF,
        season_id="2025/2026",
        status="FINISHED",
        ft_home_goals=ft_home,
        ft_away_goals=ft_away,
        ht_home_goals=0,
        ht_away_goals=0,
    )
    session.add(match)
    await session.flush()

    pit = PitFeatureSnapshot(
        canonical_match_id=match.canonical_match_id,
        as_of_time=AS_OF,
        features_json={"stub": 1.0},
        pit_integrity_passed=True,
        aggregate_data_score=90.0,
    )
    session.add(pit)
    await session.flush()

    pred = PredictionSnapshot(
        feature_snapshot_id=pit.feature_snapshot_id,
        model_tier="MODEL_TIER_A",
        model_version="test",
        raw_probabilities={"H": 0.5, "D": 0.25, "A": 0.25},
        calibrated_probabilities={"H": 0.5, "D": 0.25, "A": 0.25},
        fair_lines={"H": 2.0, "D": 4.0, "A": 4.0},
    )
    session.add(pred)
    await session.flush()

    bet = BetSnapshot(
        prediction_id=pred.prediction_id,
        market_type=market_type,
        selection=selection,
        taken_odds=taken_odds,
        raw_kelly_fraction=0.02,
        data_weight=0.9,
        model_weight=0.9,
        final_stake_fraction=0.01,
        final_stake_amount=stake,
        status="PENDING",
    )
    session.add(bet)

    # Pinnacle closing observation at kickoff.
    lake = RawDataLake(
        source="pinnacle",
        entity_type="odds",
        payload={"odds": {"H": closing_home, "D": 3.5, "A": 4.0}},
        observed_at=KICKOFF,
        canonical_match_id=match.canonical_match_id,
    )
    session.add(lake)
    await session.flush()
    return bet, match


@pytest.mark.asyncio
async def test_settle_pending_ah_quarter_and_clv(session: AsyncSession) -> None:
    bet, match = await _seed_finished_match_with_bet(
        session,
        market_type="AH",
        selection="AH Home -0.25",
        taken_odds=1.95,
        stake=100.0,
        ft_home=1,
        ft_away=1,
        closing_home=1.85,
    )

    # For AH, closing lookup may not find H key — inject explicit closing.
    engine = SettlementEngineV2(
        session,
        closing_odds_lookup=lambda *_a, **_k: 1.85,
    )
    summary = await engine.settle_pending_bets(KICKOFF + timedelta(hours=3))

    assert summary["settled"] == 1
    assert summary["details"][0]["status"] == "HALF_LOSS"
    assert summary["details"][0]["pnl"] == pytest.approx(-50.0)

    await session.refresh(bet)
    assert bet.status == "HALF_LOSS"
    assert bet.pnl == pytest.approx(-50.0)
    assert bet.settled_at is not None
    assert bet.sharp_closing_odds == pytest.approx(1.85)
    assert bet.clv_value == pytest.approx(math.log(1.95 / 1.85))


@pytest.mark.asyncio
async def test_settle_pending_ah_minus_0_75_half_win(session: AsyncSession) -> None:
    bet, _match = await _seed_finished_match_with_bet(
        session,
        market_type="AH",
        selection="AH Home -0.75",
        taken_odds=1.90,
        stake=100.0,
        ft_home=1,
        ft_away=0,
        closing_home=1.80,
    )
    engine = SettlementEngineV2(
        session,
        closing_odds_lookup=lambda *_a, **_k: 1.80,
    )
    summary = await engine.settle_pending_bets(KICKOFF + timedelta(hours=2))
    assert summary["settled"] == 1
    assert summary["details"][0]["status"] == "HALF_WIN"

    await session.refresh(bet)
    assert bet.status == "HALF_WIN"
    assert bet.pnl == pytest.approx(45.0)
    assert bet.clv_value == pytest.approx(math.log(1.90 / 1.80))


@pytest.mark.asyncio
async def test_settle_1x2_won_uses_pinnacle_closing(session: AsyncSession) -> None:
    bet, _match = await _seed_finished_match_with_bet(
        session,
        market_type="1X2",
        selection="Home",
        taken_odds=2.20,
        stake=50.0,
        ft_home=2,
        ft_away=1,
        closing_home=2.00,
    )
    engine = SettlementEngineV2(session)  # real Pinnacle lake path
    summary = await engine.settle_pending_bets(KICKOFF + timedelta(hours=2))
    assert summary["settled"] == 1
    assert summary["details"][0]["status"] == "WON"

    await session.refresh(bet)
    assert bet.status == "WON"
    assert bet.pnl == pytest.approx(50.0 * (2.20 - 1.0))
    assert bet.clv_value == pytest.approx(math.log(2.20 / 2.00))
    assert bet.sharp_closing_odds == pytest.approx(2.00)
