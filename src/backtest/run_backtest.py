"""Walk-Forward Backtesting Master — CLI async chronological replay.

Usage
-----
    python -m src.backtest.run_backtest --dry-run
    python -m src.backtest.run_backtest --start-date 2024-01-01 --end-date 2026-09-01 \\
        --initial-bankroll 10000

Replay loop
-----------
1. Iterate matches chronologically in ``[start, end]``
2. ``as_of_time = kickoff_utc - 30 minutes``
3. ``run_quant_pipeline`` → price only when PIT ok and aggregate_data_score ≥ 50
4. ``RiskEngine.calculate_stake`` / PaperTrader → paper bet when stake > 0
5. After kickoff, ``SettlementEngineV2`` (or in-memory settle) → PnL + CLV

Model evaluation
----------------
Matches without model probabilities (DB / fixtures) get Dixon–Coles 1X2
probabilities from :class:`WalkForwardValidator` folds: for each fold one
model per league is fit on finished matches with ``kickoff < test_start`` only
and predicts that fold's test matches (no look-ahead). LightGBM is not
refit here (``lightgbm_probs=None``).

Data honesty
------------
* Matches without real 1X2 odds are skipped (``n_skipped_no_odds``).
* ``closing_odds`` come from a real closing snapshot (football-data
  ``PSCH``/``AvgCH``/``B365CH`` or a later lake odds row) or are ``None`` —
  CLV is then excluded, never computed against the entry price itself.
* An empty / unreachable DB raises; synthetic data runs only with
  ``--synthetic`` and is reported as ``mode="synthetic"``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import random
import sys
from collections import Counter
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence
from uuid import UUID, uuid4

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from src.backtest.report import (
    QuantReport,
    build_quant_report,
    merge_tier_counter,
    render_quant_report,
)
from src.db.schema_v2 import (
    CanonicalCompetition,
    CanonicalMatch,
    CanonicalTeam,
    PitFeatureSnapshot,
    PredictionSnapshot,
    RawDataLake,
    init_schema_v2_async,
)
from src.execution.paper_trader import build_1x2_candidates
from src.execution.settlement_v2 import SettlementEngineV2, settle_market
from src.pipeline.run_pipeline import run_quant_pipeline
from src.risk import ExposureManager, RiskEngine, RiskEngineConfig
from src.validation.metrics import QuantMetrics, devig_odds
from src.validation.walk_forward import WalkForwardValidator

logger = logging.getLogger(__name__)

DEFAULT_CONCURRENCY = 8
DRY_RUN_MATCH_COUNT = 50
PIT_LEAD = timedelta(minutes=30)
# Lake odds observed up to this long after kickoff still count as "closing".
CLOSING_GRACE = timedelta(minutes=5)

# football-data.co.uk 1X2 column families.
_PRE_CLOSE_FAMILIES: dict[str, tuple[str, str, str]] = {
    "B365": ("B365H", "B365D", "B365A"),
    "PS": ("PSH", "PSD", "PSA"),
    "Avg": ("AvgH", "AvgD", "AvgA"),
}
_CLOSING_FAMILIES: dict[str, tuple[str, str, str]] = {
    "PSC": ("PSCH", "PSCD", "PSCA"),
    "AvgC": ("AvgCH", "AvgCD", "AvgCA"),
    "B365C": ("B365CH", "B365CD", "B365CA"),
}


class BacktestDataError(RuntimeError):
    """Raised when a live backtest has no usable real data (no silent fallback)."""

_TEAMS = (
    "Arsenal",
    "Chelsea",
    "Liverpool",
    "Man City",
    "Man United",
    "Tottenham",
    "Newcastle",
    "Aston Villa",
    "Brighton",
    "West Ham",
    "Fulham",
    "Brentford",
    "Crystal Palace",
    "Everton",
    "Wolves",
    "Bournemouth",
)


# ---------------------------------------------------------------------------
# Match fixture model
# ---------------------------------------------------------------------------


@dataclass
class BacktestMatch:
    """One fixture ready for PIT pricing + settlement."""

    canonical_match_id: UUID
    kickoff_utc: datetime
    home_team: str
    away_team: str
    league: str = "EPL"
    status: str = "FINISHED"
    ft_home_goals: int | None = None
    ft_away_goals: int | None = None
    # Real entry 1X2 odds (empty ⇒ match is skipped, never priced on fakes).
    bookmaker_odds: dict[str, float] = field(default_factory=dict)
    # Real closing 1X2 market or None ⇒ excluded from CLV stats.
    closing_odds: dict[str, float] | None = None
    # Empty ⇒ filled by the walk-forward Dixon–Coles fit (or skipped).
    dixon_coles_probs: dict[str, float] = field(default_factory=dict)
    lightgbm_probs: dict[str, float] | None = None
    raw_records: list[dict[str, Any]] = field(default_factory=list)
    expected_sources: list[str] | None = field(
        default_factory=lambda: ["odds_api", "stats", "news"]
    )
    quality_band: str = "A"  # A | B | X (synthetic hint)
    competition: str | None = None


@dataclass
class BacktestConfig:
    start_date: date = date(2024, 1, 1)
    end_date: date = date(2026, 9, 1)
    initial_bankroll: float = 10_000.0
    dry_run: bool = False
    concurrency: int = DEFAULT_CONCURRENCY
    fixtures_path: Path | None = None
    database_url: str | None = None
    seed: int = 42
    # Synthetic fixtures only when explicitly requested (mode="synthetic").
    synthetic: bool = False
    # Walk-forward model fitting (DB / fixture matches without model probs).
    train_window_years: int = 3
    test_window_months: int = 3
    min_train_matches: int = 60
    dc_xi: float = 0.0018
    # Entry price preference for football-data rows (pre-close snapshot).
    entry_odds_family: str = "B365"
    # When a row only has closing odds, bet at the close (CLV then excluded).
    allow_closing_as_entry: bool = True


@dataclass
class _PaperBet:
    """In-memory paper bet (also mirrored to DB when session available)."""

    bet_id: UUID
    match_id: UUID
    prediction_id: UUID
    market_type: str
    selection: str
    taken_odds: float
    closing_odds: float | None
    stake: float
    probability: float
    model_tier: str
    kickoff_utc: datetime
    status: str = "PENDING"
    pnl: float | None = None
    clv_value: float | None = None


# ---------------------------------------------------------------------------
# Data sources
# ---------------------------------------------------------------------------


def generate_synthetic_matches(
    *,
    n: int = DRY_RUN_MATCH_COUNT,
    end: datetime | None = None,
    seed: int = 42,
) -> list[BacktestMatch]:
    """Generate finished fixtures with PIT-clean records and mixed edges."""
    rng = random.Random(seed)
    end_dt = end or datetime(2026, 9, 1, tzinfo=timezone.utc)
    if end_dt.tzinfo is None:
        end_dt = end_dt.replace(tzinfo=timezone.utc)

    matches: list[BacktestMatch] = []
    for i in range(n):
        kickoff = end_dt - timedelta(days=(n - i), hours=rng.randint(12, 20))
        as_of = kickoff - PIT_LEAD
        home, away = rng.sample(_TEAMS, 2)
        band = rng.choices(["A", "B", "X"], weights=[0.55, 0.30, 0.15], k=1)[0]

        # Model lean home often; book prices lag → positive EV on Home sometimes.
        p_h = rng.uniform(0.42, 0.58)
        p_d = rng.uniform(0.22, 0.30)
        p_a = max(0.05, 1.0 - p_h - p_d)
        # Renormalize
        s = p_h + p_d + p_a
        p_h, p_d, p_a = p_h / s, p_d / s, p_a / s

        # Book odds with vig (~5%).
        fair_odds = {"H": 1.0 / p_h, "D": 1.0 / p_d, "A": 1.0 / p_a}
        # Soften book toward equal so model has edge on favourite.
        book = {
            "H": round(fair_odds["H"] * rng.uniform(1.02, 1.12), 2),
            "D": round(fair_odds["D"] * rng.uniform(0.95, 1.05), 2),
            "A": round(fair_odds["A"] * rng.uniform(0.95, 1.08), 2),
        }
        # Closing drifts slightly toward true probs.
        close = {
            "H": round(book["H"] * rng.uniform(0.94, 1.02), 2),
            "D": round(book["D"] * rng.uniform(0.97, 1.03), 2),
            "A": round(book["A"] * rng.uniform(0.97, 1.04), 2),
        }

        # Sample FT result from model probs.
        r = rng.random()
        if r < p_h:
            ft_h, ft_a = rng.choice([(1, 0), (2, 0), (2, 1), (3, 1)])
        elif r < p_h + p_d:
            ft_h, ft_a = rng.choice([(0, 0), (1, 1), (2, 2)])
        else:
            ft_h, ft_a = rng.choice([(0, 1), (0, 2), (1, 2), (1, 3)])

        raw_records = _synthetic_raw_records(as_of, band=band, rng=rng)
        mid = uuid4()
        matches.append(
            BacktestMatch(
                canonical_match_id=mid,
                kickoff_utc=kickoff,
                home_team=home,
                away_team=away,
                league="EPL",
                status="FINISHED",
                ft_home_goals=ft_h,
                ft_away_goals=ft_a,
                bookmaker_odds=book,
                closing_odds=close,
                dixon_coles_probs={"H": p_h, "D": p_d, "A": p_a},
                lightgbm_probs={
                    "H": max(0.05, p_h + rng.uniform(-0.03, 0.03)),
                    "D": max(0.05, p_d + rng.uniform(-0.02, 0.02)),
                    "A": max(0.05, p_a + rng.uniform(-0.03, 0.03)),
                },
                raw_records=raw_records,
                quality_band=band,
            )
        )
        # Renormalize LGBM
        last = matches[-1]
        s2 = sum(last.lightgbm_probs.values())
        last.lightgbm_probs = {k: v / s2 for k, v in last.lightgbm_probs.items()}

    matches.sort(key=lambda m: m.kickoff_utc)
    return matches


def _synthetic_raw_records(
    as_of: datetime,
    *,
    band: str,
    rng: random.Random,
) -> list[dict[str, Any]]:
    """PIT-valid records; band X injects a future leak."""
    if band == "A":
        sources = [
            ("odds_api", 98.0, 25),
            ("stats", 97.0, 15),
            ("news", 96.0, 8),
        ]
    elif band == "B":
        sources = [("odds_api", 55.0, 400)]
    else:
        sources = [
            ("odds_api", 90.0, 20),
            ("stats", 88.0, 10),
        ]

    records: list[dict[str, Any]] = []
    for i, (src, q, mins_ago) in enumerate(sources):
        records.append(
            {
                "id": f"{src}_{i}",
                "observed_at": as_of - timedelta(minutes=mins_ago),
                "source_type": src,
                "source_quality": q,
            }
        )
    if band == "X":
        records.append(
            {
                "id": "leak",
                "observed_at": as_of + timedelta(hours=2),
                "source_type": "oracle",
                "source_quality": 99.0,
            }
        )
    return records


def load_matches_from_fixtures(path: Path) -> list[BacktestMatch]:
    """Load BacktestMatch list from a JSON fixture file."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("matches") or data.get("fixtures") or []
    out: list[BacktestMatch] = []
    for row in data:
        kickoff = _parse_dt(row["kickoff_utc"])
        mid = UUID(str(row.get("canonical_match_id") or uuid4()))
        as_of = kickoff - PIT_LEAD
        raw = row.get("raw_records")
        if not raw:
            band = str(row.get("quality_band") or "A")
            raw = _synthetic_raw_records(as_of, band=band, rng=random.Random(0))
        out.append(
            BacktestMatch(
                canonical_match_id=mid,
                kickoff_utc=kickoff,
                home_team=str(row.get("home_team") or "Home"),
                away_team=str(row.get("away_team") or "Away"),
                league=str(row.get("league") or "EPL"),
                status=str(row.get("status") or "FINISHED"),
                ft_home_goals=row.get("ft_home_goals"),
                ft_away_goals=row.get("ft_away_goals"),
                # No fake fallbacks: missing odds ⇒ skipped; missing closing ⇒
                # no CLV (never copied from the entry price).
                bookmaker_odds=_valid_1x2(row.get("bookmaker_odds")) or {},
                closing_odds=_valid_1x2(row.get("closing_odds")),
                dixon_coles_probs=dict(row.get("dixon_coles_probs") or {}),
                lightgbm_probs=dict(row.get("lightgbm_probs") or {}) or None,
                raw_records=list(raw),
                expected_sources=list(
                    row.get("expected_sources") or ["odds_api", "stats", "news"]
                ),
                quality_band=str(row.get("quality_band") or "A"),
                competition=row.get("competition"),
            )
        )
    out.sort(key=lambda m: m.kickoff_utc)
    return out


async def load_matches_from_db(
    session: AsyncSession,
    *,
    start: datetime,
    end: datetime,
    stats: dict[str, int] | None = None,
    entry_odds_family: str = "B365",
    allow_closing_as_entry: bool = True,
) -> list[BacktestMatch]:
    """Load canonical matches in ``[start, end)`` with **real** odds only.

    * ``league`` / ``competition`` come from the match's competition.
    * Entry odds: latest lake odds snapshot observed ≤ ``kickoff − PIT_LEAD``,
      else the football-data pre-close columns (``B365H``/``PSH``/``AvgH``),
      else (``allow_closing_as_entry``) the closing columns with
      ``closing_odds=None`` (CLV undefined). No odds ⇒ ``bookmaker_odds={}``
      (the backtester skips it but still uses it for model training).
    * Closing odds: football-data closing columns (``PSCH``…), else a lake
      odds snapshot observed after the entry snapshot and ≤ kickoff + grace
      (Pinnacle preferred); otherwise ``None``.
    * ``raw_records``: real lake rows observed ≤ as-of, plus one derived
      record for football-data pre-close odds (known before kickoff).
    * Model probabilities are left empty — the walk-forward fit fills them.

    ``stats`` (optional) receives ``n_loaded``, ``n_no_odds``,
    ``n_entry_at_close`` and ``n_with_closing`` counters.
    """
    from sqlalchemy.orm import selectinload

    counters = stats if stats is not None else {}
    for key in ("n_loaded", "n_no_odds", "n_entry_at_close", "n_with_closing"):
        counters.setdefault(key, 0)

    stmt = (
        select(CanonicalMatch)
        .options(
            selectinload(CanonicalMatch.home_team),
            selectinload(CanonicalMatch.away_team),
            selectinload(CanonicalMatch.competition),
        )
        .where(
            CanonicalMatch.kickoff_utc >= start,
            CanonicalMatch.kickoff_utc < end,
        )
        .order_by(CanonicalMatch.kickoff_utc.asc())
    )
    rows = (await session.execute(stmt)).scalars().unique().all()
    if not rows:
        return []

    lake = await _lake_rows_by_match(session, [m.canonical_match_id for m in rows])

    out: list[BacktestMatch] = []
    for m in rows:
        kickoff = _ensure_aware(m.kickoff_utc)
        as_of = kickoff - PIT_LEAD
        lake_rows = lake.get(m.canonical_match_id, [])
        entry, closing, at_close, records = _odds_and_records_from_lake(
            lake_rows,
            kickoff=kickoff,
            as_of=as_of,
            entry_family=entry_odds_family,
            allow_closing_as_entry=allow_closing_as_entry,
        )
        counters["n_loaded"] += 1
        if not entry:
            counters["n_no_odds"] += 1
        if at_close:
            counters["n_entry_at_close"] += 1
        if closing:
            counters["n_with_closing"] += 1

        league = _competition_label(m) or "UNKNOWN"
        out.append(
            BacktestMatch(
                canonical_match_id=m.canonical_match_id,
                kickoff_utc=kickoff,
                home_team=_team_label(m, "home"),
                away_team=_team_label(m, "away"),
                league=league,
                competition=league,
                status=str(m.status or "SCHEDULED"),
                ft_home_goals=m.ft_home_goals,
                ft_away_goals=m.ft_away_goals,
                bookmaker_odds=entry or {},
                closing_odds=closing,
                dixon_coles_probs={},
                lightgbm_probs=None,
                raw_records=records,
                # Mirror live scans: coverage from sources actually present.
                expected_sources=None,
            )
        )
    return out


async def _lake_rows_by_match(
    session: AsyncSession,
    match_ids: Sequence[UUID],
    *,
    chunk: int = 500,
) -> dict[UUID, list[RawDataLake]]:
    """All lake rows for ``match_ids`` (chunked IN queries), oldest first."""
    out: dict[UUID, list[RawDataLake]] = {}
    ids = list(match_ids)
    for i in range(0, len(ids), chunk):
        part = ids[i : i + chunk]
        stmt = (
            select(RawDataLake)
            .where(RawDataLake.canonical_match_id.in_(part))
            .order_by(RawDataLake.observed_at.asc())
        )
        for row in (await session.execute(stmt)).scalars().all():
            if row.canonical_match_id is not None:
                out.setdefault(row.canonical_match_id, []).append(row)
    return out


def _odds_and_records_from_lake(
    rows: Sequence[Any],
    *,
    kickoff: datetime,
    as_of: datetime,
    entry_family: str = "B365",
    allow_closing_as_entry: bool = True,
) -> tuple[
    dict[str, float] | None,
    dict[str, float] | None,
    bool,
    list[dict[str, Any]],
]:
    """Return ``(entry_odds, closing_odds, entry_is_closing, raw_records)``."""
    entry: dict[str, float] | None = None
    entry_obs: datetime | None = None
    fd_pre: dict[str, float] | None = None
    fd_close: dict[str, float] | None = None
    late: list[tuple[datetime, bool, dict[str, float]]] = []
    records: list[dict[str, Any]] = []

    for row in rows:
        payload = row.payload if isinstance(getattr(row, "payload", None), Mapping) else {}
        obs = _ensure_aware(row.observed_at) if row.observed_at is not None else None
        source = str(getattr(row, "source", "") or "")

        # football-data history rows (observed_at = kickoff, carry results):
        # take the pre-close / closing columns, never the row itself.
        pre = _family_odds(payload, _PRE_CLOSE_FAMILIES, prefer=entry_family)
        close = _family_odds(payload, _CLOSING_FAMILIES, prefer="PSC")
        if pre and fd_pre is None:
            fd_pre = pre
            records.append(
                {
                    "id": f"{row.raw_id}:pre_close_odds",
                    "observed_at": as_of,
                    "source_type": source or "football-data",
                    "derived": "pre_close_odds",
                    "payload": {"odds": dict(pre)},
                }
            )
        if close and fd_close is None:
            fd_close = close

        snap = _odds_map_from_payload(payload)
        if obs is None:
            continue
        if obs <= as_of:
            records.append(
                {
                    "id": str(row.raw_id),
                    "observed_at": obs,
                    "source_type": source,
                    "payload": dict(payload),
                }
            )
            if snap and (entry_obs is None or obs >= entry_obs):
                entry, entry_obs = snap, obs
        elif snap and obs <= kickoff + CLOSING_GRACE:
            late.append((obs, "pinnacle" in source.lower(), snap))

    if entry is None and fd_pre:
        entry = fd_pre

    closing: dict[str, float] | None = fd_close
    if closing is None and late:
        # Pinnacle first, then the latest snapshot before kickoff (+grace).
        late.sort(key=lambda t: (t[1], t[0]))
        closing = late[-1][2]

    at_close = False
    if entry is None and closing is not None and allow_closing_as_entry:
        entry, closing, at_close = closing, None, True
    return entry, closing, at_close, records


def _family_odds(
    payload: Mapping[str, Any],
    families: Mapping[str, tuple[str, str, str]],
    *,
    prefer: str,
) -> dict[str, float] | None:
    order = [prefer] + [f for f in families if f != prefer]
    for fam in order:
        cols = families.get(fam)
        if not cols or not all(c in payload for c in cols):
            continue
        odds = _valid_1x2(dict(zip(("H", "D", "A"), (payload[c] for c in cols))))
        if odds:
            return odds
    return None


def _odds_map_from_payload(payload: Mapping[str, Any]) -> dict[str, float] | None:
    for key in ("odds", "bookmaker_odds", "closing_odds"):
        raw = payload.get(key)
        if isinstance(raw, Mapping):
            nested = raw.get("1X2") if isinstance(raw.get("1X2"), Mapping) else raw
            odds = _valid_1x2(nested)
            if odds:
                return odds
    return None


def _valid_1x2(raw: Any) -> dict[str, float] | None:
    """``{"H","D","A"}`` decimal odds (> 1.0, finite) or ``None``."""
    if not isinstance(raw, Mapping):
        return None
    aliases = {
        "H": ("H", "home", "Home", "1"),
        "D": ("D", "draw", "Draw", "X"),
        "A": ("A", "away", "Away", "2"),
    }
    out: dict[str, float] = {}
    for key, names in aliases.items():
        val = None
        for name in names:
            if name in raw and raw[name] is not None:
                try:
                    val = float(raw[name])
                except (TypeError, ValueError):
                    val = None
                break
        if val is None or not math.isfinite(val) or val <= 1.0:
            return None
        out[key] = val
    return out


def _competition_label(match: Any) -> str | None:
    comp = getattr(match, "competition", None)
    if comp is None:
        return None
    code = getattr(comp, "code", None)
    if code:
        return str(code)
    name = getattr(comp, "name", None)
    return str(name) if name else None


def _team_label(match: Any, side: str) -> str:
    team = getattr(match, f"{side}_team", None)
    name = getattr(team, "name", None) if team is not None else None
    if name:
        return str(name)
    tid = getattr(match, f"{side}_team_id", None)
    return f"team:{tid}" if tid is not None else side.capitalize()


async def _odds_from_lake(
    session: AsyncSession, match_id: UUID
) -> dict[str, float] | None:
    """Backward-compatible helper: latest real 1X2 odds map for a match."""
    rows = (await _lake_rows_by_match(session, [match_id])).get(match_id, [])
    for row in reversed(rows):
        payload = row.payload if isinstance(row.payload, Mapping) else {}
        odds = _odds_map_from_payload(payload)
        if odds:
            return odds
    return None


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class WalkForwardBacktester:
    """Chronological paper-trading replay with gated pricing + settlement."""

    def __init__(
        self,
        config: BacktestConfig,
        *,
        model_factory: Any | None = None,
    ) -> None:
        self.config = config
        # ``model_factory()`` → object with ``fit(df)`` / ``predict_match_probs``
        # (default: Dixon–Coles). Injectable for tests.
        self._model_factory = model_factory
        self.risk = RiskEngine(
            RiskEngineConfig(min_data_score=50.0, min_ev_threshold=0.05)
        )
        self.exposure = ExposureManager()
        self.bankroll = float(config.initial_bankroll)
        self.equity: list[float] = [self.bankroll]
        self.paper_bets: list[_PaperBet] = []
        self.settled: list[dict[str, Any]] = []
        self.y_true: list[int] = []
        self.y_prob: list[float] = []
        self.tier_counts: Counter[str] = Counter()
        self.n_priced = 0
        self.n_skipped_gate = 0
        self.n_skipped_no_odds = 0
        self.n_skipped_no_model = 0
        self.n_folds = 0
        self.fold_log: list[dict[str, Any]] = []
        self.mode = "dry-run" if config.dry_run else "live"
        if config.synthetic:
            self.mode = "synthetic"
        self.source = ""
        self._last_day: date | None = None
        self._sem = asyncio.Semaphore(max(1, int(config.concurrency)))

    async def run(
        self,
        matches: Sequence[BacktestMatch],
        *,
        history: Sequence[BacktestMatch] | None = None,
        mode: str | None = None,
        source: str | None = None,
    ) -> QuantReport:
        """Replay ``matches`` chronologically; return QuantReport.

        ``history`` holds earlier finished matches used only as model training
        data (never bet). Matches lacking model probabilities are predicted by
        per-fold Dixon–Coles fits trained strictly before each test window.
        """
        if mode:
            self.mode = str(mode)
        if source is not None:
            self.source = str(source)
        ordered = sorted(matches, key=lambda m: m.kickoff_utc)
        if not ordered:
            return self._report(n_matches=0)

        self._fit_walk_forward(ordered, list(history or []))

        priceable: list[BacktestMatch] = []
        for m in ordered:
            if not _valid_1x2(m.bookmaker_odds):
                self.n_skipped_no_odds += 1
                continue
            if not m.dixon_coles_probs:
                self.n_skipped_no_model += 1
                continue
            priceable.append(m)
        if self.n_skipped_no_odds:
            logger.info("Skipped %d matches without real odds", self.n_skipped_no_odds)
        if self.n_skipped_no_model:
            logger.info(
                "Skipped %d matches without a walk-forward model prediction",
                self.n_skipped_no_model,
            )

        engine = create_async_engine("sqlite+aiosqlite:///:memory:")
        await init_schema_v2_async(engine)
        factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

        async with factory() as session:
            await self._seed_matches(session, priceable)
            await session.commit()

            # Day-batch: parallel pipeline (no shared session), then sequential
            # persist / stake / settle (bankroll + AsyncSession are serial).
            for day, day_matches in _group_by_day(priceable):
                if self._last_day is not None and day != self._last_day:
                    self.exposure.reset_daily()
                self._last_day = day

                pipe_outs = await self._price_batch(day_matches)
                for match, pipe_out in zip(day_matches, pipe_outs):
                    await self._persist_pipeline(session, pipe_out)
                    await self._place_from_pipeline(session, match, pipe_out)

                # Settle anything whose kickoff has passed by end of day.
                settle_as_of = datetime(
                    day.year, day.month, day.day, 23, 59, tzinfo=timezone.utc
                ) + timedelta(days=1)
                await self._settle(session, settle_as_of)

            await session.commit()

        await engine.dispose()
        return self._report(n_matches=len(ordered))

    def _report(self, *, n_matches: int) -> QuantReport:
        return build_quant_report(
            settled_bets=self.settled,
            equity_curve=self.equity,
            y_true=self.y_true,
            y_prob=self.y_prob,
            tier_counts=self.tier_counts,
            initial_bankroll=self.config.initial_bankroll,
            final_bankroll=self.bankroll,
            n_matches=n_matches,
            n_priced=self.n_priced,
            n_skipped_gate=self.n_skipped_gate,
            mode=self.mode,
            n_skipped_no_odds=self.n_skipped_no_odds,
            n_skipped_no_model=self.n_skipped_no_model,
            n_folds=self.n_folds,
            source=self.source,
        )

    # ------------------------------------------------------------------
    # Walk-forward model fitting
    # ------------------------------------------------------------------

    def _new_model(self) -> Any:
        if self._model_factory is not None:
            return self._model_factory()
        from src.dixon_coles import DixonColesModel

        return DixonColesModel(xi=float(self.config.dc_xi))

    def _fit_walk_forward(
        self,
        matches: Sequence[BacktestMatch],
        history: Sequence[BacktestMatch],
    ) -> None:
        """Fill ``dixon_coles_probs`` for matches lacking them, fold by fold.

        For each :class:`TimeSplit` and league, the model is fit only on
        finished matches with ``train_start <= kickoff < test_start`` and
        predicts that league's matches in ``[test_start, test_end)``.
        """
        need = [m for m in matches if not m.dixon_coles_probs]
        if not need:
            return

        pool: dict[UUID, BacktestMatch] = {}
        for m in list(history) + list(matches):
            if m.ft_home_goals is None or m.ft_away_goals is None:
                continue
            pool.setdefault(m.canonical_match_id, m)
        train_pool = sorted(pool.values(), key=lambda m: _ensure_aware(m.kickoff_utc))

        years = [_ensure_aware(m.kickoff_utc).year for m in need]
        window = max(1, int(self.config.train_window_years))
        try:
            splits = WalkForwardValidator.generate_time_splits(
                min(years) - window,
                max(years),
                train_window_years=window,
                test_window_months=max(1, int(self.config.test_window_months)),
            )
        except ValueError:
            splits = []
        logger.info("Walk-forward folds: %d", len(splits))

        min_train = max(1, int(self.config.min_train_matches))
        for split in splits:
            test = [
                m
                for m in need
                if split.test_start <= _ensure_aware(m.kickoff_utc) < split.test_end
            ]
            if not test:
                continue
            by_league: dict[str, list[BacktestMatch]] = {}
            for m in test:
                by_league.setdefault(m.league, []).append(m)

            for league, league_test in by_league.items():
                train = [
                    p
                    for p in train_pool
                    if p.league == league
                    and split.train_start <= _ensure_aware(p.kickoff_utc) < split.test_start
                ]
                if len(train) < min_train:
                    continue
                try:
                    model = self._new_model().fit(_training_frame(train))
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "Model fit failed league=%s fold=%s: %s",
                        league,
                        split.test_start.date(),
                        exc,
                    )
                    continue
                self.n_folds += 1
                self.fold_log.append(
                    {
                        "league": league,
                        "train_start": split.train_start,
                        "train_end": split.train_end,
                        "test_start": split.test_start,
                        "test_end": split.test_end,
                        "n_train": len(train),
                        "max_train_kickoff": max(
                            _ensure_aware(p.kickoff_utc) for p in train
                        ),
                        "n_test": len(league_test),
                    }
                )
                for m in league_test:
                    try:
                        probs = model.predict_match_probs(m.home_team, m.away_team)
                    except Exception:  # noqa: BLE001
                        continue
                    clean = {
                        k: float(probs[k]) for k in ("H", "D", "A") if k in probs
                    }
                    total = sum(clean.values())
                    if len(clean) == 3 and total > 0 and math.isfinite(total):
                        m.dixon_coles_probs = {k: v / total for k, v in clean.items()}

    async def _seed_matches(
        self, session: AsyncSession, matches: Sequence[BacktestMatch]
    ) -> None:
        team_ids: dict[str, UUID] = {}
        comp_ids: dict[str, UUID] = {}

        for m in matches:
            league = str(m.league or "UNKNOWN")
            if league not in comp_ids:
                comp = CanonicalCompetition(
                    name=str(m.competition or league),
                    code=league[:32],
                    mapping_status="MAPPED",
                )
                session.add(comp)
                await session.flush()
                comp_ids[league] = comp.id
            for name in (m.home_team, m.away_team):
                if name not in team_ids:
                    t = CanonicalTeam(name=name, mapping_status="MAPPED")
                    session.add(t)
                    await session.flush()
                    team_ids[name] = t.id

            row = CanonicalMatch(
                canonical_match_id=m.canonical_match_id,
                competition_id=comp_ids[league],
                home_team_id=team_ids[m.home_team],
                away_team_id=team_ids[m.away_team],
                kickoff_utc=m.kickoff_utc,
                season_id=str(m.kickoff_utc.year),
                status=m.status,
                ft_home_goals=m.ft_home_goals,
                ft_away_goals=m.ft_away_goals,
            )
            session.add(row)
            # Pinnacle closing for CLV.
            if m.closing_odds:
                session.add(
                    RawDataLake(
                        source="pinnacle",
                        entity_type="odds",
                        payload={"odds": dict(m.closing_odds)},
                        observed_at=m.kickoff_utc,
                        canonical_match_id=m.canonical_match_id,
                    )
                )
        await session.flush()

    async def _price_batch(
        self,
        matches: Sequence[BacktestMatch],
    ) -> list[dict[str, Any]]:
        """Bounded parallel pricing; sessions are intentionally omitted."""

        async def _one(m: BacktestMatch) -> dict[str, Any]:
            async with self._sem:
                as_of = _ensure_aware(m.kickoff_utc) - PIT_LEAD
                match_data: dict[str, Any] = {}
                if m.expected_sources:
                    match_data["expected_sources"] = list(m.expected_sources)
                if m.quality_band == "B":
                    match_data["expected_sources"] = [
                        "odds_api",
                        "stats",
                        "news",
                        "injury",
                    ]
                    match_data["consistency_conflicts"] = 1
                return dict(
                    await run_quant_pipeline(
                        m.canonical_match_id,
                        as_of,
                        raw_records=m.raw_records,
                        dixon_coles_probs=m.dixon_coles_probs or None,
                        lightgbm_probs=m.lightgbm_probs or None,
                        bookmaker_odds=m.bookmaker_odds,
                        match_data=match_data,
                        session=None,
                        persist=False,
                    )
                )

        return list(await asyncio.gather(*[_one(m) for m in matches]))

    async def _persist_pipeline(
        self, session: AsyncSession, pipe_out: Mapping[str, Any]
    ) -> None:
        """Write pit + prediction snapshots so BetSnapshot FKs resolve."""
        pit_payload = pipe_out.get("pit_feature_snapshot") or {}
        pred_payload = pipe_out.get("prediction_snapshot") or {}
        if not pit_payload or not pred_payload:
            return
        pit = PitFeatureSnapshot(
            feature_snapshot_id=pit_payload["feature_snapshot_id"],
            canonical_match_id=pit_payload["canonical_match_id"],
            as_of_time=pit_payload["as_of_time"],
            feature_schema_version=str(
                pit_payload.get("feature_schema_version") or "v2_stub"
            ),
            features_json=dict(pit_payload.get("features_json") or {}),
            data_quality_metrics=pit_payload.get("data_quality_metrics"),
            aggregate_data_score=pit_payload.get("aggregate_data_score"),
            pit_integrity_passed=bool(pit_payload.get("pit_integrity_passed", True)),
            lineage_mapping=pit_payload.get("lineage_mapping"),
        )
        pred = PredictionSnapshot(
            prediction_id=pred_payload["prediction_id"],
            feature_snapshot_id=pred_payload["feature_snapshot_id"],
            model_tier=str(pred_payload.get("model_tier") or "MODEL_TIER_X"),
            model_version=str(pred_payload.get("model_version") or "backtest"),
            calibrator_version=pred_payload.get("calibrator_version"),
            raw_probabilities=dict(pred_payload.get("raw_probabilities") or {}),
            calibrated_probabilities=pred_payload.get("calibrated_probabilities"),
            fair_lines=pred_payload.get("fair_lines"),
            model_confidence_score=pred_payload.get("model_confidence_score"),
        )
        session.add(pit)
        session.add(pred)
        await session.flush()

    async def _place_from_pipeline(
        self,
        session: AsyncSession,
        match: BacktestMatch,
        pipe_out: Mapping[str, Any],
    ) -> None:
        merge_tier_counter(self.tier_counts, str(pipe_out.get("model_tier")))
        pit_ok = bool(pipe_out.get("pit_integrity_passed"))
        score = float(pipe_out.get("aggregate_data_score") or 0.0)

        if not pit_ok or score < 50.0 or pipe_out.get("no_bet"):
            self.n_skipped_gate += 1
            return

        self.n_priced += 1
        prediction_id = pipe_out.get("prediction_id")
        if prediction_id is None:
            return

        # Confidence for RiskEngine (0–100).
        ens = pipe_out.get("ensemble")
        conf_raw = getattr(ens, "model_confidence_score", None) if ens else None
        if conf_raw is None:
            conf = score
        else:
            conf = float(conf_raw)
            if conf <= 1.0:
                conf *= 100.0

        candidates = build_1x2_candidates(pipe_out, match.bookmaker_odds)
        open_bets: list[dict[str, Any]] = []

        for cand in candidates:
            stake_res = self.risk.calculate_stake(
                probability=float(cand["probability"]),
                odds=float(cand["odds"]),
                data_score=score,
                model_confidence=conf,
                existing_match_bets=open_bets,
                market=str(cand.get("market_type") or "1X2"),
                selection=str(cand.get("selection") or ""),
            )
            frac = float(stake_res.get("final_stake") or 0.0)
            if frac <= 0.0:
                continue

            exposure = self.exposure.check_exposure(frac, league=match.league)
            if not exposure["allowed"]:
                continue

            accepted = await self.exposure.accept_bet(
                frac,
                league=match.league,
                market_type=str(cand.get("market_type") or "1X2"),
                selection=str(cand["selection"]),
                taken_odds=float(cand["odds"]),
                prediction_id=prediction_id,
                stake_result=stake_res,
                bankroll=self.bankroll,
                session=session,
                persist=True,
            )
            if not accepted.get("accepted"):
                continue

            record = accepted.get("record") or {}
            bet_id = record.get("bet_id") or uuid4()
            stake_amt = float(record.get("final_stake_amount") or frac * self.bankroll)
            sel_key = _selection_to_hda(str(cand["selection"]))
            # CLV reference = de-vigged real closing price (None ⇒ no CLV).
            closing = None
            fair_close = _fair_closing(match.closing_odds)
            if sel_key and fair_close:
                closing = float(fair_close.get(sel_key) or 0) or None

            paper = _PaperBet(
                bet_id=bet_id if isinstance(bet_id, UUID) else UUID(str(bet_id)),
                match_id=match.canonical_match_id,
                prediction_id=prediction_id
                if isinstance(prediction_id, UUID)
                else UUID(str(prediction_id)),
                market_type=str(cand.get("market_type") or "1X2"),
                selection=str(cand["selection"]),
                taken_odds=float(cand["odds"]),
                closing_odds=closing,
                stake=stake_amt,
                probability=float(cand["probability"]),
                model_tier=str(pipe_out.get("model_tier") or ""),
                kickoff_utc=_ensure_aware(match.kickoff_utc),
            )
            self.paper_bets.append(paper)
            open_bets.append(
                {
                    "market_type": paper.market_type,
                    "selection": paper.selection,
                    "bet_id": paper.bet_id,
                }
            )

            # Binary calibration sample for the selection.
            if match.ft_home_goals is not None and match.ft_away_goals is not None:
                y = _outcome_hit(
                    paper.selection,
                    int(match.ft_home_goals),
                    int(match.ft_away_goals),
                )
                if y is not None:
                    self.y_true.append(y)
                    self.y_prob.append(paper.probability)

        await session.flush()

    async def _settle(self, session: AsyncSession, as_of: datetime) -> None:
        """Settle pending DB bets via SettlementEngineV2; sync in-memory ledger."""
        closing_lookup = self._closing_lookup()
        engine = SettlementEngineV2(session, closing_odds_lookup=closing_lookup)
        summary = await engine.settle_pending_bets(as_of)

        # Sync paper ledger from DB settlement details.
        by_id = {b.bet_id: b for b in self.paper_bets if b.status == "PENDING"}
        for detail in summary.get("details") or []:
            bid = detail.get("bet_id")
            paper = by_id.get(bid) if isinstance(bid, UUID) else by_id.get(UUID(str(bid)))
            if paper is None:
                # Still record orphan detail for metrics.
                row = {
                    "bet_id": bid,
                    "status": detail.get("status"),
                    "pnl": float(detail.get("pnl") or 0.0),
                    "stake": 0.0,
                    "clv_value": detail.get("clv_value"),
                }
                self.settled.append(row)
                continue

            paper.status = str(detail.get("status") or paper.status)
            paper.pnl = float(detail.get("pnl") or 0.0)
            clv = detail.get("clv_value")
            if clv is None and paper.closing_odds and paper.taken_odds > 0:
                clv = QuantMetrics.closing_line_value(paper.taken_odds, paper.closing_odds)
            paper.clv_value = (
                None
                if clv is None or (isinstance(clv, float) and math.isnan(clv))
                else float(clv)
            )
            self.bankroll += paper.pnl
            self.equity.append(self.bankroll)
            self.settled.append(
                {
                    "bet_id": paper.bet_id,
                    "status": paper.status,
                    "pnl": paper.pnl,
                    "stake": paper.stake,
                    "clv_value": paper.clv_value,
                    "selection": paper.selection,
                    "taken_odds": paper.taken_odds,
                    "model_tier": paper.model_tier,
                }
            )

        # Fallback settle for any pending paper bets not linked in DB.
        for paper in self.paper_bets:
            if paper.status != "PENDING":
                continue
            if paper.kickoff_utc > _ensure_aware(as_of):
                continue
            match_row = await session.get(CanonicalMatch, paper.match_id)
            if match_row is None or match_row.ft_home_goals is None:
                continue
            status, pnl, _ = settle_market(
                market_type=paper.market_type,
                selection=paper.selection,
                taken_odds=paper.taken_odds,
                stake=paper.stake,
                ft_home=int(match_row.ft_home_goals),
                ft_away=int(match_row.ft_away_goals or 0),
            )
            paper.status = status
            paper.pnl = float(pnl)
            if paper.closing_odds:
                clv = QuantMetrics.closing_line_value(paper.taken_odds, paper.closing_odds)
                paper.clv_value = None if math.isnan(clv) else clv
            self.bankroll += paper.pnl
            self.equity.append(self.bankroll)
            self.settled.append(
                {
                    "bet_id": paper.bet_id,
                    "status": paper.status,
                    "pnl": paper.pnl,
                    "stake": paper.stake,
                    "clv_value": paper.clv_value,
                    "selection": paper.selection,
                    "taken_odds": paper.taken_odds,
                    "model_tier": paper.model_tier,
                }
            )

        await session.flush()

    def _closing_lookup(self):
        by_match: dict[UUID, dict[str, float]] = {}
        # Rebuild from paper bets' closing_odds keyed by match — also store
        # selection-level via H/D/A from original matches is better; use paper.
        for b in self.paper_bets:
            if b.closing_odds is None:
                continue
            # We only have per-selection closing on the bet; SettlementEngine
            # asks by market/selection — return that odds when selection matches.
            by_match.setdefault(b.match_id, {})[b.selection.lower()] = b.closing_odds

        # Also keep H/D/A maps from fixtures via settled path: inject from paper
        # taken + known close on HDA keys.
        hda_by_match: dict[UUID, dict[str, float]] = {}
        for b in self.paper_bets:
            if b.closing_odds is None:
                continue
            key = _selection_to_hda(b.selection)
            if key:
                hda_by_match.setdefault(b.match_id, {})[key] = b.closing_odds

        def _lookup(
            match_id: UUID, market: str, selection: str, kickoff: datetime
        ) -> float | None:
            _ = (market, kickoff)
            sel_l = selection.lower()
            if match_id in by_match and sel_l in by_match[match_id]:
                return by_match[match_id][sel_l]
            hda = hda_by_match.get(match_id) or {}
            key = _selection_to_hda(selection)
            if key and key in hda:
                return hda[key]
            return None

        return _lookup


# ---------------------------------------------------------------------------
# Orchestration / CLI
# ---------------------------------------------------------------------------


async def run_backtest(config: BacktestConfig) -> QuantReport:
    """Load matches (fixtures → DB, or synthetic on request) and replay.

    Raises :class:`BacktestDataError` when the DB is unreachable or has no
    matches in range — synthetic data is used **only** with
    ``config.synthetic=True`` and is then reported as ``mode="synthetic"``.
    """
    start = datetime(
        config.start_date.year,
        config.start_date.month,
        config.start_date.day,
        tzinfo=timezone.utc,
    )
    end = datetime(
        config.end_date.year,
        config.end_date.month,
        config.end_date.day,
        tzinfo=timezone.utc,
    ) + timedelta(days=1)

    matches: list[BacktestMatch] = []
    history: list[BacktestMatch] = []
    source: str
    load_stats: dict[str, int] = {}

    if config.fixtures_path is not None:
        if not config.fixtures_path.exists():
            raise BacktestDataError(f"Fixtures file not found: {config.fixtures_path}")
        loaded = load_matches_from_fixtures(config.fixtures_path)
        history = [m for m in loaded if _ensure_aware(m.kickoff_utc) < start]
        matches = loaded
        source = "fixtures"
    elif config.synthetic:
        matches = generate_synthetic_matches(
            n=DRY_RUN_MATCH_COUNT,
            end=end - timedelta(seconds=1),
            seed=config.seed,
        )
        source = "synthetic"
        logger.info("Using %d synthetic matches (explicitly requested)", len(matches))
    else:
        hist_start = datetime(
            start.year - max(1, int(config.train_window_years)), 1, 1, tzinfo=timezone.utc
        )
        try:
            from src.db.session import async_session_factory, create_async_engine_from_url

            eng = create_async_engine_from_url(config.database_url)
            factory = async_session_factory(eng)
            async with factory() as session:
                matches = await load_matches_from_db(
                    session,
                    start=start,
                    end=end,
                    stats=load_stats,
                    entry_odds_family=config.entry_odds_family,
                    allow_closing_as_entry=config.allow_closing_as_entry,
                )
                history = await load_matches_from_db(
                    session,
                    start=hist_start,
                    end=start,
                    entry_odds_family=config.entry_odds_family,
                    allow_closing_as_entry=config.allow_closing_as_entry,
                )
            await eng.dispose()
        except Exception as exc:  # noqa: BLE001
            raise BacktestDataError(
                f"DB match load failed ({exc}); refusing to fall back to synthetic "
                "data. Fix the DB / --database-url, or pass --synthetic."
            ) from exc
        if not matches:
            raise BacktestDataError(
                f"No canonical matches in DB for [{start.date()}, {end.date()}); "
                "refusing to fall back to synthetic data. Import history first "
                "or pass --synthetic."
            )
        source = "database"
        logger.info(
            "DB load: %s (history=%d)",
            ", ".join(f"{k}={v}" for k, v in sorted(load_stats.items())),
            len(history),
        )

    # Filter to date window (earlier fixture rows stay as training history).
    matches = [
        m
        for m in matches
        if start <= _ensure_aware(m.kickoff_utc) < end
    ]
    matches.sort(key=lambda m: m.kickoff_utc)

    dry_run_cut: list[BacktestMatch] = []
    if config.dry_run and len(matches) > DRY_RUN_MATCH_COUNT:
        dry_run_cut = matches[:-DRY_RUN_MATCH_COUNT]
        matches = matches[-DRY_RUN_MATCH_COUNT:]

    if source == "synthetic":
        mode = "synthetic"
    elif config.dry_run:
        mode = "dry-run"
    else:
        mode = "live"

    logger.info(
        "Backtest: %d matches · source=%s · mode=%s · bankroll=%.2f",
        len(matches),
        source,
        mode,
        config.initial_bankroll,
    )

    bt = WalkForwardBacktester(config)
    report = await bt.run(
        matches,
        history=history + dry_run_cut,
        mode=mode,
        source=source,
    )
    return report


def parse_args(argv: Sequence[str] | None = None) -> BacktestConfig:
    p = argparse.ArgumentParser(
        description="Walk-Forward Backtesting Master (quant replay)"
    )
    p.add_argument("--start-date", type=str, default="2024-01-01")
    p.add_argument("--end-date", type=str, default="2026-09-01")
    p.add_argument("--initial-bankroll", type=float, default=10_000.0)
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Replay the last ~50 matches only",
    )
    p.add_argument(
        "--synthetic",
        action="store_true",
        help="Use generated synthetic fixtures (report mode=synthetic)",
    )
    p.add_argument("--train-window-years", type=int, default=3)
    p.add_argument("--test-window-months", type=int, default=3)
    p.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    p.add_argument("--fixtures", type=str, default=None, help="JSON fixtures path")
    p.add_argument("--database-url", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("-v", "--verbose", action="store_true")
    ns = p.parse_args(list(argv) if argv is not None else None)

    if ns.verbose:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    else:
        logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
        # Synthetic X-tier fixtures intentionally trip PIT hard gates (CRITICAL).
        _quiet = logging.CRITICAL + 1
        logging.getLogger("src.data.pit_engine").setLevel(_quiet)
        logging.getLogger("src.data.quality_monitor").setLevel(_quiet)
        logging.getLogger("src.models.ensemble_v2").setLevel(_quiet)
        logging.getLogger("src.pipeline.run_pipeline").setLevel(_quiet)

    return BacktestConfig(
        start_date=_parse_date(ns.start_date),
        end_date=_parse_date(ns.end_date),
        initial_bankroll=float(ns.initial_bankroll),
        dry_run=bool(ns.dry_run),
        concurrency=max(1, int(ns.concurrency)),
        fixtures_path=Path(ns.fixtures) if ns.fixtures else None,
        database_url=ns.database_url,
        seed=int(ns.seed),
        synthetic=bool(ns.synthetic),
        train_window_years=max(1, int(ns.train_window_years)),
        test_window_months=max(1, int(ns.test_window_months)),
    )


def main(argv: Sequence[str] | None = None) -> int:
    config = parse_args(argv)
    try:
        report = asyncio.run(run_backtest(config))
    except BacktestDataError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    text = render_quant_report(report)
    try:
        print(text)
    except UnicodeEncodeError:
        enc = getattr(sys.stdout, "encoding", None) or "utf-8"
        sys.stdout.buffer.write(text.encode(enc, errors="replace"))
        sys.stdout.buffer.write(b"\n")
    return 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ensure_aware(ts: datetime) -> datetime:
    if ts.tzinfo is None:
        return ts.replace(tzinfo=timezone.utc)
    return ts


def _parse_date(s: str) -> date:
    return date.fromisoformat(s.strip())


def _parse_dt(s: str | datetime) -> datetime:
    if isinstance(s, datetime):
        return _ensure_aware(s)
    raw = str(s).strip().replace("Z", "+00:00")
    dt = datetime.fromisoformat(raw)
    return _ensure_aware(dt)


def _fair_closing(closing: Mapping[str, float] | None) -> dict[str, float] | None:
    """De-vigged closing 1X2 prices (``None`` when no complete real close)."""
    valid = _valid_1x2(closing)
    if not valid:
        return None
    return devig_odds(valid)


def _training_frame(matches: Sequence[BacktestMatch]) -> Any:
    """Finished matches → DataFrame for ``DixonColesModel.fit``."""
    import pandas as pd

    return pd.DataFrame(
        {
            "Date": [
                _ensure_aware(m.kickoff_utc).replace(tzinfo=None) for m in matches
            ],
            "HomeTeam": [m.home_team for m in matches],
            "AwayTeam": [m.away_team for m in matches],
            "FTHG": [int(m.ft_home_goals) for m in matches],  # type: ignore[arg-type]
            "FTAG": [int(m.ft_away_goals) for m in matches],  # type: ignore[arg-type]
        }
    )


def _group_by_day(
    matches: Sequence[BacktestMatch],
) -> list[tuple[date, list[BacktestMatch]]]:
    groups: dict[date, list[BacktestMatch]] = {}
    order: list[date] = []
    for m in matches:
        d = _ensure_aware(m.kickoff_utc).date()
        if d not in groups:
            groups[d] = []
            order.append(d)
        groups[d].append(m)
    return [(d, groups[d]) for d in order]


def _selection_to_hda(selection: str) -> str | None:
    s = selection.strip().lower()
    if s in {"home", "h", "1"}:
        return "H"
    if s in {"draw", "d", "x"}:
        return "D"
    if s in {"away", "a", "2"}:
        return "A"
    return None


def _outcome_hit(selection: str, ft_h: int, ft_a: int) -> int | None:
    key = _selection_to_hda(selection)
    if key is None:
        return None
    if ft_h > ft_a:
        result = "H"
    elif ft_h < ft_a:
        result = "A"
    else:
        result = "D"
    return 1 if key == result else 0


# Re-exports for package __init__
__all__ = [
    "BacktestConfig",
    "BacktestDataError",
    "BacktestMatch",
    "QuantReport",
    "WalkForwardBacktester",
    "build_quant_report",
    "generate_synthetic_matches",
    "render_quant_report",
    "run_backtest",
    "main",
]


if __name__ == "__main__":
    sys.exit(main())
