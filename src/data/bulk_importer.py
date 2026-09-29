"""Bulk historical importer for Football-Data.co.uk CSVs into Quant Engine v2."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
from typing import Any, Optional
from urllib.request import Request, urlopen

import pandas as pd
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.data.competition_resolver import CompetitionResolver
from src.data.entity_resolver import EntityResolver, MAPPED
from src.db.schema_v2 import (
    CanonicalCompetition,
    CanonicalMatch,
    CompetitionAlias,
    RawDataLake,
)

logger = logging.getLogger(__name__)

SOURCE_TYPE = "football-data"
DEFAULT_BATCH_SIZE = 100
USER_AGENT = "score-quant-engine/1.0 (+bulk-importer; research)"

# football-data.co.uk division code → canonical display name.
COMPETITION_CODE_NAMES: dict[str, str] = {
    "E0": "Premier League",
    "E1": "Championship",
    "SP1": "La Liga",
    "I1": "Serie A",
    "D1": "Bundesliga",
    "F1": "Ligue 1",
    "N1": "Eredivisie",
    "JPN": "J-League",
    "USA": "MLS",
    "CL": "Champions League",
}


@dataclass
class ImportStats:
    """Counters returned by :meth:`BulkFootballDataImporter.import_csv_season_data`."""

    rows_read: int = 0
    rows_skipped: int = 0
    raw_inserted: int = 0
    matches_upserted: int = 0
    matches_updated: int = 0
    competition_id: Optional[str] = None
    errors: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "rows_read": self.rows_read,
            "rows_skipped": self.rows_skipped,
            "raw_inserted": self.raw_inserted,
            "matches_upserted": self.matches_upserted,
            "matches_updated": self.matches_updated,
            "competition_id": self.competition_id,
            "errors": list(self.errors),
        }


class BulkFootballDataImporter:
    """Load Football-Data CSVs into ``raw_data_lake`` + ``canonical_matches``.

    Point-in-time safety: ``RawDataLake.observed_at`` is set to kickoff UTC so
    historical rows never leak future information relative to kickoff.
    """

    def __init__(
        self,
        session: AsyncSession,
        *,
        entity_resolver: Optional[EntityResolver] = None,
        competition_resolver: Optional[CompetitionResolver] = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> None:
        self.session = session
        self.entity_resolver = entity_resolver or EntityResolver(session)
        self.competition_resolver = competition_resolver or CompetitionResolver(
            session, entity_resolver=self.entity_resolver
        )
        self.batch_size = max(1, int(batch_size))

    async def import_csv_season_data(
        self,
        csv_url_or_file_path: str,
        competition_code: str,
    ) -> ImportStats:
        """Import one season CSV (URL or local path) for ``competition_code``.

        Parameters
        ----------
        csv_url_or_file_path:
            ``https://...`` URL or filesystem path to a Football-Data CSV.
        competition_code:
            Division / league code (e.g. ``E0``, ``SP1``, ``JPN``).

        Returns
        -------
        ImportStats
            Insert / skip counters (also logged).
        """
        stats = ImportStats()
        code = competition_code.strip().upper()
        try:
            df = self._load_csv(csv_url_or_file_path)
        except Exception as exc:  # noqa: BLE001
            msg = f"Failed to load CSV {csv_url_or_file_path!r}: {exc}"
            logger.error(msg)
            stats.errors.append(msg)
            return stats

        stats.rows_read = int(len(df))
        if df.empty:
            logger.warning("Empty CSV: %s", csv_url_or_file_path)
            return stats

        competition_id = await self._ensure_competition(code, df)
        if competition_id is None:
            msg = f"Could not resolve competition for code={code!r}"
            logger.error(msg)
            stats.errors.append(msg)
            return stats
        stats.competition_id = str(competition_id)

        pending_raw: list[RawDataLake] = []
        pending_flush = 0

        for idx, row in df.iterrows():
            try:
                parsed = self._parse_row(row)
            except Exception as exc:  # noqa: BLE001
                stats.rows_skipped += 1
                logger.warning("Skip row %s: parse error: %s", idx, exc)
                continue

            if parsed is None:
                stats.rows_skipped += 1
                continue

            kickoff, home_name, away_name, payload = parsed
            try:
                home_id = await self.entity_resolver.get_or_create_canonical_team(
                    self.session, home_name, SOURCE_TYPE, None
                )
                away_id = await self.entity_resolver.get_or_create_canonical_team(
                    self.session, away_name, SOURCE_TYPE, None
                )
            except Exception as exc:  # noqa: BLE001
                stats.rows_skipped += 1
                logger.warning(
                    "Skip row %s (%s vs %s): team resolve failed: %s",
                    idx,
                    home_name,
                    away_name,
                    exc,
                )
                continue

            season_id = self._season_id(kickoff, payload)
            match, created = await self._upsert_finished_match(
                competition_id=competition_id,
                home_team_id=home_id,
                away_team_id=away_id,
                kickoff_utc=kickoff,
                season_id=season_id,
                payload=payload,
            )
            if created:
                stats.matches_upserted += 1
            else:
                stats.matches_updated += 1

            raw = RawDataLake(
                source=SOURCE_TYPE,
                entity_type="match",
                source_entity_id=self._source_entity_id(code, home_name, away_name, kickoff),
                payload=payload,
                source_timestamp=kickoff,
                observed_at=kickoff,  # PIT-safe: no post-kickoff leak for history
                canonical_match_id=match.canonical_match_id,
            )
            pending_raw.append(raw)
            pending_flush += 1
            stats.raw_inserted += 1

            if pending_flush >= self.batch_size:
                self.session.add_all(pending_raw)
                await self.session.flush()
                pending_raw.clear()
                pending_flush = 0

        if pending_raw:
            self.session.add_all(pending_raw)
            await self.session.flush()

        logger.info(
            "Imported %s code=%s read=%d skip=%d raw=%d upsert=%d update=%d",
            csv_url_or_file_path,
            code,
            stats.rows_read,
            stats.rows_skipped,
            stats.raw_inserted,
            stats.matches_upserted,
            stats.matches_updated,
        )
        return stats

    async def _ensure_competition(
        self,
        code: str,
        df: pd.DataFrame,
    ) -> Optional[Any]:
        """Return canonical competition UUID for ``code``, creating if needed."""
        display = COMPETITION_CODE_NAMES.get(code, code)

        # Prefer existing row keyed by code.
        by_code = (
            await self.session.execute(
                select(CanonicalCompetition).where(CanonicalCompetition.code == code)
            )
        ).scalar_one_or_none()
        if by_code is not None:
            await self._ensure_code_alias(by_code.id, code)
            return by_code.id

        # Exact name match (avoids staging when we already know the label).
        by_name = (
            await self.session.execute(
                select(CanonicalCompetition).where(
                    CanonicalCompetition.name == display
                )
            )
        ).scalar_one_or_none()
        if by_name is not None:
            if not by_name.code:
                by_name.code = code
                await self.session.flush()
            await self._ensure_code_alias(by_name.id, code)
            return by_name.id

        # Fuzzy resolve against existing canonicals (may stage if no good match).
        has_any = (
            await self.session.execute(select(CanonicalCompetition.id).limit(1))
        ).scalar_one_or_none()
        if has_any is not None:
            resolved = await self.competition_resolver.resolve_competition(
                self.session, SOURCE_TYPE, display
            )
            if resolved is not None:
                await self._ensure_code_alias(resolved, code)
                comp = (
                    await self.session.execute(
                        select(CanonicalCompetition).where(
                            CanonicalCompetition.id == resolved
                        )
                    )
                ).scalar_one()
                if not comp.code:
                    comp.code = code
                    await self.session.flush()
                return resolved

        # Seed a MAPPED canonical competition for known codes, then alias.
        comp = CanonicalCompetition(
            name=display,
            code=code,
            mapping_status=MAPPED,
            meta_json={"created_via": "bulk_importer", "competition_code": code},
        )
        self.session.add(comp)
        await self.session.flush()
        await self._ensure_code_alias(comp.id, code)
        # Optional Div label as secondary alias (ignore if Div is just the code).
        if "Div" in df.columns:
            div_vals = (
                df["Div"].dropna().astype(str).str.strip().loc[lambda s: s != ""]
            )
            if not div_vals.empty:
                first_div = str(div_vals.iloc[0])
                if first_div.upper() != code and first_div != display:
                    await self.competition_resolver.resolve_competition(
                        self.session, SOURCE_TYPE, first_div
                    )
        logger.info("Seeded canonical competition %s (%s)", display, code)
        return comp.id

    async def _ensure_code_alias(self, canonical_id: Any, code: str) -> None:
        existing = await self.entity_resolver._find_competition_alias(
            self.session,
            source_type=SOURCE_TYPE,
            source_name=code,
            source_competition_id=code,
        )
        if existing is not None:
            return
        self.session.add(
            CompetitionAlias(
                canonical_competition_id=canonical_id,
                source_type=SOURCE_TYPE,
                source_competition_id=code,
                source_name=COMPETITION_CODE_NAMES.get(code, code),
                mapping_status=MAPPED,
            )
        )
        await self.session.flush()

    async def _upsert_finished_match(
        self,
        *,
        competition_id: Any,
        home_team_id: Any,
        away_team_id: Any,
        kickoff_utc: datetime,
        season_id: Optional[str],
        payload: dict[str, Any],
    ) -> tuple[CanonicalMatch, bool]:
        """Insert or update a finished match; return ``(match, created)``."""
        existing = (
            await self.session.execute(
                select(CanonicalMatch).where(
                    CanonicalMatch.competition_id == competition_id,
                    CanonicalMatch.home_team_id == home_team_id,
                    CanonicalMatch.away_team_id == away_team_id,
                    CanonicalMatch.kickoff_utc == kickoff_utc,
                )
            )
        ).scalar_one_or_none()

        ft_h = _as_optional_int(payload.get("FTHG"))
        ft_a = _as_optional_int(payload.get("FTAG"))
        ht_h = _as_optional_int(payload.get("HTHG"))
        ht_a = _as_optional_int(payload.get("HTAG"))
        status = "FINISHED" if ft_h is not None and ft_a is not None else "SCHEDULED"

        if existing is not None:
            existing.ft_home_goals = ft_h
            existing.ft_away_goals = ft_a
            existing.ht_home_goals = ht_h
            existing.ht_away_goals = ht_a
            existing.status = status
            if season_id and not existing.season_id:
                existing.season_id = season_id
            await self.session.flush()
            return existing, False

        match = CanonicalMatch(
            competition_id=competition_id,
            season_id=season_id,
            home_team_id=home_team_id,
            away_team_id=away_team_id,
            kickoff_utc=kickoff_utc,
            status=status,
            ft_home_goals=ft_h,
            ft_away_goals=ft_a,
            ht_home_goals=ht_h,
            ht_away_goals=ht_a,
        )
        self.session.add(match)
        await self.session.flush()
        return match, True

    def _load_csv(self, csv_url_or_file_path: str) -> pd.DataFrame:
        path_or_url = csv_url_or_file_path.strip()
        if path_or_url.lower().startswith(("http://", "https://")):
            text = _fetch_text(path_or_url)
            return pd.read_csv(StringIO(text))
        path = Path(path_or_url)
        if not path.is_file():
            raise FileNotFoundError(f"CSV not found: {path}")
        return pd.read_csv(path)

    def _parse_row(
        self, row: pd.Series
    ) -> Optional[tuple[datetime, str, str, dict[str, Any]]]:
        """Parse one CSV row → ``(kickoff_utc, home, away, payload)`` or ``None``."""
        home = _as_str(row.get("HomeTeam"))
        away = _as_str(row.get("AwayTeam"))
        if not home or not away:
            logger.warning("Skip row missing HomeTeam/AwayTeam")
            return None

        kickoff = _parse_kickoff(row.get("Date"), row.get("Time") if "Time" in row.index else None)
        if kickoff is None:
            logger.warning("Skip row %s vs %s: bad Date/Time", home, away)
            return None

        # Require FT goals for finished historical import; skip incomplete.
        fthg = _as_optional_int(row.get("FTHG") if "FTHG" in row.index else None)
        ftag = _as_optional_int(row.get("FTAG") if "FTAG" in row.index else None)
        if fthg is None or ftag is None:
            logger.warning("Skip row %s vs %s: missing FTHG/FTAG", home, away)
            return None

        payload = _row_to_payload(row)
        payload["FTHG"] = fthg
        payload["FTAG"] = ftag
        payload["HomeTeam"] = home
        payload["AwayTeam"] = away
        payload["kickoff_utc"] = kickoff.isoformat()
        return kickoff, home, away, payload

    @staticmethod
    def _season_id(kickoff: datetime, payload: dict[str, Any]) -> Optional[str]:
        if payload.get("Season"):
            return str(payload["Season"])
        # European-style Aug–May cut-over.
        year = kickoff.year
        start = year if kickoff.month >= 7 else year - 1
        return f"{start}/{str(start + 1)[-2:]}"

    @staticmethod
    def _source_entity_id(
        code: str, home: str, away: str, kickoff: datetime
    ) -> str:
        day = kickoff.strftime("%Y%m%d")
        return f"{code}:{day}:{home}:{away}"


def _fetch_text(url: str, timeout: float = 45.0) -> str:
    request = Request(url, headers={"User-Agent": USER_AGENT, "Accept": "text/csv,*/*"})
    with urlopen(request, timeout=timeout) as response:
        charset = response.headers.get_content_charset() or "utf-8"
        return response.read().decode(charset, errors="replace")


def _as_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return None
    return text


def _as_optional_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null", ""}:
        return None
    try:
        return int(float(text))
    except (TypeError, ValueError):
        return None


def _parse_kickoff(date_val: Any, time_val: Any) -> Optional[datetime]:
    """Parse football-data ``Date`` (+ optional ``Time``) as UTC-naive → aware UTC.

    Football-Data dates are local kickoff dates without a reliable TZ; we store
    them as UTC wall-clock (common for historical FD imports).
    """
    date_str = _as_str(date_val)
    if not date_str:
        return None
    parsed_date = pd.to_datetime(date_str, dayfirst=True, format="mixed", errors="coerce")
    if pd.isna(parsed_date):
        return None

    hour, minute = 15, 0  # FD often omits Time; midday-ish default
    time_str = _as_str(time_val)
    if time_str:
        try:
            parts = time_str.replace(".", ":").split(":")
            hour = int(parts[0])
            minute = int(parts[1]) if len(parts) > 1 else 0
        except (TypeError, ValueError, IndexError):
            logger.debug("Could not parse Time %r; using default 15:00", time_str)

    return datetime(
        int(parsed_date.year),
        int(parsed_date.month),
        int(parsed_date.day),
        hour,
        minute,
        tzinfo=timezone.utc,
    )


def _row_to_payload(row: pd.Series) -> dict[str, Any]:
    """Convert a pandas row to a JSON-serialisable dict (skip NaNs)."""
    out: dict[str, Any] = {}
    for key, value in row.items():
        key_str = str(key)
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            continue
        if pd.isna(value):
            continue
        if hasattr(value, "item"):
            try:
                value = value.item()
            except (ValueError, AttributeError):
                pass
        if isinstance(value, (datetime, pd.Timestamp)):
            ts = pd.Timestamp(value)
            if ts.tzinfo is None:
                ts = ts.tz_localize("UTC")
            out[key_str] = ts.isoformat()
        elif isinstance(value, (str, int, float, bool)):
            out[key_str] = value
        else:
            out[key_str] = str(value)
    return out


def football_data_season_url(competition_code: str, season_start: int) -> str:
    """Build the standard ``mmz4281/{yy}{yy+1}/{code}.csv`` URL."""
    start = season_start % 100
    end = (season_start + 1) % 100
    season_code = f"{start:02d}{end:02d}"
    code = competition_code.strip().upper()
    return f"https://www.football-data.co.uk/mmz4281/{season_code}/{code}.csv"


def football_data_new_league_url(file_stem: str) -> str:
    """Build alternate ``/new/{STEM}.csv`` URL (JPN, USA, …)."""
    stem = file_stem.strip().upper()
    return f"https://www.football-data.co.uk/new/{stem}.csv"
