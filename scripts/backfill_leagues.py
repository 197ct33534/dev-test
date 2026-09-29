"""CLI: backfill last ~3 seasons of major leagues into Quant Engine v2.

Examples
--------
    python scripts/backfill_leagues.py --dry-run
    python scripts/backfill_leagues.py --leagues EPL,LALIGA --seasons 2023,2024,2025
    python scripts/backfill_leagues.py --leagues EPL --seasons 2024

Football-Data.co.uk codes
-------------------------
    EPL            E0   mmz4281/{season}/E0.csv
    LALIGA         SP1  mmz4281/{season}/SP1.csv
    SERIE_A        I1   mmz4281/{season}/I1.csv
    BUNDESLIGA     D1   mmz4281/{season}/D1.csv
    LIGUE_1        F1   mmz4281/{season}/F1.csv
    EREDIVISIE     N1   mmz4281/{season}/N1.csv
    CHAMPIONSHIP   E1   mmz4281/{season}/E1.csv

Alternate / fallback URLs (not season-sliced the same way)
---------------------------------------------------------
    J_LEAGUE  → https://www.football-data.co.uk/new/JPN.csv
                (single multi-year file; --seasons documented for filtering intent)
    MLS       → https://www.football-data.co.uk/new/USA.csv
                (same pattern as JPN)
    UCL / Champions League
              → football-data.co.uk does **not** publish a CL CSV in mmz4281.
                Fallbacks: Flashscore / Fotmob via ``scripts/fetch_league.py``
                or a manual CSV path with ``BulkFootballDataImporter``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Sequence

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sqlalchemy import select  # noqa: E402

from src.data.bulk_importer import (  # noqa: E402
    COMPETITION_CODE_NAMES,
    BulkFootballDataImporter,
    football_data_new_league_url,
    football_data_season_url,
)
from src.db.schema_v2 import CanonicalCompetition, init_schema_v2_async  # noqa: E402
from src.db.session import async_session_factory, create_async_engine_from_url  # noqa: E402

logger = logging.getLogger(__name__)

DEFAULT_SEASONS = (2023, 2024, 2025)

_UCL_NOTES = (
    "No football-data.co.uk Champions League CSV. "
    "Fallback: Flashscore/Fotmob via scripts/fetch_league.py "
    "or import a local CSV with BulkFootballDataImporter."
)


@dataclass(frozen=True)
class LeagueSpec:
    """One backfill target."""

    key: str
    code: str
    name: str
    mode: str  # "season" | "new_file" | "unsupported"
    new_stem: Optional[str] = None
    notes: str = ""


LEAGUE_CATALOG: dict[str, LeagueSpec] = {
    "EPL": LeagueSpec("EPL", "E0", "Premier League", "season"),
    "LALIGA": LeagueSpec("LALIGA", "SP1", "La Liga", "season"),
    "LA_LIGA": LeagueSpec("LA_LIGA", "SP1", "La Liga", "season"),
    "SERIE_A": LeagueSpec("SERIE_A", "I1", "Serie A", "season"),
    "SERIEA": LeagueSpec("SERIEA", "I1", "Serie A", "season"),
    "BUNDESLIGA": LeagueSpec("BUNDESLIGA", "D1", "Bundesliga", "season"),
    "LIGUE_1": LeagueSpec("LIGUE_1", "F1", "Ligue 1", "season"),
    "EREDIVISIE": LeagueSpec("EREDIVISIE", "N1", "Eredivisie", "season"),
    "CHAMPIONSHIP": LeagueSpec("CHAMPIONSHIP", "E1", "Championship", "season"),
    "J_LEAGUE": LeagueSpec(
        "J_LEAGUE",
        "JPN",
        "J-League",
        "new_file",
        new_stem="JPN",
        notes="Uses /new/JPN.csv (not mmz4281).",
    ),
    "MLS": LeagueSpec(
        "MLS",
        "USA",
        "MLS",
        "new_file",
        new_stem="USA",
        notes="Uses /new/USA.csv (not mmz4281).",
    ),
    "UCL": LeagueSpec("UCL", "CL", "Champions League", "unsupported", notes=_UCL_NOTES),
    "CHAMPIONS_LEAGUE": LeagueSpec(
        "CHAMPIONS_LEAGUE",
        "CL",
        "Champions League",
        "unsupported",
        notes=_UCL_NOTES,
    ),
}

DEFAULT_LEAGUES: tuple[str, ...] = (
    "EPL",
    "LALIGA",
    "SERIE_A",
    "BUNDESLIGA",
    "LIGUE_1",
    "EREDIVISIE",
    "CHAMPIONSHIP",
    "J_LEAGUE",
    "MLS",
    "UCL",
)


def _parse_seasons(raw: Optional[str]) -> list[int]:
    """Parse ``2023,2024`` or ``2023-2026`` (exclusive end → starts 2023..2025)."""
    if not raw:
        return list(DEFAULT_SEASONS)
    parts = [p.strip() for p in raw.replace(";", ",").split(",") if p.strip()]
    out: list[int] = []
    for part in parts:
        if "-" in part and part.count("-") == 1:
            a, b = part.split("-", 1)
            start, end = int(a), int(b)
            if end < start:
                raise ValueError(f"Invalid season range: {part}")
            if end - start > 20:
                raise ValueError(f"Season range too wide: {part}")
            # 2023-2026 → seasons covering through 2026 calendar → starts 2023,2024,2025
            if end - start >= 2:
                out.extend(range(start, end))
            else:
                out.extend(range(start, end + 1))
        else:
            out.append(int(part))
    return sorted(set(out))


def _parse_leagues(raw: Optional[str]) -> list[str]:
    if not raw:
        return list(DEFAULT_LEAGUES)
    keys = [p.strip().upper() for p in raw.replace(";", ",").split(",") if p.strip()]
    unknown = [k for k in keys if k not in LEAGUE_CATALOG]
    if unknown:
        known = ", ".join(sorted(set(DEFAULT_LEAGUES) | {"LA_LIGA", "SERIEA", "CHAMPIONS_LEAGUE"}))
        raise SystemExit(f"Unknown league(s): {unknown}. Known: {known}")
    return keys


async def _ensure_seeded_competitions(
    session: Any, leagues: Sequence[LeagueSpec]
) -> None:
    """Pre-seed canonical competitions so fuzzy resolver has targets."""
    for spec in leagues:
        existing = (
            await session.execute(
                select(CanonicalCompetition).where(
                    CanonicalCompetition.code == spec.code
                )
            )
        ).scalar_one_or_none()
        if existing is not None:
            continue
        by_name = (
            await session.execute(
                select(CanonicalCompetition).where(
                    CanonicalCompetition.name == spec.name
                )
            )
        ).scalar_one_or_none()
        if by_name is not None:
            if not by_name.code:
                by_name.code = spec.code
            continue
        session.add(
            CanonicalCompetition(
                name=spec.name,
                code=spec.code,
                mapping_status="MAPPED",
                meta_json={"created_via": "backfill_leagues"},
            )
        )
    await session.flush()


def _plan_jobs(
    league_keys: Sequence[str],
    seasons: Sequence[int],
) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    for key in league_keys:
        spec = LEAGUE_CATALOG[key]
        if spec.mode == "unsupported":
            jobs.append(
                {
                    "league": key,
                    "code": spec.code,
                    "name": spec.name,
                    "status": "skipped_unsupported",
                    "notes": spec.notes,
                    "url": None,
                    "season": None,
                }
            )
            continue
        if spec.mode == "new_file":
            url = football_data_new_league_url(spec.new_stem or spec.code)
            jobs.append(
                {
                    "league": key,
                    "code": spec.code,
                    "name": spec.name,
                    "status": "pending",
                    "notes": spec.notes,
                    "url": url,
                    "season": "all_in_file",
                    "filter_years": list(seasons),
                }
            )
            continue
        for start in seasons:
            url = football_data_season_url(spec.code, int(start))
            jobs.append(
                {
                    "league": key,
                    "code": spec.code,
                    "name": spec.name,
                    "status": "pending",
                    "notes": spec.notes,
                    "url": url,
                    "season": int(start),
                }
            )
    return jobs


async def run_backfill(
    *,
    league_keys: Sequence[str],
    seasons: Sequence[int],
    dry_run: bool,
    database_url: Optional[str] = None,
) -> dict[str, Any]:
    jobs = _plan_jobs(league_keys, seasons)
    summary: dict[str, Any] = {
        "dry_run": dry_run,
        "seasons": list(seasons),
        "leagues": list(league_keys),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "jobs": [],
        "competition_code_names": dict(COMPETITION_CODE_NAMES),
    }

    if dry_run:
        for job in jobs:
            entry = {
                **job,
                "status": job["status"]
                if job["status"] != "pending"
                else "dry_run",
            }
            summary["jobs"].append(entry)
            logger.info(
                "[dry-run] %s %s → %s (%s)",
                job["league"],
                job.get("season"),
                job.get("url") or "N/A",
                entry["status"],
            )
        summary["finished_at"] = datetime.now(timezone.utc).isoformat()
        return summary

    engine = create_async_engine_from_url(database_url)
    await init_schema_v2_async(engine)
    factory = async_session_factory(engine)

    specs = [LEAGUE_CATALOG[k] for k in league_keys]
    async with factory() as session:
        await _ensure_seeded_competitions(session, specs)
        importer = BulkFootballDataImporter(session)

        for job in jobs:
            if job["status"] == "skipped_unsupported":
                summary["jobs"].append(job)
                logger.warning(
                    "Skip unsupported league %s: %s", job["league"], job["notes"]
                )
                continue

            url = job["url"]
            code = job["code"]
            try:
                stats = await importer.import_csv_season_data(url, code)
                entry = {
                    **job,
                    "status": "ok" if not stats.errors else "error",
                    "stats": stats.as_dict(),
                }
            except Exception as exc:  # noqa: BLE001
                logger.exception("Import failed for %s", url)
                entry = {**job, "status": "error", "error": str(exc)}
            summary["jobs"].append(entry)

        await session.commit()

    await engine.dispose()
    summary["finished_at"] = datetime.now(timezone.utc).isoformat()
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Backfill Football-Data.co.uk seasons into quant_engine_v2",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned URLs / skips without writing to the database",
    )
    parser.add_argument(
        "--seasons",
        type=str,
        default=None,
        help="Comma-separated season start years or a range "
        "(default: 2023,2024,2025). Example: 2023-2026 → 2023,2024,2025",
    )
    parser.add_argument(
        "--leagues",
        type=str,
        default=None,
        help="Comma-separated league keys "
        f"(default: {','.join(DEFAULT_LEAGUES)})",
    )
    parser.add_argument(
        "--database-url",
        type=str,
        default=None,
        help="Override DATABASE_URL "
        "(default: sqlite+aiosqlite:///./data/quant_engine_v2.db)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        dest="list_leagues",
        help="Print league catalog and exit",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="DEBUG logging",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.list_leagues:
        catalog = {
            k: {
                "code": v.code,
                "name": v.name,
                "mode": v.mode,
                "notes": v.notes,
            }
            for k, v in LEAGUE_CATALOG.items()
        }
        print(json.dumps(catalog, indent=2, ensure_ascii=False))
        return 0

    try:
        seasons = _parse_seasons(args.seasons)
        leagues = _parse_leagues(args.leagues)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    summary = asyncio.run(
        run_backfill(
            league_keys=leagues,
            seasons=seasons,
            dry_run=bool(args.dry_run),
            database_url=args.database_url,
        )
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False, default=str))
    errors = [j for j in summary["jobs"] if j.get("status") == "error"]
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
