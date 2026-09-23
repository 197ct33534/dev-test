#!/usr/bin/env python3
"""Migrate legacy league SQLite DBs into ``data/global_matches.db``.

Imports matches from ``epl_matches.db``, ``uwcl_matches.db``, and any other
``*_matches.db`` under ``data/`` (excluding the global DB itself).

Usage
-----
::

    python scripts/migrate_to_global_db.py
    python scripts/migrate_to_global_db.py --dry-run
    python scripts/migrate_to_global_db.py --db data/global_matches.db

Idempotent: re-running upserts by ``match_id`` / canonical team name.
Legacy DBs are left untouched so Streamlit / scanner keep working as fallback.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow ``python scripts/migrate_to_global_db.py`` from repo root.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import DATA_DIR, LEAGUE_CONFIG, read_matches_from_db
from src.global_db import (
    DEFAULT_COMPETITIONS,
    GLOBAL_DB_PATH,
    connect_global_db,
    import_legacy_matches_df,
    upsert_competition,
)


def discover_source_dbs(data_dir: Path = DATA_DIR) -> list[tuple[str, Path]]:
    """Return ``(comp_id, path)`` pairs for legacy league databases."""
    known: dict[str, Path] = {}
    for code, cfg in LEAGUE_CONFIG.items():
        path = Path(cfg["db_path"])  # type: ignore[arg-type]
        if path.is_file():
            known[code.upper()] = path

    # Any other ``*_matches.db`` (e.g. future WSL) not already mapped.
    for path in sorted(data_dir.glob("*_matches.db")):
        if path.resolve() == Path(GLOBAL_DB_PATH).resolve():
            continue
        if path.name == "global_matches.db":
            continue
        stem = path.stem  # epl_matches → epl
        if stem.endswith("_matches"):
            stem = stem[: -len("_matches")]
        code = stem.upper()
        if code == "GLOBAL":
            continue
        if code not in known:
            known[code] = path
    return sorted(known.items(), key=lambda x: x[0])


def migrate(
    *,
    target: Path = GLOBAL_DB_PATH,
    sources: list[tuple[str, Path]] | None = None,
    dry_run: bool = False,
) -> dict[str, dict[str, int]]:
    """Run migration; returns per-comp upsert stats."""
    pairs = sources if sources is not None else discover_source_dbs()
    if not pairs:
        raise FileNotFoundError(
            f"No legacy *_matches.db found under {DATA_DIR}. "
            "Load league data first (Streamlit / load_league_data)."
        )

    # Ensure default competitions exist even on dry-run (in-memory check).
    weight_by_comp = {c: w for c, _, w in DEFAULT_COMPETITIONS}

    if dry_run:
        conn = connect_global_db(":memory:", init=True)
    else:
        conn = connect_global_db(target, init=True)

    results: dict[str, dict[str, int]] = {}
    try:
        for comp_id, path in pairs:
            # Register unknown comps with default weight 0.85 (cup-like) or 1.0.
            if comp_id not in weight_by_comp:
                default_w = 0.85 if "CL" in comp_id or "CUP" in comp_id else 1.0
                upsert_competition(conn, comp_id, comp_id, default_w)
                weight_by_comp[comp_id] = default_w
            else:
                # Re-seed known defaults (idempotent).
                name = next(
                    (n for c, n, _ in DEFAULT_COMPETITIONS if c == comp_id),
                    comp_id,
                )
                upsert_competition(conn, comp_id, name, weight_by_comp[comp_id])

            df = read_matches_from_db(path)
            stats = import_legacy_matches_df(
                conn, df, comp_id=comp_id, dry_run=dry_run
            )
            results[comp_id] = {**stats, "source_rows": int(len(df))}
            print(
                f"[{comp_id}] {path.name}: "
                f"source={len(df)} upserted={stats['matches_upserted']} "
                f"skipped={stats['skipped']} teams={stats['teams_touched']}"
                + (" (dry-run)" if dry_run else "")
            )
        if not dry_run:
            conn.commit()
    finally:
        conn.close()
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Migrate league SQLite DBs into global_matches.db"
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=GLOBAL_DB_PATH,
        help=f"Target global DB (default: {GLOBAL_DB_PATH})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse & count without writing to disk",
    )
    args = parser.parse_args(argv)

    print(f"Target: {args.db}" + (" [dry-run]" if args.dry_run else ""))
    sources = discover_source_dbs()
    if not sources:
        print("No source DBs found.", file=sys.stderr)
        return 1
    print("Sources:")
    for cid, path in sources:
        print(f"  - {cid}: {path}")

    results = migrate(target=args.db, sources=sources, dry_run=args.dry_run)
    total = sum(r["matches_upserted"] for r in results.values())
    print(f"Done. Total matches upserted: {total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
