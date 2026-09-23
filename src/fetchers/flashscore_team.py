"""Flashscore Team Results feeds (multi-competition rest / fatigue).

URL pattern
-----------
``https://www.flashscore.com/team/{slug}/{team_hash}/results/``

Example: Vissel Kobe → ``/team/vissel-kobe/698tGI9q/results/``.
Both slug and hash are required (wrong slug → HTTP 404).

The SSR page embeds ``cjs.initialFeeds["results"]`` (full recent list) and
``summary-results`` (short slice). Competition headers (``ZA`` / ``ZK``)
separate J1, Emperor's Cup, AFC CL, friendlies, etc. so rest_days can use
every competitive match, not just one cup DB.
"""

from __future__ import annotations

import logging
import re
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.error import HTTPError, URLError

import pandas as pd

from src.data_loader import (
    FLASHSCORE_FSIGN,
    _fetch_text,
    _flashscore_extract_initial_feed,
    _flashscore_parse_fields,
)
from src.fetchers.flashscore_league import (
    EMPERORS_CUP_TEAM_ALIASES,
    resolve_league_team_name,
)

logger = logging.getLogger(__name__)

FLASHSCORE_BASE = "https://www.flashscore.com"
DEFAULT_N_MATCHES = 10
DEFAULT_RETRIES = 3
DEFAULT_TIMEOUT_S = 45.0
DEFAULT_RETRY_DELAY_S = 1.0
# Lazy Top-20 sync: short timeout, few workers, 24h feed TTL.
FAST_FEED_TIMEOUT_S = 3.0
FAST_FEED_MAX_WORKERS = 3
FAST_FEED_MAX_TEAMS = 10
FEED_STALE_HOURS = 24.0

# Fallback when competition_name cannot be mapped to a real registry code.
GENERIC_TEAM_FEED_COMP_ID = "FLASH_TEAM"

# Built-in hash ↔ JP_* map (overridden / extended by config/leagues.json).
DEFAULT_TEAM_HASHES: dict[str, dict[str, str]] = {
    "JP_VISSEL_KOBE": {
        "hash": "698tGI9q",
        "slug": "vissel-kobe",
        "name": "Vissel Kobe",
    },
    "JP_SAGAN_TOSU": {
        "hash": "nsRRyAda",
        "slug": "sagan-tosu",
        "name": "Sagan Tosu",
    },
    "JP_MACHIDA": {
        "hash": "CUSC1dab",
        "slug": "machida-zelvia",
        "name": "Machida Zelvia",
    },
    "JP_TOCHIGI_CITY": {
        "hash": "4MvNs2n5",
        "slug": "tochigi-city",
        "name": "Tochigi City",
    },
}

# Flashscore competition labels → stable comp_id (prefer real codes).
_COMP_NAME_RULES: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\bj1\b", re.I), "J1"),
    (re.compile(r"\bj2\b", re.I), "J2"),
    (re.compile(r"\bj3\b", re.I), "J3"),
    (re.compile(r"emperor", re.I), "EMPERORS_CUP"),
    (re.compile(r"afc\s*champions|asian\s*champions|\bacl\b", re.I), "ACL"),
    (re.compile(r"j\.?\s*league\s*cup|levain", re.I), "J_LEAGUE_CUP"),
    (re.compile(r"club\s*friendly|friendly", re.I), "FRIENDLY"),
    (re.compile(r"\blaliga\b|la\s*liga", re.I), "LALIGA"),
    (re.compile(r"premier\s*league", re.I), "EPL"),
    (re.compile(r"champions\s*league\s*women|uwcl|women.?s\s*champions", re.I), "UWCL"),
    (re.compile(r"women.?s\s*super\s*league|\bwsl\b", re.I), "WSL"),
)

_COUNTRY_SUFFIX_RE = re.compile(r"\s*\([A-Za-z]{2,4}\)\s*$")
_SLUG_RE = re.compile(r"[^a-z0-9]+")


def _fetch_text_retry(
    url: str,
    *,
    retries: int = DEFAULT_RETRIES,
    timeout: float = DEFAULT_TIMEOUT_S,
) -> str:
    last_exc: Exception | None = None
    for attempt in range(max(1, retries)):
        try:
            return _fetch_text(
                url,
                timeout=timeout,
                referer="https://www.flashscore.com/",
                extra_headers={
                    "Accept": "text/html,*/*",
                    "x-fsign": FLASHSCORE_FSIGN,
                    "X-Fsign": FLASHSCORE_FSIGN,
                },
            )
        except (HTTPError, URLError, TimeoutError, RuntimeError, OSError) as exc:
            last_exc = exc
            if attempt + 1 < retries:
                time.sleep(DEFAULT_RETRY_DELAY_S * (attempt + 1))
    raise RuntimeError(f"Failed to fetch {url}: {last_exc}") from last_exc


def _load_team_hashes_from_config() -> dict[str, dict[str, str]]:
    """Merge ``config/leagues.json`` ``_team_hashes`` over :data:`DEFAULT_TEAM_HASHES`.

    Note: :func:`src.league_registry.load_leagues_json` strips ``_*`` meta keys,
    so we read the raw JSON file here.
    """
    import json

    out = {k: dict(v) for k, v in DEFAULT_TEAM_HASHES.items()}
    try:
        from src.league_registry import LEAGUES_JSON_PATH

        if not LEAGUES_JSON_PATH.is_file():
            return out
        with LEAGUES_JSON_PATH.open(encoding="utf-8") as fh:
            raw = json.load(fh)
    except Exception:  # noqa: BLE001
        return out
    extra = raw.get("_team_hashes") if isinstance(raw, dict) else None
    if not isinstance(extra, dict):
        return out
    for team_id, meta in extra.items():
        tid = str(team_id).strip().upper()
        if not tid or not isinstance(meta, dict):
            continue
        h = str(meta.get("hash") or meta.get("team_hash") or "").strip()
        if not h:
            continue
        entry = dict(out.get(tid) or {})
        entry["hash"] = h
        if meta.get("slug"):
            entry["slug"] = str(meta["slug"]).strip().strip("/")
        if meta.get("name"):
            entry["name"] = str(meta["name"]).strip()
        out[tid] = entry
    return out


def team_hash_registry() -> dict[str, dict[str, str]]:
    """Return ``{JP_*: {hash, slug, name}}`` (config + built-ins)."""
    return _load_team_hashes_from_config()


def team_id_for_hash(team_hash: str, *, db_path: Path | str | None = None) -> str | None:
    """Map Flashscore team hash → stable ``JP_*`` / ``ES_*`` code."""
    h = str(team_hash or "").strip()
    if not h:
        return None
    try:
        from src.global_db import (
            GLOBAL_DB_PATH,
            connect_global_db,
            find_team_id_by_flashscore_hash,
        )

        path = Path(db_path) if db_path is not None else GLOBAL_DB_PATH
        if str(path) == ":memory:" or Path(path).is_file():
            conn = connect_global_db(path, init=True)
            try:
                tid_int = find_team_id_by_flashscore_hash(conn, h)
                if tid_int is not None:
                    row = conn.execute(
                        "SELECT canonical_name FROM teams WHERE team_id = ?",
                        (tid_int,),
                    ).fetchone()
                    if row:
                        return str(row["canonical_name"])
            finally:
                conn.close()
    except Exception:  # noqa: BLE001
        pass
    for tid, meta in team_hash_registry().items():
        if str(meta.get("hash") or "") == h:
            return tid
    return None


def hash_for_team_id(team_id: str) -> str | None:
    """Map stable ``JP_*`` code → Flashscore team hash."""
    meta = resolve_team_hash(team_id)
    if not meta:
        return None
    h = str(meta.get("hash") or "").strip()
    return h or None


def slug_for_team_id(team_id: str) -> str | None:
    meta = resolve_team_hash(team_id)
    if not meta:
        return None
    slug = str(meta.get("slug") or "").strip().strip("/")
    return slug or None


def resolve_team_hash(
    team_id: str,
    *,
    db_path: Path | str | None = None,
    prefer_db: bool = True,
) -> dict[str, str] | None:
    """Resolve ``JP_*`` / ``ES_*`` / display alias → ``{hash, slug, name}``.

    Lookup order
    ------------
    1. ``teams.flashscore_hash`` in ``global_matches.db`` (preferred)
    2. ``config/leagues.json`` ``_team_hashes`` + :data:`DEFAULT_TEAM_HASHES`
       (last-resort fallback for known JP_* seeds)

    Accepts stable codes (``JP_MACHIDA``, ``ES_BARCELONA``) and common display
    names. Returns ``None`` when no Flashscore hash is known.
    """
    raw = str(team_id or "").strip()
    if not raw:
        return None
    tid = raw.upper()

    if prefer_db:
        db_meta = _resolve_team_hash_from_db(raw, db_path=db_path)
        if db_meta is not None:
            return db_meta

    registry = team_hash_registry()
    meta = registry.get(tid)
    if meta and str(meta.get("hash") or "").strip():
        return {
            "hash": str(meta["hash"]).strip(),
            "slug": str(meta.get("slug") or "").strip().strip("/"),
            "name": str(meta.get("name") or tid).strip(),
            "team_id": tid,
            "source": "static",
        }
    # Display name / alias → JP_* then re-lookup.
    try:
        mapped = resolve_league_team_name(raw, "EMPERORS_CUP")
    except Exception:  # noqa: BLE001
        mapped = raw
    mapped_u = str(mapped or "").strip().upper()
    if mapped_u and mapped_u != tid:
        if prefer_db:
            db_meta2 = _resolve_team_hash_from_db(mapped_u, db_path=db_path)
            if db_meta2 is not None:
                return db_meta2
        meta2 = registry.get(mapped_u)
        if meta2 and str(meta2.get("hash") or "").strip():
            return {
                "hash": str(meta2["hash"]).strip(),
                "slug": str(meta2.get("slug") or "").strip().strip("/"),
                "name": str(meta2.get("name") or mapped_u).strip(),
                "team_id": mapped_u,
                "source": "static",
            }
    # Case-insensitive name match against registry display names.
    needle = raw.casefold()
    for code, entry in registry.items():
        name = str(entry.get("name") or "").strip()
        if name and name.casefold() == needle and str(entry.get("hash") or "").strip():
            return {
                "hash": str(entry["hash"]).strip(),
                "slug": str(entry.get("slug") or "").strip().strip("/"),
                "name": name,
                "team_id": code,
                "source": "static",
            }
    # Try other league alias tables (LaLiga ES_* etc.) against DB only once more.
    if prefer_db:
        for league in ("LALIGA", "EPL", "UWCL"):
            try:
                mapped_lg = resolve_league_team_name(raw, league)
            except Exception:  # noqa: BLE001
                continue
            if mapped_lg and str(mapped_lg).upper() != tid:
                db_meta3 = _resolve_team_hash_from_db(mapped_lg, db_path=db_path)
                if db_meta3 is not None:
                    return db_meta3
    return None


def _resolve_team_hash_from_db(
    team_id: str,
    *,
    db_path: Path | str | None = None,
) -> dict[str, str] | None:
    """Load hash/slug from ``teams`` table when present."""
    try:
        from src.global_db import GLOBAL_DB_PATH, connect_global_db, get_team_flashscore_meta
    except Exception:  # noqa: BLE001
        return None

    path = Path(db_path) if db_path is not None else GLOBAL_DB_PATH
    if str(path) != ":memory:" and not Path(path).is_file():
        return None
    try:
        conn = connect_global_db(path, init=True)
        try:
            meta = get_team_flashscore_meta(conn, team_id)
            if meta is None or not meta.get("hash"):
                # Also try men's gender explicitly for display names.
                meta = get_team_flashscore_meta(conn, team_id, gender="M")
            if meta is None or not meta.get("hash"):
                meta = get_team_flashscore_meta(conn, team_id, gender="W")
            if meta is None or not meta.get("hash"):
                return None
            return {
                "hash": str(meta["hash"]),
                "slug": str(meta.get("slug") or "").strip().strip("/"),
                "name": str(meta.get("canonical_name") or team_id),
                "team_id": str(meta.get("canonical_name") or team_id),
                "db_team_id": int(meta["team_id"]),
                "source": "db",
            }
        finally:
            conn.close()
    except Exception:  # noqa: BLE001
        return None


def slugify_team_name(name: str) -> str:
    """Best-effort Flashscore slug from a display name (``Vissel Kobe`` → ``vissel-kobe``)."""
    raw = str(name or "").strip().lower()
    raw = _COUNTRY_SUFFIX_RE.sub("", raw).strip()
    slug = _SLUG_RE.sub("-", raw).strip("-")
    return slug


def team_results_url(team_hash: str, *, slug: str | None = None, team_id: str | None = None) -> str:
    """Build ``/team/{slug}/{hash}/results/`` URL."""
    h = str(team_hash or "").strip()
    if not h:
        raise ValueError("team_hash is required")
    s = (slug or "").strip().strip("/")
    if not s and team_id:
        s = slug_for_team_id(team_id) or ""
    if not s and team_id:
        meta = team_hash_registry().get(str(team_id).upper()) or {}
        s = slugify_team_name(str(meta.get("name") or team_id))
    if not s:
        # Last resort — Flashscore 404s without a real slug, but keep a placeholder.
        s = "team"
    return f"{FLASHSCORE_BASE}/team/{s}/{h}/results/"


def strip_flashscore_country_suffix(name: str) -> str:
    """Remove ACL-style `` (Jpn)`` / `` (Tha)`` suffixes from team names."""
    return _COUNTRY_SUFFIX_RE.sub("", str(name or "").strip()).strip()


def infer_comp_id(competition_name: str) -> str:
    """Map Flashscore competition label → ``comp_id``.

    Prefers real codes (``J1``, ``EMPERORS_CUP``, ``ACL``, …). Unrecognised
    labels fall back to :data:`GENERIC_TEAM_FEED_COMP_ID` (``FLASH_TEAM``) so
    matches still land in ``global_matches.db`` for rest/fatigue.
    """
    label = str(competition_name or "").strip()
    if not label:
        return GENERIC_TEAM_FEED_COMP_ID
    # Prefer short code after colon: "JAPAN: J1 League"
    short = label.split(":")[-1].strip() if ":" in label else label
    for pat, code in _COMP_NAME_RULES:
        if pat.search(short) or pat.search(label):
            return code
    return GENERIC_TEAM_FEED_COMP_ID


def _resolve_feed_team_name(name: str) -> str:
    """Map scraped Flashscore name → stable ``JP_*`` / ``ES_*`` when known."""
    cleaned = strip_flashscore_country_suffix(name)
    if not cleaned:
        return cleaned
    # Prefer Emperor's Cup / J-League alias table (covers JP_* codes).
    mapped = resolve_league_team_name(cleaned, "EMPERORS_CUP")
    if mapped.startswith("JP_"):
        return mapped
    # Direct alias miss: try EMPERORS_CUP_TEAM_ALIASES exact keys.
    if cleaned in EMPERORS_CUP_TEAM_ALIASES:
        return EMPERORS_CUP_TEAM_ALIASES[cleaned]
    # LaLiga / other league stable codes.
    for league in ("LALIGA", "EPL", "UWCL"):
        try:
            mapped_lg = resolve_league_team_name(cleaned, league)
        except Exception:  # noqa: BLE001
            continue
        if mapped_lg and mapped_lg != cleaned and (
            mapped_lg.startswith("ES_")
            or mapped_lg.startswith("JP_")
            or "_" in mapped_lg
        ):
            return mapped_lg
        if mapped_lg.startswith("ES_"):
            return mapped_lg
    return mapped


def _resolve_jp_team_name(name: str) -> str:
    """Backward-compatible alias for :func:`_resolve_feed_team_name`."""
    return _resolve_feed_team_name(name)


def _parse_team_results_feed(raw: str) -> list[dict[str, Any]]:
    """Parse finished events from a team ``results`` / ``summary-results`` feed."""
    rows: list[dict[str, Any]] = []
    current_comp = ""
    current_comp_short = ""
    current_path = ""
    for sec in raw.split("~"):
        fields = _flashscore_parse_fields(sec)
        if not fields:
            continue
        if "ZA" in fields or "ZK" in fields:
            current_comp = str(fields.get("ZA") or fields.get("ZK") or current_comp)
            current_comp_short = str(fields.get("ZK") or fields.get("ZAC") or "")
            current_path = str(fields.get("ZL") or "")
            continue
        if "AA" not in fields or "AE" not in fields:
            continue
        status = str(fields.get("AB", ""))
        if status != "3":
            continue
        try:
            ts = int(fields.get("AD") or 0)
        except (TypeError, ValueError):
            ts = 0
        if not ts:
            continue
        kickoff = datetime.fromtimestamp(ts, tz=timezone.utc).replace(tzinfo=None)
        try:
            hg = int(fields.get("AG", ""))
            ag = int(fields.get("AH", ""))
        except (TypeError, ValueError):
            continue
        home_raw = str(fields.get("AE") or "")
        away_raw = str(fields.get("AF") or "")
        home = _resolve_feed_team_name(home_raw)
        away = _resolve_feed_team_name(away_raw)
        comp_name = current_comp or current_comp_short or "Unknown"
        eid = str(fields["AA"])
        rows.append(
            {
                "match_date": pd.Timestamp(kickoff).normalize(),
                "kickoff": pd.Timestamp(kickoff),
                "home_team": home,
                "away_team": away,
                "home_team_raw": strip_flashscore_country_suffix(home_raw),
                "away_team_raw": strip_flashscore_country_suffix(away_raw),
                "score": f"{hg}-{ag}",
                "home_goals": hg,
                "away_goals": ag,
                "competition_name": comp_name,
                "comp_id": infer_comp_id(comp_name),
                "flashscore_event_id": eid,
                "flashscore_path": current_path or None,
                "source": "flashscore_team",
            }
        )
    # Newest first in feed; keep order, drop duplicate event ids.
    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    for row in rows:
        eid = str(row["flashscore_event_id"])
        if eid in seen:
            continue
        seen.add(eid)
        unique.append(row)
    return unique


def fetch_team_recent_matches(
    team_hash: str,
    n_matches: int = DEFAULT_N_MATCHES,
    *,
    slug: str | None = None,
    team_id: str | None = None,
    prefer_full_feed: bool = True,
    timeout: float = DEFAULT_TIMEOUT_S,
    retries: int = DEFAULT_RETRIES,
) -> list[dict[str, Any]]:
    """Scrape last ``n_matches`` finished games for a team across all competitions.

    Parameters
    ----------
    team_hash:
        Flashscore team id (e.g. ``698tGI9q`` for Vissel Kobe).
    n_matches:
        Maximum finished matches to return (most recent first).
    slug / team_id:
        Used to build the results URL when ``slug`` is omitted.
    prefer_full_feed:
        Prefer ``results`` feed over ``summary-results``.
    timeout / retries:
        HTTP timeout (seconds) and retry count (use ``timeout=3`` for lazy sync).

    Returns
    -------
    list[dict]
        Each dict: ``match_date``, ``home_team``, ``away_team``, ``score``,
        ``competition_name``, plus ``flashscore_event_id``, ``home_goals``,
        ``away_goals``, ``comp_id`` for DB upsert. On failure returns ``[]``
        and emits a warning (never raises for network/parse errors).
    """
    tid = team_id or team_id_for_hash(team_hash)
    try:
        url = team_results_url(team_hash, slug=slug, team_id=tid)
    except ValueError as exc:
        warnings.warn(f"team results URL: {exc}", stacklevel=2)
        return []

    try:
        html = _fetch_text_retry(url, retries=max(1, int(retries)), timeout=float(timeout))
    except Exception as exc:  # noqa: BLE001
        msg = f"Flashscore team results failed for {team_hash}: {exc}"
        warnings.warn(msg, stacklevel=2)
        logger.warning(msg)
        return []

    raw = ""
    if prefer_full_feed:
        raw = _flashscore_extract_initial_feed(html, "results")
    if not raw:
        raw = _flashscore_extract_initial_feed(html, "summary-results")
    if not raw:
        msg = f"No results feed on team page {url}"
        warnings.warn(msg, stacklevel=2)
        logger.warning(msg)
        return []

    rows = _parse_team_results_feed(raw)
    n = max(0, int(n_matches))
    return rows[:n] if n else rows


def fetch_team_recent_matches_by_id(
    team_id: str,
    n_matches: int = DEFAULT_N_MATCHES,
    *,
    timeout: float = DEFAULT_TIMEOUT_S,
    retries: int = DEFAULT_RETRIES,
) -> list[dict[str, Any]]:
    """Convenience: resolve ``JP_*`` → hash/slug then :func:`fetch_team_recent_matches`."""
    meta = resolve_team_hash(team_id)
    if not meta:
        tid = str(team_id or "").strip().upper()
        warnings.warn(f"No Flashscore hash registered for {tid}", stacklevel=2)
        return []
    tid = str(meta.get("team_id") or team_id).strip().upper()
    return fetch_team_recent_matches(
        meta["hash"],
        n_matches=n_matches,
        slug=meta.get("slug") or None,
        team_id=tid,
        timeout=timeout,
        retries=retries,
    )


def fetch_teams_recent_matches_parallel(
    team_ids: Sequence[str],
    n_matches: int = DEFAULT_N_MATCHES,
    *,
    max_workers: int = 4,
) -> dict[str, list[dict[str, Any]]]:
    """Fetch several team feeds concurrently (ThreadPoolExecutor)."""
    ids = [str(t).strip().upper() for t in team_ids if str(t).strip()]
    out: dict[str, list[dict[str, Any]]] = {tid: [] for tid in ids}
    if not ids:
        return out
    workers = max(1, min(int(max_workers), len(ids)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {
            pool.submit(fetch_team_recent_matches_by_id, tid, n_matches): tid
            for tid in ids
        }
        for fut in as_completed(futs):
            tid = futs[fut]
            try:
                out[tid] = fut.result()
            except Exception as exc:  # noqa: BLE001
                warnings.warn(f"parallel team fetch {tid}: {exc}", stacklevel=2)
                out[tid] = []
    return out


def matches_to_legacy_frames(
    matches: Sequence[Mapping[str, Any]],
) -> dict[str, pd.DataFrame]:
    """Group scraped team matches into legacy DataFrames keyed by ``comp_id``."""
    by_comp: dict[str, list[dict[str, Any]]] = {}
    for m in matches:
        comp = str(m.get("comp_id") or GENERIC_TEAM_FEED_COMP_ID).upper()
        date = m.get("match_date")
        home = str(m.get("home_team") or "")
        away = str(m.get("away_team") or "")
        if not home or not away or date is None or (isinstance(date, float) and pd.isna(date)):
            continue
        try:
            hg = int(m.get("home_goals"))  # type: ignore[arg-type]
            ag = int(m.get("away_goals"))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            score = str(m.get("score") or "")
            if "-" not in score:
                continue
            try:
                hg_s, ag_s = score.split("-", 1)
                hg, ag = int(hg_s), int(ag_s)
            except (TypeError, ValueError):
                continue
        eid = str(m.get("flashscore_event_id") or "").strip()
        row = {
            "Date": pd.Timestamp(date).normalize(),
            "HomeTeam": home,
            "AwayTeam": away,
            "FTHG": hg,
            "FTAG": ag,
            "FTR": "H" if hg > ag else ("A" if hg < ag else "D"),
            "Source": str(m.get("source") or "flashscore_team"),
            "league_id": comp,
            "FlashscoreEventId": eid or None,
            # Prefer event id for idempotent re-import within the same comp.
            "Match_ID": eid or None,
        }
        by_comp.setdefault(comp, []).append(row)

    frames: dict[str, pd.DataFrame] = {}
    for comp, rows in by_comp.items():
        df = pd.DataFrame(rows)
        if df.empty:
            continue
        # Dedupe: event id first, else date+home+away.
        if "FlashscoreEventId" in df.columns:
            df = df.sort_values("Date").drop_duplicates(
                subset=["FlashscoreEventId"], keep="last"
            )
        df = df.drop_duplicates(subset=["Date", "HomeTeam", "AwayTeam"], keep="last")
        frames[comp] = df.reset_index(drop=True)
    return frames


def _existing_match_keys(conn: Any) -> set[tuple[str, str, str]]:
    """Set of ``(match_date, home_canon, away_canon)`` already in DB (cross-comp).

    Uses stored ``canonical_name`` values as-is (no live alias remap). Re-applying
    ``canonical_team_name`` would map legacy ``Barcelona`` rows to ``ES_BARCELONA``
    and incorrectly block fresh team-feed upserts under stable codes.
    """
    rows = conn.execute(
        """
        SELECT m.match_date, th.canonical_name, ta.canonical_name
        FROM matches m
        JOIN teams th ON th.team_id = m.home_team_id
        JOIN teams ta ON ta.team_id = m.away_team_id
        """
    ).fetchall()
    keys: set[tuple[str, str, str]] = set()
    for date_s, home, away in rows:
        keys.add((str(date_s)[:10], str(home), str(away)))
    return keys


def _existing_event_match_ids(conn: Any) -> set[str]:
    """Match ids that already embed a Flashscore event id (``COMP|eventId``)."""
    rows = conn.execute("SELECT match_id FROM matches").fetchall()
    return {str(r[0]) for r in rows}


def persist_team_matches_to_global_db(
    matches: Sequence[Mapping[str, Any]],
    *,
    db_path: Path | str | None = None,
    skip_cross_comp_duplicates: bool = True,
) -> dict[str, Any]:
    """Upsert scraped team-page matches into ``global_matches.db``.

    Uses inferred ``comp_id`` (``J1``, ``EMPERORS_CUP``, ``ACL``, …) when the
    competition name matches; otherwise ``FLASH_TEAM``. Team names are stored
    as stable ``JP_*`` codes when aliases resolve.

    Dedup
    -----
    * Within batch: Flashscore event id, then date+home+away.
    * Against DB: skip rows whose date+home+away already exist in any
      competition (avoids double-counting rest_days when Emperor's Cup
      league scrape already inserted the same fixture).
    """
    from src.global_db import (
        GLOBAL_DB_PATH,
        build_global_match_id,
        canonical_team_name,
        competition_gender,
        connect_global_db,
        upsert_competition,
        upsert_match,
        upsert_team,
    )

    ensure_jp_aliases_registered()
    path = Path(db_path) if db_path is not None else GLOBAL_DB_PATH
    frames = matches_to_legacy_frames(matches)
    stats: dict[str, Any] = {
        "comps": {},
        "matches_upserted": 0,
        "skipped_duplicate": 0,
        "skipped_empty": 0,
    }
    if not frames:
        stats["skipped_empty"] = len(matches)
        return stats

    # Human labels for new comps.
    comp_labels = {
        "J1": "J1 League - Japan",
        "J2": "J2 League - Japan",
        "J3": "J3 League - Japan",
        "EMPERORS_CUP": "Emperor's Cup - Japan",
        "ACL": "AFC Champions League",
        "J_LEAGUE_CUP": "J.League Cup - Japan",
        "FRIENDLY": "Club Friendly",
        "LALIGA": "LaLiga - Spain",
        "EPL": "Premier League",
        "UWCL": "UEFA Women's Champions League",
        "WSL": "Women's Super League",
        GENERIC_TEAM_FEED_COMP_ID: "Flashscore team feed (unmapped)",
    }

    conn = connect_global_db(path, init=True)
    try:
        existing_keys = _existing_match_keys(conn) if skip_cross_comp_duplicates else set()
        existing_ids = _existing_event_match_ids(conn)

        for comp, df in frames.items():
            upsert_competition(
                conn,
                comp,
                comp_labels.get(comp, comp),
                0.75 if comp not in {"EMPERORS_CUP", "J1"} else (0.8 if comp == "EMPERORS_CUP" else 0.9),
            )
            gender = competition_gender(comp)
            upserted = 0
            skipped = 0
            for _, row in df.iterrows():
                home = canonical_team_name(str(row["HomeTeam"]))
                away = canonical_team_name(str(row["AwayTeam"]))
                date_s = pd.Timestamp(row["Date"]).strftime("%Y-%m-%d")
                key = (date_s, home, away)
                eid = str(row.get("Match_ID") or row.get("FlashscoreEventId") or "").strip()
                event_match_id = f"{comp}|{eid}" if eid else None

                if skip_cross_comp_duplicates and key in existing_keys:
                    skipped += 1
                    continue
                if event_match_id and event_match_id in existing_ids:
                    skipped += 1
                    continue

                home_id = upsert_team(conn, home, gender=gender, comp_id=comp)
                away_id = upsert_team(conn, away, gender=gender, comp_id=comp)
                # Prefer date+teams key so we align with league scrapes that lack
                # Match_ID; still embed event id when this is a fresh insert path
                # that won't collide with an existing date-key row.
                match_id = build_global_match_id(comp, date_s, home, away)
                if match_id in existing_ids:
                    skipped += 1
                    continue

                upsert_match(
                    conn,
                    match_id=match_id,
                    comp_id=comp,
                    home_team_id=home_id,
                    away_team_id=away_id,
                    match_date=date_s,
                    home_score=int(row["FTHG"]),
                    away_score=int(row["FTAG"]),
                    extras={
                        "FTR": row.get("FTR"),
                        "fotmob_match_id": eid or None,
                    },
                )
                existing_keys.add(key)
                existing_ids.add(match_id)
                if event_match_id:
                    existing_ids.add(event_match_id)
                upserted += 1

            stats["comps"][comp] = {
                "rows": int(len(df)),
                "upserted": upserted,
                "skipped_duplicate": skipped,
            }
            stats["matches_upserted"] += upserted
            stats["skipped_duplicate"] += skipped

        conn.commit()
    finally:
        conn.close()

    return stats


def refresh_team_feeds_for_sides(
    team_ids: Sequence[str],
    *,
    n_matches: int = DEFAULT_N_MATCHES,
    db_path: Path | str | None = None,
    max_workers: int = 4,
) -> dict[str, Any]:
    """Fetch + persist team feeds for fixture sides (parallel).

    Safe no-op for team ids without a registered Flashscore hash (EPL/UWCL).
    """
    known = [str(t).strip().upper() for t in team_ids if hash_for_team_id(str(t))]
    unknown = [
        str(t).strip().upper()
        for t in team_ids
        if str(t).strip() and not hash_for_team_id(str(t))
    ]
    result: dict[str, Any] = {
        "requested": [str(t).strip().upper() for t in team_ids if str(t).strip()],
        "fetched": {},
        "persist": {},
        "skipped_no_hash": unknown,
    }
    if not known:
        return result

    feeds = fetch_teams_recent_matches_parallel(
        known, n_matches=n_matches, max_workers=max_workers
    )
    all_matches: list[dict[str, Any]] = []
    for tid, rows in feeds.items():
        result["fetched"][tid] = len(rows)
        all_matches.extend(rows)

    # Deduplicate across both sides (shared fixtures).
    by_eid: dict[str, dict[str, Any]] = {}
    no_eid: list[dict[str, Any]] = []
    for m in all_matches:
        eid = str(m.get("flashscore_event_id") or "")
        if eid:
            by_eid[eid] = m
        else:
            no_eid.append(m)
    merged = list(by_eid.values()) + no_eid
    result["persist"] = persist_team_matches_to_global_db(merged, db_path=db_path)
    return result


def ensure_jp_aliases_registered() -> None:
    """Register ``JP_*`` / ``ES_*`` identity aliases into global ``TEAM_ALIASES``."""
    from src.fetchers.flashscore_league import register_league_aliases_globally

    register_league_aliases_globally("EMPERORS_CUP")
    register_league_aliases_globally("LALIGA")
    try:
        register_league_aliases_globally("EPL")
    except Exception:  # noqa: BLE001
        pass


def seed_static_team_hashes_to_db(
    *,
    db_path: Path | str | None = None,
    comp_id: str = "EMPERORS_CUP",
) -> int:
    """Copy last-resort :data:`DEFAULT_TEAM_HASHES` / config into ``teams`` table."""
    from src.global_db import GLOBAL_DB_PATH, connect_global_db, upsert_team_flashscore_hash

    path = Path(db_path) if db_path is not None else GLOBAL_DB_PATH
    n = 0
    conn = connect_global_db(path, init=True)
    try:
        for tid, meta in team_hash_registry().items():
            h = str(meta.get("hash") or "").strip()
            if not h:
                continue
            upsert_team_flashscore_hash(
                conn,
                tid,
                h,
                flashscore_slug=str(meta.get("slug") or "").strip() or None,
                comp_id=comp_id,
            )
            n += 1
        conn.commit()
    finally:
        conn.close()
    return n


def _list_upcoming_within_days(
    *,
    days_ahead: int = 3,
    league_keys: Sequence[str] | None = None,
    db_path: Path | str | None = None,
    now: datetime | None = None,
) -> pd.DataFrame:
    """Load upcoming fixtures from global DB cache within ``days_ahead`` days."""
    from src.global_db import GLOBAL_DB_PATH, connect_global_db, load_upcoming_from_db

    path = Path(db_path) if db_path is not None else GLOBAL_DB_PATH
    ref = now or datetime.now(timezone.utc).replace(tzinfo=None)
    ref_ts = pd.Timestamp(ref)
    horizon = ref_ts + pd.Timedelta(days=float(days_ahead))

    codes: list[str]
    if league_keys:
        codes = [str(k).strip().upper() for k in league_keys if str(k).strip()]
    else:
        codes = []
        try:
            conn = connect_global_db(path, init=True)
            try:
                rows = conn.execute(
                    "SELECT DISTINCT comp_id FROM upcoming_fixtures"
                ).fetchall()
                codes = [str(r[0]).upper() for r in rows]
            finally:
                conn.close()
        except Exception:  # noqa: BLE001
            codes = []

    frames: list[pd.DataFrame] = []
    for code in codes:
        df, _ = load_upcoming_from_db(code, db_path=path)
        if df is None or df.empty:
            continue
        work = df.copy()
        work["league_id"] = code
        kick = pd.to_datetime(
            work["Kickoff"] if "Kickoff" in work.columns else work.get("Date"),
            errors="coerce",
        )
        mask = (kick >= ref_ts) & (kick <= horizon)
        work = work.loc[mask].copy()
        if not work.empty:
            frames.append(work)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def _unique_team_codes_from_upcoming(upcoming: pd.DataFrame) -> list[str]:
    """Collect unique HomeTeam/AwayTeam codes from an upcoming frame."""
    names: list[str] = []
    for col in ("HomeTeam", "AwayTeam", "home", "away", "home_team", "away_team"):
        if col not in upcoming.columns:
            continue
        for v in upcoming[col].tolist():
            s = str(v or "").strip()
            if s:
                names.append(s)
    # Preserve order, casefold-dedupe by upper.
    seen: set[str] = set()
    out: list[str] = []
    for n in names:
        key = n.upper()
        if key in seen:
            continue
        seen.add(key)
        out.append(n)
    return out


def _ensure_hashes_for_teams(
    team_codes: Sequence[str],
    league_keys: Sequence[str],
    *,
    db_path: Path | str | None = None,
) -> dict[str, Any]:
    """Fill missing flashscore_hash via league fixtures/results scrape."""
    from src.fetchers.flashscore_league import scrape_and_persist_league_team_hashes

    missing = [t for t in team_codes if not hash_for_team_id(t)]
    stats: dict[str, Any] = {
        "missing_before": list(missing),
        "scrapes": {},
        "missing_after": [],
    }
    if not missing:
        return stats
    for league in league_keys:
        try:
            stats["scrapes"][league] = scrape_and_persist_league_team_hashes(
                league, db_path=db_path
            )
        except Exception as exc:  # noqa: BLE001
            stats["scrapes"][league] = {"error": str(exc)}
            logger.warning("hash scrape %s: %s", league, exc)
    stats["missing_after"] = [t for t in team_codes if not hash_for_team_id(t)]
    return stats


def should_update_team_feed(
    team_id: int | str,
    *,
    db_path: Path | str | None = None,
    now: datetime | None = None,
    stale_hours: float = FEED_STALE_HOURS,
) -> bool:
    """Return True when Team Feed should be scraped (missing or older than 24h)."""
    from src.global_db import GLOBAL_DB_PATH, connect_global_db, get_team_feed_meta

    path = Path(db_path) if db_path is not None else GLOBAL_DB_PATH
    try:
        conn = connect_global_db(path, init=True)
        try:
            meta = get_team_feed_meta(conn, team_id)
        finally:
            conn.close()
    except Exception:  # noqa: BLE001
        return True
    if meta is None:
        return True
    raw = str(meta.get("feed_updated_at") or "").strip()
    if not raw:
        return True
    try:
        updated = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        if updated.tzinfo is None:
            updated = updated.replace(tzinfo=timezone.utc)
    except (TypeError, ValueError):
        return True
    ref = now or datetime.now(timezone.utc)
    if getattr(ref, "tzinfo", None) is None:
        ref = ref.replace(tzinfo=timezone.utc)
    age_h = (ref - updated).total_seconds() / 3600.0
    return age_h >= float(stale_hours)


def _team_codes_from_upcoming_frame(
    upcoming_matches: pd.DataFrame | Sequence[Mapping[str, Any]],
    *,
    max_teams: int = FAST_FEED_MAX_TEAMS,
) -> list[str]:
    """Unique home/away codes from the first ``max_teams`` UI rows."""
    if isinstance(upcoming_matches, pd.DataFrame):
        rows = upcoming_matches.head(max(0, int(max_teams)))
        records = rows.to_dict(orient="records")
    else:
        records = list(upcoming_matches)[: max(0, int(max_teams))]
    seen: set[str] = set()
    out: list[str] = []
    for row in records:
        for key in (
            "HomeTeam",
            "AwayTeam",
            "home",
            "away",
            "home_team",
            "away_team",
        ):
            raw = str(row.get(key) or "").strip()
            if not raw:
                continue
            uk = raw.upper()
            if uk in seen:
                continue
            seen.add(uk)
            out.append(raw)
    return out


def sync_upcoming_teams_fast(
    upcoming_matches: pd.DataFrame | Sequence[Mapping[str, Any]],
    max_teams: int = FAST_FEED_MAX_TEAMS,
    *,
    n_matches: int = DEFAULT_N_MATCHES,
    max_workers: int = FAST_FEED_MAX_WORKERS,
    timeout: float = FAST_FEED_TIMEOUT_S,
    db_path: Path | str | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Lazy Team Feed sync for Top-N upcoming UI sides (rate-limit friendly).

    * Only teams appearing in the first ``max_teams`` upcoming rows.
    * Skips sides whose ``feed_updated_at`` is fresher than 24h.
    * ``ThreadPoolExecutor(max_workers=3)`` + per-request ``timeout`` (default 3s).
    * On timeout/error keeps prior ``last_match_date`` / matches; still does not
      stamp ``feed_updated_at`` so the next call retries.
    """
    from src.global_db import (
        GLOBAL_DB_PATH,
        connect_global_db,
        resolve_team_id,
        set_team_feed_meta,
    )

    ensure_jp_aliases_registered()
    path = Path(db_path) if db_path is not None else GLOBAL_DB_PATH
    ref = now or datetime.now(timezone.utc)
    if getattr(ref, "tzinfo", None) is None:
        ref = ref.replace(tzinfo=timezone.utc)

    codes = _team_codes_from_upcoming_frame(
        upcoming_matches, max_teams=max_teams
    )
    result: dict[str, Any] = {
        "requested_codes": codes,
        "to_fetch": [],
        "skipped_fresh": [],
        "skipped_no_hash": [],
        "fetched": {},
        "errors": {},
        "persist": {},
        "updated_meta": [],
    }
    if not codes:
        return result

    # Resolve integer team ids + filter by 24h TTL.
    conn = connect_global_db(path, init=True)
    id_by_code: dict[str, int] = {}
    try:
        for code in codes:
            tid = resolve_team_id(conn, code, gender="M", create=False)
            if tid is None:
                tid = resolve_team_id(conn, code, gender="W", create=False)
            if tid is None:
                # Create stub so hash resolve / later upsert can attach.
                from src.global_db import upsert_team

                tid = upsert_team(conn, code, gender="M")
            id_by_code[code.upper()] = int(tid)
            if not hash_for_team_id(code):
                result["skipped_no_hash"].append(code)
                continue
            if should_update_team_feed(tid, db_path=path, now=ref):
                result["to_fetch"].append(code)
            else:
                result["skipped_fresh"].append(code)
        conn.commit()
    finally:
        conn.close()

    to_fetch = list(result["to_fetch"])
    if not to_fetch:
        return result

    workers = max(1, min(int(max_workers), 3, len(to_fetch)))
    feeds: dict[str, list[dict[str, Any]]] = {}

    def _one(code: str) -> tuple[str, list[dict[str, Any]], str | None]:
        try:
            rows = fetch_team_recent_matches_by_id(
                code,
                n_matches=n_matches,
                timeout=float(timeout),
                retries=1,
            )
            return code, rows, None
        except Exception as exc:  # noqa: BLE001
            return code, [], str(exc)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(_one, c): c for c in to_fetch}
        for fut in as_completed(futs):
            code, rows, err = fut.result()
            result["fetched"][code] = len(rows)
            if err:
                result["errors"][code] = err
            if rows:
                feeds[code] = rows

    all_matches: list[dict[str, Any]] = []
    for rows in feeds.values():
        all_matches.extend(rows)
    if all_matches:
        by_eid: dict[str, dict[str, Any]] = {}
        no_eid: list[dict[str, Any]] = []
        for m in all_matches:
            eid = str(m.get("flashscore_event_id") or "")
            if eid:
                by_eid[eid] = m
            else:
                no_eid.append(m)
        result["persist"] = persist_team_matches_to_global_db(
            list(by_eid.values()) + no_eid, db_path=path
        )

    # Stamp last_match_date + feed_updated_at only for successful fetches.
    conn = connect_global_db(path, init=True)
    try:
        for code, rows in feeds.items():
            tid = id_by_code.get(code.upper())
            if tid is None:
                continue
            last_date = None
            if rows:
                try:
                    last_date = pd.Timestamp(rows[0]["match_date"]).strftime("%Y-%m-%d")
                except (TypeError, ValueError, KeyError):
                    last_date = None
            set_team_feed_meta(
                conn,
                tid,
                last_match_date=last_date,
                feed_updated_at=ref,
            )
            result["updated_meta"].append(
                {"team": code, "team_id": tid, "last_match_date": last_date}
            )
        conn.commit()
    finally:
        conn.close()

    return result


def update_all_upcoming_teams_rest_days(
    days_ahead: int = 3,
    league_keys: Sequence[str] | None = None,
    *,
    n_matches: int = DEFAULT_N_MATCHES,
    max_workers: int = 6,
    db_path: Path | str | None = None,
    now: datetime | None = None,
    seed_static: bool = True,
) -> dict[str, Any]:
    """Batch-refresh multi-comp rest_days for upcoming fixture sides.

    Steps
    -----
    1. Load upcoming fixtures from ``global_matches.upcoming_fixtures`` within
       ``days_ahead`` days (optionally filtered by ``league_keys``).
    2. Collect unique team codes; fill missing ``flashscore_hash`` by scraping
       league fixtures/results (no hardcoding required for new clubs).
    3. Parallel-fetch team results feeds (``max_workers`` ≈ 5–8), persist
       multi-comp matches, compute ``rest_days`` / ``matches_last_14d``.
    4. Write ``team_rest_cache`` + enrich upcoming ``row_json`` so Lite Mode
       can read cache first (<0.1s).

    Returns a summary dict with counts, per-team rest samples, and outliers
    (``rest_days >= 14`` while ``matches_last_14d`` suggests recent play — or
    the inverse: capped rest when feed looked empty).
    """
    from src.features import calculate_multi_comp_features
    from src.global_db import (
        GLOBAL_DB_PATH,
        connect_global_db,
        competition_gender,
        load_upcoming_from_db,
        read_matches_as_legacy,
        resolve_team_id,
        save_upcoming_to_db,
        set_team_rest_cache,
    )

    ensure_jp_aliases_registered()
    path = Path(db_path) if db_path is not None else GLOBAL_DB_PATH
    if seed_static:
        try:
            seed_static_team_hashes_to_db(db_path=path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("seed static hashes: %s", exc)

    leagues = (
        [str(k).strip().upper() for k in league_keys if str(k).strip()]
        if league_keys
        else None
    )
    # Default leagues when caller passes None — cover configured comps with cache.
    if leagues is None:
        leagues = ["EMPERORS_CUP", "LALIGA", "EPL", "UWCL"]

    upcoming = _list_upcoming_within_days(
        days_ahead=days_ahead,
        league_keys=leagues,
        db_path=path,
        now=now,
    )
    result: dict[str, Any] = {
        "days_ahead": int(days_ahead),
        "league_keys": leagues,
        "upcoming_rows": int(len(upcoming)),
        "teams": {},
        "fixtures": [],
        "outliers": [],
        "hash_fill": {},
        "persist": {},
        "fetched": {},
    }
    if upcoming.empty:
        result["note"] = "no upcoming fixtures in cache within horizon"
        return result

    team_codes = _unique_team_codes_from_upcoming(upcoming)
    result["team_codes"] = team_codes
    result["hash_fill"] = _ensure_hashes_for_teams(
        team_codes, leagues, db_path=path
    )

    known = [t for t in team_codes if hash_for_team_id(t)]
    unknown = [t for t in team_codes if not hash_for_team_id(t)]
    result["skipped_no_hash"] = unknown

    workers = max(1, min(int(max_workers), 8, max(1, len(known))))
    if known:
        feeds = fetch_teams_recent_matches_parallel(
            known, n_matches=n_matches, max_workers=workers
        )
        all_matches: list[dict[str, Any]] = []
        for tid, rows in feeds.items():
            result["fetched"][tid] = len(rows)
            all_matches.extend(rows)
        by_eid: dict[str, dict[str, Any]] = {}
        no_eid: list[dict[str, Any]] = []
        for m in all_matches:
            eid = str(m.get("flashscore_event_id") or "")
            if eid:
                by_eid[eid] = m
            else:
                no_eid.append(m)
        merged = list(by_eid.values()) + no_eid
        result["persist"] = persist_team_matches_to_global_db(merged, db_path=path)

    # Recompute rest from global history.
    try:
        hist = read_matches_as_legacy(path)
    except Exception:  # noqa: BLE001
        hist = pd.DataFrame()

    conn = connect_global_db(path, init=True)
    team_rest: dict[tuple[str, str], dict[str, Any]] = {}
    try:
        # Per-team rest keyed by (code, ref_date).
        for _, fx in upcoming.iterrows():
            home = str(fx.get("HomeTeam") or fx.get("home") or "").strip()
            away = str(fx.get("AwayTeam") or fx.get("away") or "").strip()
            kick = fx.get("Kickoff") if "Kickoff" in fx.index else fx.get("Date")
            if not home or not away or kick is None or pd.isna(kick):
                continue
            kick_ts = pd.Timestamp(kick)
            ref_date = kick_ts.strftime("%Y-%m-%d")
            comp = str(
                fx.get("league_id") or fx.get("league") or fx.get("comp_id") or ""
            ).strip().upper()
            gender = competition_gender(comp) if comp else "M"

            side_feats: dict[str, dict[str, Any]] = {}
            for side, name in (("home", home), ("away", away)):
                key = (name.upper(), ref_date)
                if key not in team_rest:
                    tid = resolve_team_id(
                        conn, name, comp_id=comp or None, gender=gender, create=False
                    )
                    feats: dict[str, Any]
                    if tid is not None and hist is not None and not hist.empty:
                        feats = calculate_multi_comp_features(
                            tid,
                            kick_ts,
                            hist,
                            upcoming_comp_id=comp or None,
                        )
                        set_team_rest_cache(
                            conn,
                            tid,
                            ref_date,
                            rest_days=feats.get("rest_days"),
                            matches_last_14d=feats.get("matches_last_14d"),
                        )
                    else:
                        feats = {
                            "rest_days": float("nan"),
                            "matches_last_14d": 0.0,
                        }
                    team_rest[key] = {
                        "team": name,
                        "db_team_id": tid,
                        "ref_date": ref_date,
                        "rest_days": feats.get("rest_days"),
                        "matches_last_14d": feats.get("matches_last_14d"),
                    }
                side_feats[side] = team_rest[key]

            h_rest = side_feats["home"].get("rest_days")
            a_rest = side_feats["away"].get("rest_days")
            h_n14 = side_feats["home"].get("matches_last_14d")
            a_n14 = side_feats["away"].get("matches_last_14d")
            fx_row = {
                "comp_id": comp,
                "home": home,
                "away": away,
                "kickoff": kick_ts.isoformat(),
                "home_rest_days": h_rest,
                "away_rest_days": a_rest,
                "home_matches_last_14d": h_n14,
                "away_matches_last_14d": a_n14,
            }
            result["fixtures"].append(fx_row)

            # Flag suspicious rest (≥14) when the side clearly played recently
            # in the refreshed feed window (matches_last_14d >= 1 should imply
            # rest_days < 14 unless hard-cap off-season logic applied).
            for side in ("home", "away"):
                rd = side_feats[side].get("rest_days")
                n14 = float(side_feats[side].get("matches_last_14d") or 0)
                try:
                    rd_f = float(rd)  # type: ignore[arg-type]
                except (TypeError, ValueError):
                    continue
                if rd_f != rd_f:
                    continue
                if rd_f >= 14.0 and n14 >= 1.0:
                    result["outliers"].append(
                        {
                            **fx_row,
                            "side": side,
                            "team": side_feats[side]["team"],
                            "reason": "rest_days>=14 but matches_last_14d>=1",
                        }
                    )

        conn.commit()
    finally:
        conn.close()

    # Enrich upcoming_fixtures row_json AFTER releasing the writer lock so the
    # cache reload/save path cannot race with an open transaction.
    for comp in {str(r.get("comp_id") or "").upper() for r in result["fixtures"]}:
        if not comp:
            continue
        cached, _ = load_upcoming_from_db(comp, db_path=path)
        if cached is None or cached.empty:
            # Fall back to in-memory upcoming slice for this competition.
            if "league_id" in upcoming.columns:
                cached = upcoming.loc[
                    upcoming["league_id"].astype(str).str.upper() == comp
                ].copy()
            else:
                continue
        if cached is None or cached.empty:
            continue
        enriched = cached.copy()
        for col in (
            "home_rest_days",
            "away_rest_days",
            "home_matches_last_14d",
            "away_matches_last_14d",
        ):
            if col not in enriched.columns:
                enriched[col] = float("nan")
        for fx_row in result["fixtures"]:
            if str(fx_row.get("comp_id") or "").upper() != comp:
                continue
            mask = (
                (enriched["HomeTeam"].astype(str) == str(fx_row["home"]))
                & (enriched["AwayTeam"].astype(str) == str(fx_row["away"]))
            )
            if not mask.any():
                continue
            enriched.loc[mask, "home_rest_days"] = fx_row["home_rest_days"]
            enriched.loc[mask, "away_rest_days"] = fx_row["away_rest_days"]
            enriched.loc[mask, "home_matches_last_14d"] = fx_row[
                "home_matches_last_14d"
            ]
            enriched.loc[mask, "away_matches_last_14d"] = fx_row[
                "away_matches_last_14d"
            ]
        save_upcoming_to_db(comp, enriched, db_path=path)

    # Compact per-team summary (first ref_date wins for display).
    for (code, _ref), info in team_rest.items():
        if code not in result["teams"]:
            result["teams"][code] = {
                "team": info["team"],
                "rest_days": info["rest_days"],
                "matches_last_14d": info["matches_last_14d"],
                "ref_date": info["ref_date"],
            }

    return result


# ---------------------------------------------------------------------------
# Manual Flashscore team URL import (UI-driven, single-team)
# ---------------------------------------------------------------------------

# Accepts /team/{slug}/{hash}/ with optional results/ suffix and query params.
_TEAM_URL_RE = re.compile(
    r"(?:https?://)?(?:www\.)?flashscore\.[a-z.]+/team/"
    r"(?P<slug>[A-Za-z0-9_-]+)/(?P<hash>[A-Za-z0-9]+)"
    r"(?:/(?:results?)?)?/?(?:\?[^#\s]*)?(?:#\S*)?\s*$",
    re.IGNORECASE,
)


def parse_flashscore_team_url(url: str) -> dict[str, str]:
    """Extract ``slug`` + ``hash`` from a Flashscore team page URL.

    Accepted forms
    --------------
    * ``https://www.flashscore.com/team/vissel-kobe/698tGI9q/``
    * ``…/team/vissel-kobe/698tGI9q/results/``
    * Query params / fragments allowed; scheme and ``www.`` optional.

    Returns
    -------
    dict
        ``{"slug": "vissel-kobe", "hash": "698tGI9q"}``.

    Raises
    ------
    ValueError
        When the URL does not match ``/team/{slug}/{hash}/``.
    """
    raw = str(url or "").strip()
    if not raw:
        raise ValueError("URL trống — hãy dán link Flashscore team.")
    m = _TEAM_URL_RE.match(raw)
    if not m:
        raise ValueError(
            "URL không hợp lệ. Ví dụ: "
            "https://www.flashscore.com/team/vissel-kobe/698tGI9q/"
        )
    slug = str(m.group("slug") or "").strip().strip("/")
    h = str(m.group("hash") or "").strip()
    if not slug or not h:
        raise ValueError(
            "URL thiếu slug hoặc hash. Ví dụ: "
            "https://www.flashscore.com/team/vissel-kobe/698tGI9q/"
        )
    return {"slug": slug, "hash": h}


def _count_team_matches_in_db(
    conn: Any,
    db_team_id: int,
) -> int:
    """Return finished-match count for ``db_team_id`` in ``matches``."""
    row = conn.execute(
        """
        SELECT COUNT(*) AS n FROM matches
        WHERE home_team_id = ? OR away_team_id = ?
        """,
        (int(db_team_id), int(db_team_id)),
    ).fetchone()
    if row is None:
        return 0
    try:
        return int(row["n"] if hasattr(row, "keys") else row[0])
    except (TypeError, ValueError, KeyError, IndexError):
        return 0


def team_flashscore_data_status(
    team_id: str | int,
    *,
    db_path: Path | str | None = None,
    gender: str | None = None,
    comp_id: str | None = None,
) -> dict[str, Any]:
    """Whether a side has ``flashscore_hash`` + history in ``global_matches.db``.

    Returns
    -------
    dict
        ``team``, ``db_team_id``, ``has_hash``, ``has_history``, ``ready``,
        ``needs_import``, ``hash``, ``slug``, ``match_count``.
    """
    from src.global_db import (
        GLOBAL_DB_PATH,
        connect_global_db,
        get_team_feed_meta,
        get_team_flashscore_meta,
        resolve_team_id,
    )

    label = str(team_id or "").strip()
    out: dict[str, Any] = {
        "team": label,
        "db_team_id": None,
        "has_hash": False,
        "has_history": False,
        "ready": False,
        "needs_import": True,
        "hash": None,
        "slug": None,
        "match_count": 0,
    }
    if not label:
        return out

    path = Path(db_path) if db_path is not None else GLOBAL_DB_PATH
    # Registry / DB hash resolve (does not require an open connection first).
    resolved = resolve_team_hash(label, db_path=path)
    if resolved and str(resolved.get("hash") or "").strip():
        out["has_hash"] = True
        out["hash"] = str(resolved["hash"]).strip()
        out["slug"] = str(resolved.get("slug") or "").strip() or None
        if resolved.get("db_team_id") is not None:
            out["db_team_id"] = int(resolved["db_team_id"])

    try:
        conn = connect_global_db(path, init=True)
    except Exception:  # noqa: BLE001
        out["needs_import"] = not (out["has_hash"] and out["has_history"])
        out["ready"] = bool(out["has_hash"] and out["has_history"])
        return out

    try:
        tid = out["db_team_id"]
        if tid is None:
            tid = resolve_team_id(
                conn, label, gender=gender, comp_id=comp_id, create=False
            )
        if tid is None and gender is None:
            tid = resolve_team_id(conn, label, gender="M", create=False)
            if tid is None:
                tid = resolve_team_id(conn, label, gender="W", create=False)
        if tid is None:
            # No DB row yet — still report missing hash/history for UI import.
            out["needs_import"] = True
            out["ready"] = False
            return out
        out["db_team_id"] = int(tid)

        meta = get_team_flashscore_meta(conn, int(tid))
        if meta and meta.get("hash"):
            out["has_hash"] = True
            out["hash"] = str(meta["hash"])
            out["slug"] = str(meta.get("slug") or "").strip() or out.get("slug")

        n = _count_team_matches_in_db(conn, int(tid))
        feed = get_team_feed_meta(conn, int(tid))
        last = (feed or {}).get("last_match_date") if feed else None
        out["match_count"] = int(n)
        out["has_history"] = bool(n > 0 or last)
    finally:
        conn.close()

    out["ready"] = bool(out["has_hash"] and out["has_history"])
    out["needs_import"] = not out["ready"]
    return out


def teams_needing_manual_import(
    home_team: str,
    away_team: str,
    *,
    db_path: Path | str | None = None,
    gender: str | None = None,
    comp_id: str | None = None,
) -> list[dict[str, Any]]:
    """Return status dicts for sides that lack hash and/or DB history."""
    missing: list[dict[str, Any]] = []
    for name in (home_team, away_team):
        label = str(name or "").strip()
        if not label:
            continue
        status = team_flashscore_data_status(
            label, db_path=db_path, gender=gender, comp_id=comp_id
        )
        if status.get("needs_import"):
            missing.append(status)
    return missing


def _recompute_rest_for_imported_team(
    conn: Any,
    db_team_id: int,
    team_code: str,
    *,
    db_path: Path | str,
    upcoming_kickoff: Any = None,
    upcoming_comp_id: str | None = None,
) -> dict[str, Any]:
    """Refresh rest_days / matches_last_14d for an upcoming fixture context."""
    from src.features import calculate_multi_comp_features
    from src.global_db import (
        read_matches_as_legacy,
        save_upcoming_to_db,
        set_team_rest_cache,
    )

    summary: dict[str, Any] = {"rest_days": None, "matches_last_14d": None}
    kick = upcoming_kickoff
    if kick is None or (isinstance(kick, float) and pd.isna(kick)):
        # Fall back to nearest upcoming fixture involving this team.
        try:
            upcoming = _list_upcoming_within_days(days_ahead=7, db_path=db_path)
        except Exception:  # noqa: BLE001
            upcoming = pd.DataFrame()
        if upcoming is not None and not upcoming.empty:
            code_u = str(team_code).strip().upper()
            for _, fx in upcoming.iterrows():
                home = str(fx.get("HomeTeam") or fx.get("home") or "").strip()
                away = str(fx.get("AwayTeam") or fx.get("away") or "").strip()
                if home.upper() != code_u and away.upper() != code_u:
                    continue
                kick = fx.get("Kickoff") if "Kickoff" in fx.index else fx.get("Date")
                if upcoming_comp_id is None:
                    upcoming_comp_id = str(
                        fx.get("league_id")
                        or fx.get("league")
                        or fx.get("comp_id")
                        or ""
                    ).strip().upper() or None
                break
    if kick is None or (isinstance(kick, float) and pd.isna(kick)):
        return summary

    kick_ts = pd.Timestamp(kick)
    ref_date = kick_ts.strftime("%Y-%m-%d")
    try:
        hist = read_matches_as_legacy(db_path)
    except Exception:  # noqa: BLE001
        hist = pd.DataFrame()

    feats = calculate_multi_comp_features(
        int(db_team_id),
        kick_ts,
        hist if hist is not None and not hist.empty else None,
        conn=conn if hist is None or hist.empty else None,
        upcoming_comp_id=upcoming_comp_id,
    )
    set_team_rest_cache(
        conn,
        int(db_team_id),
        ref_date,
        rest_days=feats.get("rest_days"),
        matches_last_14d=feats.get("matches_last_14d"),
    )
    summary["rest_days"] = feats.get("rest_days")
    summary["matches_last_14d"] = feats.get("matches_last_14d")
    summary["ref_date"] = ref_date

    # Patch upcoming_fixtures cache when this team appears.
    if upcoming_comp_id:
        try:
            from src.global_db import load_upcoming_from_db

            enriched, _ = load_upcoming_from_db(upcoming_comp_id, db_path=db_path)
            if enriched is not None and not enriched.empty:
                code_u = str(team_code).strip().upper()
                for col_side, rest_col, n14_col in (
                    ("HomeTeam", "home_rest_days", "home_matches_last_14d"),
                    ("AwayTeam", "away_rest_days", "away_matches_last_14d"),
                ):
                    if col_side not in enriched.columns:
                        continue
                    mask = enriched[col_side].astype(str).str.upper() == code_u
                    if not mask.any():
                        continue
                    if rest_col not in enriched.columns:
                        enriched[rest_col] = float("nan")
                    if n14_col not in enriched.columns:
                        enriched[n14_col] = 0.0
                    enriched.loc[mask, rest_col] = feats.get("rest_days")
                    enriched.loc[mask, n14_col] = feats.get("matches_last_14d")
                save_upcoming_to_db(upcoming_comp_id, enriched, db_path=db_path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("upcoming rest patch failed: %s", exc)
            summary["upcoming_patch_error"] = str(exc)

    return summary


def import_team_from_flashscore_url(
    team_id: str,
    url: str,
    *,
    n_matches: int = DEFAULT_N_MATCHES,
    db_path: Path | str | None = None,
    upcoming_kickoff: Any = None,
    upcoming_comp_id: str | None = None,
    timeout: float = DEFAULT_TIMEOUT_S,
) -> dict[str, Any]:
    """Upsert Flashscore hash from a pasted URL and fetch ~10 recent matches.

    Parameters
    ----------
    team_id:
        Club code / display name to attach the hash to (e.g. ``JP_VISSEL_KOBE``).
    url:
        Flashscore team page URL (``/team/{slug}/{hash}/``).
    n_matches:
        Recent finished games to scrape (default 10).
    upcoming_kickoff / upcoming_comp_id:
        Optional fixture context to recompute ``rest_days`` / ``matches_last_14d``.

    Returns
    -------
    dict
        Success: ``{"ok": True, "slug", "hash", "matches_fetched", ...}``.
        Failure: ``{"ok": False, "error": "..."}``.
    """
    from src.global_db import (
        GLOBAL_DB_PATH,
        connect_global_db,
        set_team_feed_meta,
        upsert_team_flashscore_hash,
    )

    label = str(team_id or "").strip()
    if not label:
        return {"ok": False, "error": "Thiếu mã/tên đội bóng."}

    try:
        parsed = parse_flashscore_team_url(url)
    except ValueError as exc:
        return {"ok": False, "error": str(exc)}

    slug = parsed["slug"]
    h = parsed["hash"]
    path = Path(db_path) if db_path is not None else GLOBAL_DB_PATH

    try:
        conn = connect_global_db(path, init=True)
        try:
            db_tid = upsert_team_flashscore_hash(
                conn,
                label,
                h,
                flashscore_slug=slug,
                comp_id=upcoming_comp_id,
            )
            conn.commit()
        finally:
            conn.close()
    except Exception as exc:  # noqa: BLE001
        logger.exception("upsert flashscore hash failed for %s", label)
        return {"ok": False, "error": f"Không lưu được hash vào DB: {exc}"}

    rows = fetch_team_recent_matches(
        h,
        n_matches=n_matches,
        slug=slug,
        team_id=label,
        timeout=float(timeout),
        retries=2,
    )
    persist_stats: dict[str, Any] = {}
    if rows:
        persist_stats = persist_team_matches_to_global_db(rows, db_path=path)

    last_date = None
    if rows:
        try:
            last_date = pd.Timestamp(rows[0]["match_date"]).strftime("%Y-%m-%d")
        except (TypeError, ValueError, KeyError):
            last_date = None

    now = datetime.now(timezone.utc)
    rest_info: dict[str, Any] = {}
    try:
        conn = connect_global_db(path, init=True)
        try:
            set_team_feed_meta(
                conn,
                int(db_tid),
                last_match_date=last_date,
                feed_updated_at=now,
            )
            rest_info = _recompute_rest_for_imported_team(
                conn,
                int(db_tid),
                label,
                db_path=path,
                upcoming_kickoff=upcoming_kickoff,
                upcoming_comp_id=upcoming_comp_id,
            )
            conn.commit()
        finally:
            conn.close()
    except Exception as exc:  # noqa: BLE001
        logger.warning("feed meta / rest recompute failed: %s", exc)
        rest_info = {"error": str(exc)}

    status = team_flashscore_data_status(label, db_path=path, comp_id=upcoming_comp_id)
    if not rows:
        return {
            "ok": False,
            "error": (
                "Đã lưu hash nhưng không lấy được trận gần đây từ Flashscore. "
                "Kiểm tra lại URL hoặc thử lại sau."
            ),
            "slug": slug,
            "hash": h,
            "db_team_id": int(db_tid),
            "matches_fetched": 0,
            "persist": persist_stats,
            "status": status,
        }

    return {
        "ok": True,
        "slug": slug,
        "hash": h,
        "db_team_id": int(db_tid),
        "matches_fetched": len(rows),
        "last_match_date": last_date,
        "persist": persist_stats,
        "rest": rest_info,
        "status": status,
    }
