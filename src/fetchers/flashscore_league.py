"""Flashscore-backed multi-league history / upcoming fixtures.

Reads season IDs from ``config/leagues.json`` via :mod:`src.league_registry`.
Reuses Flashscore HTML feed parsers and Odds GraphQL helpers from
:mod:`src.data_loader`.

Flashscore tournament result pages only embed a recent slice of finished
matches in SSR. When ``fd_div`` is set (e.g. ``SP1`` for LaLiga), full-season
history is loaded from football-data.co.uk and team names are mapped through
league-specific aliases. Upcoming (~48h) fixtures + odds always come from
Flashscore.
"""

from __future__ import annotations

import logging
import re
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence
from urllib.error import HTTPError, URLError

import pandas as pd

from src.data_loader import (
    FLASHSCORE_FSIGN,
    OPTIONAL_COLUMNS,
    REQUIRED_COLUMNS,
    TEAM_ALIASES,
    _flashscore_extract_initial_feed,
    _flashscore_parse_fields,
    _fetch_text,
    canonicalize_team_columns,
    clean_matches,
    fetch_flashscore_fixture_events,
    fetch_flashscore_match_odds,
    fetch_flashscore_odds_for_events,
    normalize_team_name,
    season_code,
)
from src.league_registry import (
    get_league_entry,
    resolve_league_config,
    season_starts_from_labels,
)

logger = logging.getLogger(__name__)

FLASHSCORE_BASE = "https://www.flashscore.com"
DEFAULT_UPCOMING_HOURS = 48
DEFAULT_RETRIES = 3
DEFAULT_TIMEOUT_S = 45.0
DEFAULT_RETRY_DELAY_S = 1.0

# Flashscore feed: PX/PY = team page hashes, WU/WV = URL slugs (JA/JB are NOT hashes).
_TEAM_URL_RE = re.compile(
    r"/team/([a-z0-9\-]+)/([A-Za-z0-9]{6,12})/",
    re.IGNORECASE,
)

# Built-in LaLiga aliases (also listed under LALIGA.team_aliases in JSON).
LALIGA_TEAM_ALIASES: dict[str, str] = {
    "Barcelona": "ES_BARCELONA",
    "FC Barcelona": "ES_BARCELONA",
    "Barca": "ES_BARCELONA",
    "Barça": "ES_BARCELONA",
    "Real Madrid": "ES_REAL_MADRID",
    "Atl. Madrid": "ES_ATLETICO",
    "Atletico Madrid": "ES_ATLETICO",
    "Atlético Madrid": "ES_ATLETICO",
    "Ath Bilbao": "ES_ATHLETIC",
    "Athletic Club": "ES_ATHLETIC",
    "Athletic Bilbao": "ES_ATHLETIC",
    "Real Sociedad": "ES_REAL_SOCIEDAD",
    "Real Betis": "ES_BETIS",
    "Betis": "ES_BETIS",
    "Sevilla": "ES_SEVILLA",
    "Villarreal": "ES_VILLARREAL",
    "Valencia": "ES_VALENCIA",
    "Osasuna": "ES_OSASUNA",
    "Celta Vigo": "ES_CELTA",
    "Celta": "ES_CELTA",
    "Getafe": "ES_GETAFE",
    "Girona": "ES_GIRONA",
    "Mallorca": "ES_MALLORCA",
    "Rayo Vallecano": "ES_RAYO",
    "Espanyol": "ES_ESPANYOL",
    "Alaves": "ES_ALAVES",
    "Alavés": "ES_ALAVES",
    "Las Palmas": "ES_LAS_PALMAS",
    "Leganes": "ES_LEGANES",
    "Leganés": "ES_LEGANES",
    "Valladolid": "ES_VALLADOLID",
    "Elche": "ES_ELCHE",
    "Levante": "ES_LEVANTE",
    "Oviedo": "ES_OVIEDO",
    "Real Oviedo": "ES_OVIEDO",
}

# football-data.co.uk short names → LaLiga stable codes.
_FD_LALIGA_ALIASES: dict[str, str] = {
    "Barcelona": "ES_BARCELONA",
    "Real Madrid": "ES_REAL_MADRID",
    "Ath Madrid": "ES_ATLETICO",
    "Ath Bilbao": "ES_ATHLETIC",
    "Sociedad": "ES_REAL_SOCIEDAD",
    "Betis": "ES_BETIS",
    "Sevilla": "ES_SEVILLA",
    "Villarreal": "ES_VILLARREAL",
    "Valencia": "ES_VALENCIA",
    "Osasuna": "ES_OSASUNA",
    "Celta": "ES_CELTA",
    "Getafe": "ES_GETAFE",
    "Girona": "ES_GIRONA",
    "Mallorca": "ES_MALLORCA",
    "Vallecano": "ES_RAYO",
    "Espanol": "ES_ESPANYOL",
    "Alaves": "ES_ALAVES",
    "Las Palmas": "ES_LAS_PALMAS",
    "Leganes": "ES_LEGANES",
    "Valladolid": "ES_VALLADOLID",
    "Elche": "ES_ELCHE",
    "Levante": "ES_LEVANTE",
    "Oviedo": "ES_OVIEDO",
}

# Built-in J-League / Emperor's Cup aliases → JP_* stable codes.
# Also listed under EMPERORS_CUP.team_aliases in config/leagues.json.
EMPERORS_CUP_TEAM_ALIASES: dict[str, str] = {
    "Gamba Osaka": "JP_G_OSAKA",
    "Vissel Kobe": "JP_VISSEL_KOBE",
    "Kashima Antlers": "JP_KASHIMA",
    "Kashima": "JP_KASHIMA",
    "Urawa Reds": "JP_URAWA",
    "Urawa": "JP_URAWA",
    "Kawasaki Frontale": "JP_KAWASAKI",
    "Kawasaki": "JP_KAWASAKI",
    "Yokohama F. Marinos": "JP_YOKOHAMA_FM",
    "Yokohama Marinos": "JP_YOKOHAMA_FM",
    "Yokohama FC": "JP_YOKOHAMA_FC",
    "FC Tokyo": "JP_FC_TOKYO",
    "Tokyo": "JP_FC_TOKYO",
    "Cerezo Osaka": "JP_CEREZO_OSAKA",
    "Sanfrecce Hiroshima": "JP_SANFRECCE",
    "Hiroshima": "JP_SANFRECCE",
    "Kashiwa Reysol": "JP_KASHIWA",
    "Kashiwa": "JP_KASHIWA",
    "Nagoya Grampus": "JP_NAGOYA",
    "Nagoya": "JP_NAGOYA",
    "Avispa Fukuoka": "JP_AVISPA_FUKUOKA",
    "Fukuoka": "JP_AVISPA_FUKUOKA",
    "Shonan Bellmare": "JP_SHONAN",
    "Shonan": "JP_SHONAN",
    "Shimizu S-Pulse": "JP_SHIMIZU",
    "Shimizu": "JP_SHIMIZU",
    "Kyoto": "JP_KYOTO",
    "Kyoto Sanga": "JP_KYOTO",
    "Machida": "JP_MACHIDA",
    "Machida Zelvia": "JP_MACHIDA",
    "Verdy": "JP_VERDY",
    "Tokyo Verdy": "JP_VERDY",
    "Sagan Tosu": "JP_SAGAN_TOSU",
    "Tosu": "JP_SAGAN_TOSU",
    "Kofu": "JP_KOFU",
    "Ventforet Kofu": "JP_KOFU",
    "Omiya Ardija": "JP_OMIYA",
    "Omiya": "JP_OMIYA",
    "Okayama": "JP_OKAYAMA",
    "Fagiano Okayama": "JP_OKAYAMA",
    "V-Varen Nagasaki": "JP_V_VAREN",
    "Nagasaki": "JP_V_VAREN",
    "Tokushima": "JP_TOKUSHIMA",
    "Tokushima Vortis": "JP_TOKUSHIMA",
    "Iwaki": "JP_IWAKI",
    "Iwaki FC": "JP_IWAKI",
    "Imabari": "JP_IMABARI",
    "FC Imabari": "JP_IMABARI",
    "Fujieda MYFC": "JP_FUJIEDA",
    "Fujieda": "JP_FUJIEDA",
    "Kagoshima Utd": "JP_KAGOSHIMA",
    "Kagoshima United": "JP_KAGOSHIMA",
    "Tochigi City": "JP_TOCHIGI_CITY",
    "Gainare Tottori": "JP_GAINARE",
    "Tottori": "JP_GAINARE",
    "Tegevajaro Miyazaki": "JP_TEGEVAJARO",
    "Miyazaki": "JP_TEGEVAJARO",
    "Chiba": "JP_CHIBA",
    "JEF United": "JP_CHIBA",
    "JEF United Chiba": "JP_CHIBA",
    "Kyoto Sangyo": "JP_KYOTO_SANGYO",
    "Albirex Niigata": "JP_NIIGATA",
    "Niigata": "JP_NIIGATA",
    "Consadole Sapporo": "JP_SAPPORO",
    "Sapporo": "JP_SAPPORO",
    "Jubilo Iwata": "JP_IWATA",
    "Iwata": "JP_IWATA",
}

# Country / competition stable-code prefixes registered into TEAM_ALIASES.
_STABLE_TEAM_PREFIXES: tuple[str, ...] = ("ES_", "JP_")


def _empty_history_frame() -> pd.DataFrame:
    cols = list(
        dict.fromkeys(
            REQUIRED_COLUMNS
            + ["Season", "SeasonStart", "FlashscoreEventId", "Kickoff", "Source", "league_id"]
            + [c for c in OPTIONAL_COLUMNS if c not in REQUIRED_COLUMNS]
        )
    )
    return pd.DataFrame(columns=cols)


def _empty_upcoming_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "Date",
            "Kickoff",
            "HomeTeam",
            "AwayTeam",
            "FlashscoreEventId",
            "Source",
            "league_id",
            "B365H",
            "B365D",
            "B365A",
            "OddsProvider",
        ]
    )


def league_team_aliases(league_key: str) -> dict[str, str]:
    """Merge global ``TEAM_ALIASES``, built-ins, and JSON ``team_aliases``."""
    code = str(league_key or "").strip().upper()
    out = dict(TEAM_ALIASES)
    if code == "LALIGA":
        out.update(LALIGA_TEAM_ALIASES)
        out.update(_FD_LALIGA_ALIASES)
    elif code in {"EMPERORS_CUP", "J_LEAGUE_CUP", "J1", "J2", "JLEAGUE"}:
        out.update(EMPERORS_CUP_TEAM_ALIASES)
    entry = get_league_entry(code) or {}
    extra = entry.get("team_aliases") or {}
    if isinstance(extra, dict):
        out.update({str(k): str(v) for k, v in extra.items()})
    return out


def resolve_league_team_name(name: str, league_key: str) -> str:
    """Map a scraped / FD team name to the league's stable code."""
    aliases = league_team_aliases(league_key)
    raw = str(name or "").strip()
    if not raw:
        return raw
    if raw in aliases:
        return aliases[raw]
    # Accent-folded fallback via normalize_team_name on a temporary alias map.
    mapped = aliases.get(raw)
    if mapped:
        return mapped
    # Try global normalizer first, then league map again.
    base = normalize_team_name(raw, known_teams=None)
    return aliases.get(base, aliases.get(raw, base))


def apply_league_team_aliases(df: pd.DataFrame, league_key: str) -> pd.DataFrame:
    """Rewrite HomeTeam / AwayTeam through :func:`resolve_league_team_name`."""
    if df.empty:
        return df.copy()
    out = df.copy()
    for col in ("HomeTeam", "AwayTeam"):
        if col in out.columns:
            out[col] = out[col].map(lambda x, lk=league_key: resolve_league_team_name(x, lk))
    return canonicalize_team_columns(out)


def _is_stable_team_code(canon: str, league_key: str) -> bool:
    """True for country prefixes (``ES_*``, ``JP_*``) or ``{LEAGUE}_*`` codes."""
    code = str(league_key or "").strip().upper()
    c = str(canon or "")
    if any(c.startswith(p) for p in _STABLE_TEAM_PREFIXES):
        return True
    return bool(code) and c.startswith(f"{code}_")


def register_league_aliases_globally(league_key: str) -> None:
    """Register stable codes into ``TEAM_ALIASES`` without clobbering UWCL names.

    Only writes identity mappings for ``ES_*`` / ``JP_*`` (or ``{LEAGUE}_*``)
    codes and aliases that already target those codes when missing.
    League-specific rewriting still happens via
    :func:`apply_league_team_aliases` before persist.
    """
    aliases = league_team_aliases(league_key)
    for alias, canon in aliases.items():
        if not canon:
            continue
        if _is_stable_team_code(canon, league_key):
            TEAM_ALIASES.setdefault(canon, canon)
            # Prefer league code when alias is unused; never overwrite UWCL maps.
            TEAM_ALIASES.setdefault(alias, canon)


def _flashscore_url(path: str, *, page: str, season_label: str | None = None) -> str:
    base = path if path.startswith("http") else f"{FLASHSCORE_BASE}{path}"
    base = base.rstrip("/")
    if season_label:
        # /football/spain/laliga + 2024-2025 → .../laliga-2024-2025/results/
        return f"{base}-{season_label}/{page.strip('/')}/"
    return f"{base}/{page.strip('/')}/"


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


def extract_team_hashes_from_html(html: str) -> list[dict[str, str]]:
    """Parse ``/team/{slug}/{hash}/`` URLs from page HTML."""
    out: list[dict[str, str]] = []
    seen: set[str] = set()
    for slug, h in _TEAM_URL_RE.findall(str(html or "")):
        key = str(h).strip()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(
            {
                "hash": key,
                "slug": str(slug).strip().strip("/"),
                "name": "",
            }
        )
    return out


def extract_team_hashes_from_feed_fields(
    fields: dict[str, str],
) -> list[dict[str, str]]:
    """Extract home/away ``{hash, slug, name}`` from one Flashscore event section.

    Uses ``PX``/``PY`` (team page hashes) + ``WU``/``WV`` (slugs). ``JA``/``JB``
    are image/participant ids and must not be used for ``/team/`` URLs.
    """
    out: list[dict[str, str]] = []
    home_hash = str(fields.get("PX") or "").strip()
    away_hash = str(fields.get("PY") or "").strip()
    home_slug = str(fields.get("WU") or "").strip().strip("/")
    away_slug = str(fields.get("WV") or "").strip().strip("/")
    home_name = str(
        fields.get("AE") or fields.get("FH") or fields.get("CX") or ""
    ).strip()
    away_name = str(fields.get("AF") or fields.get("FK") or "").strip()
    if home_hash:
        out.append({"hash": home_hash, "slug": home_slug, "name": home_name})
    if away_hash:
        out.append({"hash": away_hash, "slug": away_slug, "name": away_name})
    return out


def extract_team_hashes_from_feed(raw: str) -> list[dict[str, str]]:
    """Parse all team hashes from a fixtures/results SSR feed body."""
    by_hash: dict[str, dict[str, str]] = {}
    for sec in str(raw or "").split("~"):
        fields = _flashscore_parse_fields(sec)
        if not fields:
            continue
        for entry in extract_team_hashes_from_feed_fields(fields):
            h = entry["hash"]
            prev = by_hash.get(h)
            if prev is None:
                by_hash[h] = entry
                continue
            # Prefer entries that carry a display name / slug.
            if not prev.get("name") and entry.get("name"):
                prev["name"] = entry["name"]
            if not prev.get("slug") and entry.get("slug"):
                prev["slug"] = entry["slug"]
    return list(by_hash.values())


def persist_team_hashes(
    entries: Sequence[dict[str, str]],
    league_key: str,
    *,
    db_path: Path | str | None = None,
) -> dict[str, Any]:
    """Upsert extracted Flashscore hashes into ``teams.flashscore_hash``.

    Resolves display names through league aliases (``JP_*`` / ``ES_*``) before
    linking to ``team_id``.
    """
    from src.global_db import GLOBAL_DB_PATH, connect_global_db, upsert_team_flashscore_hash

    code = str(league_key or "").strip().upper()
    register_league_aliases_globally(code)
    path = Path(db_path) if db_path is not None else GLOBAL_DB_PATH
    stats: dict[str, Any] = {
        "comp_id": code,
        "seen": 0,
        "upserted": 0,
        "skipped": 0,
    }
    if not entries:
        return stats

    conn = connect_global_db(path, init=True)
    try:
        for entry in entries:
            h = str(entry.get("hash") or "").strip()
            if not h:
                stats["skipped"] += 1
                continue
            stats["seen"] += 1
            raw_name = str(entry.get("name") or "").strip()
            slug = str(entry.get("slug") or "").strip().strip("/") or None
            if raw_name:
                resolved = resolve_league_team_name(raw_name, code)
            elif slug:
                # Fallback: slug → Title Case guess (rare; HTML-only links).
                resolved = resolve_league_team_name(
                    slug.replace("-", " ").title(), code
                )
            else:
                stats["skipped"] += 1
                continue
            if not resolved:
                stats["skipped"] += 1
                continue
            upsert_team_flashscore_hash(
                conn,
                resolved,
                h,
                flashscore_slug=slug,
                comp_id=code,
            )
            stats["upserted"] += 1
        conn.commit()
    finally:
        conn.close()
    return stats


def scrape_and_persist_league_team_hashes(
    league_key: str,
    *,
    pages: Sequence[str] = ("fixtures", "results"),
    db_path: Path | str | None = None,
) -> dict[str, Any]:
    """Fetch league fixtures/results SSR pages and upsert team hashes.

    Used when upcoming sides lack ``flashscore_hash`` — no hardcoding required.
    """
    warnings_list: list[str] = []
    try:
        code, cfg = resolve_league_config(league_key)
    except ValueError as exc:
        return {"comp_id": str(league_key).upper(), "error": str(exc), "upserted": 0}

    entry = get_league_entry(code) or {}
    path = str(cfg.get("flashscore_path") or entry.get("flashscore_path") or "")
    if not path:
        return {
            "comp_id": code,
            "error": "no flashscore_path",
            "upserted": 0,
        }

    all_entries: dict[str, dict[str, str]] = {}
    for page in pages:
        try:
            if page == "fixtures" and cfg.get("flashscore_fixtures_url"):
                url = str(cfg["flashscore_fixtures_url"])
            else:
                url = _flashscore_url(path, page=page)
            html = _fetch_text_retry(url)
        except Exception as exc:  # noqa: BLE001
            warnings_list.append(f"{page}: {exc}")
            logger.warning("hash scrape %s %s: %s", code, page, exc)
            continue

        feed_key = "summary-fixtures" if page == "fixtures" else "summary-results"
        raw = _flashscore_extract_initial_feed(html, feed_key)
        if not raw and page == "results":
            raw = _flashscore_extract_initial_feed(html, "results")
        for item in extract_team_hashes_from_feed(raw):
            all_entries[item["hash"]] = item
        for item in extract_team_hashes_from_html(html):
            prev = all_entries.get(item["hash"])
            if prev is None:
                all_entries[item["hash"]] = item
            elif not prev.get("slug") and item.get("slug"):
                prev["slug"] = item["slug"]

    persist = persist_team_hashes(list(all_entries.values()), code, db_path=db_path)
    persist["warnings"] = warnings_list
    persist["hashes_found"] = len(all_entries)
    return persist


def _events_from_feed(
    raw: str,
    *,
    league_key: str,
    only_finished: bool | None = None,
    only_unplayed: bool | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sec in raw.split("~"):
        fields = _flashscore_parse_fields(sec)
        if "AA" not in fields or "AE" not in fields:
            continue
        status = str(fields.get("AB", ""))
        # Flashscore: 1=scheduled, 2=live, 3=finished
        if only_finished is True and status != "3":
            continue
        if only_unplayed is True and status == "3":
            continue
        try:
            ts = int(fields.get("AD") or 0)
        except (TypeError, ValueError):
            ts = 0
        if not ts:
            continue
        kickoff = datetime.fromtimestamp(ts, tz=timezone.utc).replace(tzinfo=None)
        home = resolve_league_team_name(fields.get("AE", ""), league_key)
        away = resolve_league_team_name(fields.get("AF", ""), league_key)
        home_hash = str(fields.get("PX") or "").strip() or None
        away_hash = str(fields.get("PY") or "").strip() or None
        home_slug = str(fields.get("WU") or "").strip().strip("/") or None
        away_slug = str(fields.get("WV") or "").strip().strip("/") or None
        row: dict[str, Any] = {
            "Date": pd.Timestamp(kickoff).normalize(),
            "Kickoff": pd.Timestamp(kickoff),
            "HomeTeam": home,
            "AwayTeam": away,
            "FlashscoreEventId": fields["AA"],
            "Source": "flashscore",
            "league_id": str(league_key).upper(),
            "HomeFlashscoreHash": home_hash,
            "AwayFlashscoreHash": away_hash,
            "HomeFlashscoreSlug": home_slug,
            "AwayFlashscoreSlug": away_slug,
        }
        if status == "3":
            try:
                hg = int(fields.get("AG", ""))
                ag = int(fields.get("AH", ""))
            except (TypeError, ValueError):
                continue
            row["FTHG"] = hg
            row["FTAG"] = ag
            row["FTR"] = "H" if hg > ag else ("A" if hg < ag else "D")
        rows.append(row)
    return rows


def fetch_flashscore_season_results(
    league_key: str,
    season_label: str,
    *,
    flashscore_path: str,
) -> pd.DataFrame:
    """Parse finished matches from a season results HTML page."""
    url = _flashscore_url(flashscore_path, page="results", season_label=season_label)
    html = _fetch_text_retry(url)
    raw = _flashscore_extract_initial_feed(html, "summary-results")
    if not raw:
        return _empty_history_frame()
    rows = _events_from_feed(raw, league_key=league_key, only_finished=True)
    if not rows:
        return _empty_history_frame()
    # Persist team hashes discovered on this page (dynamic; no hardcoding).
    try:
        hash_entries = extract_team_hashes_from_feed(raw)
        hash_entries.extend(extract_team_hashes_from_html(html))
        if hash_entries:
            persist_team_hashes(hash_entries, league_key)
    except Exception as exc:  # noqa: BLE001
        logger.warning("persist hashes from results %s: %s", league_key, exc)
    label = str(season_label).strip()
    start_year = int(label.split("-")[0])
    # Calendar-year cups (Emperor's Cup "2026") keep a single-year Season label;
    # Aug–May leagues ("2024-2025") use football-data style "2024/25".
    if "-" in label:
        season_str = f"{start_year}/{str(start_year + 1)[-2:]}"
    else:
        season_str = label
    for row in rows:
        row["Season"] = season_str
        row["SeasonStart"] = start_year
    return pd.DataFrame(rows)


def _download_fd_div_seasons(
    fd_div: str,
    season_starts: Sequence[int],
    *,
    league_key: str,
) -> tuple[pd.DataFrame, list[str]]:
    """Download football-data.co.uk seasons for an arbitrary division code."""
    from io import StringIO

    from urllib.request import Request, urlopen

    from src.data_loader import USER_AGENT

    frames: list[pd.DataFrame] = []
    errors: list[str] = []
    for start in season_starts:
        code = season_code(int(start))
        url = f"https://www.football-data.co.uk/mmz4281/{code}/{fd_div}.csv"
        try:
            request = Request(
                url, headers={"User-Agent": USER_AGENT, "Accept": "text/csv"}
            )
            with urlopen(request, timeout=45) as response:
                charset = response.headers.get_content_charset() or "utf-8"
                text = response.read().decode(charset, errors="replace")
            df = pd.read_csv(StringIO(text))
            meta = pd.DataFrame(
                {
                    "Season": f"{start}/{str(start + 1)[-2:]}",
                    "SeasonStart": int(start),
                },
                index=df.index,
            )
            frames.append(pd.concat([df, meta], axis=1))
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{fd_div} {start}/{start + 1}: {exc}")
            logger.warning("FD download failed %s: %s", url, exc)

    if not frames:
        return _empty_history_frame(), errors

    cleaned = clean_matches(pd.concat(frames, ignore_index=True))
    cleaned["league_id"] = str(league_key).upper()
    cleaned["Source"] = "football-data"
    cleaned = apply_league_team_aliases(cleaned, league_key)
    return cleaned, errors


def fetch_league_history(
    league_key: str,
    n_seasons: int = 2,
    *,
    include_flashscore_results: bool = True,
    prefer_fd: bool = True,
) -> pd.DataFrame:
    """Fetch finished matches for ``league_key`` (JSON season IDs / fd_div).

    Parameters
    ----------
    league_key:
        Registry key, e.g. ``LALIGA``.
    n_seasons:
        Number of most-recent seasons from the JSON ``seasons`` map (or
        calendar default when the map is empty and ``fd_div`` is set).
    include_flashscore_results:
        Also scrape Flashscore season results pages (recent FT scores).
    prefer_fd:
        When ``fd_div`` is configured, use football-data as the primary
        historical corpus (complete seasons + corners/odds).

    Returns
    -------
    pd.DataFrame
        Cleaned match table. On total failure returns an empty frame and
        records warnings on ``attrs['download_warnings']``.
    """
    warnings_list: list[str] = []
    try:
        code, cfg = resolve_league_config(league_key)
    except ValueError as exc:
        warnings_list.append(str(exc))
        out = _empty_history_frame()
        out.attrs["download_warnings"] = warnings_list
        return out

    register_league_aliases_globally(code)
    entry = get_league_entry(code) or {}
    seasons_map = dict(cfg.get("seasons") or entry.get("seasons") or {})
    season_rows = season_starts_from_labels(seasons_map, n_seasons=n_seasons)
    fd_div = cfg.get("fd_div")
    path = str(cfg.get("flashscore_path") or entry.get("flashscore_path") or "")

    frames: list[pd.DataFrame] = []

    # 1) Full history via football-data when available.
    if prefer_fd and fd_div:
        if season_rows:
            starts = [s[2] for s in season_rows]
        else:
            from src.data_loader import default_season_starts

            starts = list(default_season_starts(n_seasons))
        try:
            fd_df, fd_errors = _download_fd_div_seasons(
                str(fd_div), starts, league_key=code
            )
            warnings_list.extend(fd_errors)
            if not fd_df.empty:
                frames.append(fd_df)
        except Exception as exc:  # noqa: BLE001
            msg = f"football-data history failed: {exc}"
            warnings_list.append(msg)
            logger.warning(msg)

    # 2) Flashscore season results (recent slice / enrichment).
    if include_flashscore_results and path and season_rows:
        for label, _sid, start in season_rows:
            try:
                fs_df = fetch_flashscore_season_results(
                    code, label, flashscore_path=path
                )
                if fs_df.empty:
                    warnings_list.append(
                        f"Flashscore results empty for {code} {label}"
                    )
                else:
                    frames.append(fs_df)
            except Exception as exc:  # noqa: BLE001
                msg = f"Flashscore results {code} {label}: {exc}"
                warnings_list.append(msg)
                logger.warning(msg)
    elif include_flashscore_results and path and not season_rows:
        # Current season page (no season labels configured).
        try:
            url = _flashscore_url(path, page="results")
            html = _fetch_text_retry(url)
            raw = _flashscore_extract_initial_feed(html, "summary-results")
            rows = _events_from_feed(raw, league_key=code, only_finished=True)
            if rows:
                frames.append(pd.DataFrame(rows))
        except Exception as exc:  # noqa: BLE001
            warnings_list.append(f"Flashscore current results: {exc}")

    if not frames:
        warn = (
            f"No history for {code}; returning empty frame. "
            + ("; ".join(warnings_list) if warnings_list else "check network / config")
        )
        warnings.warn(warn, stacklevel=2)
        out = _empty_history_frame()
        out.attrs["download_warnings"] = warnings_list or [warn]
        out.attrs["league"] = code
        return out

    combined = pd.concat(frames, ignore_index=True, sort=False)
    # Prefer football-data rows when duplicating the same fixture date/teams.
    if "Source" in combined.columns:
        combined["_src_rank"] = combined["Source"].map(
            {"football-data": 0, "flashscore": 1}
        ).fillna(2)
        combined = combined.sort_values("_src_rank")
    subset = ["Date", "HomeTeam", "AwayTeam"]
    if all(c in combined.columns for c in subset):
        combined = combined.drop_duplicates(subset=subset, keep="first")
    combined = combined.drop(columns=["_src_rank"], errors="ignore")

    # Keep rows that have FT scores (history only).
    if "FTHG" in combined.columns and "FTAG" in combined.columns:
        combined = combined.dropna(subset=["FTHG", "FTAG"])
        combined["FTHG"] = combined["FTHG"].astype(int)
        combined["FTAG"] = combined["FTAG"].astype(int)
        if "FTR" not in combined.columns or combined["FTR"].isna().any():
            combined["FTR"] = [
                "H" if h > a else ("A" if h < a else "D")
                for h, a in zip(combined["FTHG"], combined["FTAG"])
            ]

    combined["league_id"] = code
    combined = apply_league_team_aliases(combined, code)
    combined = combined.sort_values(["Date", "HomeTeam", "AwayTeam"]).reset_index(
        drop=True
    )
    combined.attrs["download_warnings"] = warnings_list
    combined.attrs["league"] = code
    combined.attrs["data_source"] = (
        "football-data+flashscore" if fd_div and prefer_fd else "flashscore"
    )
    return combined


def fetch_league_upcoming(
    league_key: str,
    *,
    within_hours: float = DEFAULT_UPCOMING_HOURS,
    include_odds: bool = True,
    max_events: int = 24,
    request_delay_s: float = 0.25,
    now: datetime | None = None,
) -> pd.DataFrame:
    """Upcoming fixtures within ``within_hours`` (+ Flashscore odds when possible)."""
    warnings_list: list[str] = []
    try:
        code, cfg = resolve_league_config(league_key)
    except ValueError as exc:
        warnings_list.append(str(exc))
        out = _empty_upcoming_frame()
        out.attrs["download_warnings"] = warnings_list
        return out

    register_league_aliases_globally(code)
    entry = get_league_entry(code) or {}
    path = str(cfg.get("flashscore_path") or entry.get("flashscore_path") or "")
    fixtures_url = cfg.get("flashscore_fixtures_url")
    if not fixtures_url and path:
        fixtures_url = _flashscore_url(path, page="fixtures")
    if not fixtures_url:
        warnings_list.append(f"No flashscore fixtures URL for {code}")
        out = _empty_upcoming_frame()
        out.attrs["download_warnings"] = warnings_list
        return out

    try:
        events = fetch_flashscore_fixture_events(
            fixtures_url=str(fixtures_url),
            only_unplayed=True,
            normalize_women=(code == "UWCL"),
        )
    except Exception as exc:  # noqa: BLE001
        warnings_list.append(f"fixtures fetch failed: {exc}")
        logger.warning("fixtures fetch failed for %s: %s", code, exc)
        out = _empty_upcoming_frame()
        out.attrs["download_warnings"] = warnings_list
        return out

    if events.empty:
        out = _empty_upcoming_frame()
        out.attrs["download_warnings"] = warnings_list + ["no upcoming fixtures"]
        return out

    events = apply_league_team_aliases(events, code)
    events["league_id"] = code

    # Upsert Flashscore team hashes discovered on the fixtures feed.
    try:
        hash_entries: list[dict[str, str]] = []
        if {
            "HomeFlashscoreHash",
            "AwayFlashscoreHash",
        }.issubset(events.columns):
            for _, ev in events.iterrows():
                hh = str(ev.get("HomeFlashscoreHash") or "").strip()
                ah = str(ev.get("AwayFlashscoreHash") or "").strip()
                if hh:
                    hash_entries.append(
                        {
                            "hash": hh,
                            "slug": str(ev.get("HomeFlashscoreSlug") or "").strip(),
                            "name": str(ev.get("HomeTeam") or ""),
                        }
                    )
                if ah:
                    hash_entries.append(
                        {
                            "hash": ah,
                            "slug": str(ev.get("AwayFlashscoreSlug") or "").strip(),
                            "name": str(ev.get("AwayTeam") or ""),
                        }
                    )
        if hash_entries:
            persist_team_hashes(hash_entries, code)
    except Exception as exc:  # noqa: BLE001
        warnings_list.append(f"hash persist: {exc}")
        logger.warning("hash persist failed for %s: %s", code, exc)

    ref = now or datetime.now(timezone.utc).replace(tzinfo=None)
    ref_ts = pd.Timestamp(ref)
    horizon = ref_ts + pd.Timedelta(hours=float(within_hours))
    kick = pd.to_datetime(events["Kickoff"], errors="coerce")
    mask = (kick >= ref_ts) & (kick <= horizon)
    events = events.loc[mask].copy()
    if events.empty:
        out = _empty_upcoming_frame()
        out.attrs["download_warnings"] = warnings_list + [
            f"no fixtures within {within_hours}h"
        ]
        out.attrs["league"] = code
        return out

    events = events.head(int(max_events)).reset_index(drop=True)

    if include_odds and "FlashscoreEventId" in events.columns:
        odds_rows: list[dict[str, Any]] = []
        event_ids = [str(x) for x in events["FlashscoreEventId"].tolist()]
        try:
            odds_by_id = fetch_flashscore_odds_for_events(event_ids, max_workers=5)
        except Exception as exc:  # noqa: BLE001
            warnings_list.append(f"parallel odds fetch failed: {exc}")
            odds_by_id = {}
            # Serial fallback with delay (preserve retries/timeouts per match).
            for idx, (_, ev) in enumerate(events.iterrows()):
                if idx and request_delay_s > 0:
                    time.sleep(request_delay_s)
                try:
                    odds = fetch_flashscore_match_odds(str(ev["FlashscoreEventId"]))
                except Exception as inner:  # noqa: BLE001
                    warnings_list.append(
                        f"odds {ev.get('FlashscoreEventId')}: {inner}"
                    )
                    continue
                if odds:
                    odds_by_id[str(ev["FlashscoreEventId"])] = odds

        for _, ev in events.iterrows():
            eid = str(ev["FlashscoreEventId"])
            odds = odds_by_id.get(eid)
            if not odds:
                continue
            odds_rows.append({"FlashscoreEventId": eid, **odds})
        if odds_rows:
            odds_df = pd.DataFrame(odds_rows)
            events = events.merge(odds_df, on="FlashscoreEventId", how="left")

    events.attrs["download_warnings"] = warnings_list
    events.attrs["league"] = code
    events.attrs["data_source"] = "flashscore"
    return events


def persist_league_matches(
    league_key: str,
    history: pd.DataFrame,
    *,
    also_legacy: bool = True,
) -> dict[str, Any]:
    """Write finished matches into ``global_matches.db`` (+ optional legacy DB)."""
    from src.data_loader import save_matches_to_db
    from src.global_db import (
        GLOBAL_DB_PATH,
        connect_global_db,
        import_legacy_matches_df,
        upsert_competition,
    )

    code, cfg = resolve_league_config(league_key)
    register_league_aliases_globally(code)
    weight = float(cfg.get("league_weight") or 1.0)
    name = str(cfg.get("label") or code)

    stats: dict[str, Any] = {
        "comp_id": code,
        "global": {},
        "legacy_rows": 0,
    }
    if history is None or history.empty:
        stats["skipped"] = True
        return stats

    work = apply_league_team_aliases(history, code)
    work["league_id"] = code

    conn = connect_global_db(GLOBAL_DB_PATH, init=True)
    try:
        upsert_competition(conn, code, name, weight)
        stats["global"] = import_legacy_matches_df(conn, work, comp_id=code)
        conn.commit()
    finally:
        conn.close()

    if also_legacy and cfg.get("db_path"):
        try:
            stats["legacy_rows"] = save_matches_to_db(work, cfg["db_path"])  # type: ignore[arg-type]
        except Exception as exc:  # noqa: BLE001
            stats["legacy_error"] = str(exc)
            logger.warning("legacy save failed for %s: %s", code, exc)

    return stats


def persist_league_upcoming(
    league_key: str,
    upcoming: pd.DataFrame,
) -> dict[str, Any]:
    """Write upcoming fixtures (+ odds) into ``global_matches.upcoming_fixtures``."""
    from src.global_db import (
        GLOBAL_DB_PATH,
        connect_global_db,
        save_upcoming_to_db,
        upsert_competition,
    )

    code, cfg = resolve_league_config(league_key)
    register_league_aliases_globally(code)
    weight = float(cfg.get("league_weight") or 1.0)
    name = str(cfg.get("label") or code)

    stats: dict[str, Any] = {"comp_id": code, "upcoming_rows": 0}
    work = upcoming
    if work is not None and not work.empty:
        work = apply_league_team_aliases(work, code)
        work = work.copy()
        work["league_id"] = code

    conn = connect_global_db(GLOBAL_DB_PATH, init=True)
    try:
        upsert_competition(conn, code, name, weight)
        conn.commit()
    finally:
        conn.close()

    n = save_upcoming_to_db(code, work if work is not None else pd.DataFrame())
    stats["upcoming_rows"] = int(n)
    return stats


def fetch_and_persist_league(
    league_key: str,
    n_seasons: int = 2,
    *,
    upcoming_hours: float = DEFAULT_UPCOMING_HOURS,
    dry_run: bool = False,
    fit_models: bool = False,
) -> dict[str, Any]:
    """Convenience: history + upcoming → global DB unless ``dry_run``.

    When ``fit_models`` is True and history is non-empty, also fit Dixon–Coles
    (+ LightGBM when sample allows) via :func:`src.models.fit_league_models`.
    """
    history = fetch_league_history(league_key, n_seasons=n_seasons)
    upcoming = fetch_league_upcoming(league_key, within_hours=upcoming_hours)
    result: dict[str, Any] = {
        "league": str(league_key).upper(),
        "history_rows": int(len(history)),
        "upcoming_rows": int(len(upcoming)),
        "warnings": list(history.attrs.get("download_warnings") or [])
        + list(upcoming.attrs.get("download_warnings") or []),
        "dry_run": dry_run,
    }
    if dry_run:
        if not history.empty:
            result["history_sample"] = history.head(3)[
                [c for c in ("Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG") if c in history.columns]
            ].to_dict(orient="records")
        if not upcoming.empty:
            cols = [
                c
                for c in (
                    "Kickoff",
                    "HomeTeam",
                    "AwayTeam",
                    "B365H",
                    "B365D",
                    "B365A",
                )
                if c in upcoming.columns
            ]
            result["upcoming_sample"] = upcoming.head(5)[cols].to_dict(orient="records")
        return result

    result["persist"] = persist_league_matches(league_key, history)
    result["persist_upcoming"] = persist_league_upcoming(league_key, upcoming)

    if fit_models and not history.empty:
        try:
            from src.models import fit_league_models

            fitted = fit_league_models(history, league=str(league_key).upper())
            result["models"] = {
                "n_train": int(fitted.n_train),
                "w_ml": float(fitted.w_ml),
                "ml_ok": fitted.ml is not None,
                "corner_ok": fitted.corner is not None,
                "warnings": list(fitted.warnings),
                "ml_error": fitted.ml_error,
            }
        except Exception as exc:  # noqa: BLE001
            result["models"] = {"ok": False, "error": str(exc)}
            logger.warning("fit_models failed for %s: %s", league_key, exc)

    return result


# ---------------------------------------------------------------------------
# Single-match deep-link: parse URL → fetch teams/odds → upsert upcoming
# ---------------------------------------------------------------------------

_MATCH_MID_QUERY_RE = re.compile(
    r"[?&#](?:mid|eventId|event_id)=([A-Za-z0-9]{6,12})\b",
    re.IGNORECASE,
)
_MATCH_SHORT_PATH_RE = re.compile(
    r"/match/(?!football(?:/|$))([A-Za-z0-9]{6,12})(?:/|$|\?)",
    re.IGNORECASE,
)
_MATCH_PATH_SIDES_RE = re.compile(
    r"/match/(?:football/)?(?P<home>[A-Za-z0-9_-]+)/(?P<away>[A-Za-z0-9_-]+)/?",
    re.IGNORECASE,
)
_SIDE_SLUG_HASH_RE = re.compile(
    r"^(?P<slug>[A-Za-z0-9_-]+)-(?P<hash>[A-Za-z0-9]{6,12})$"
)
_ENV_JSON_RE = re.compile(r"window\.environment\s*=\s*(\{.*?\});", re.DOTALL)
_MATCH_URL_EXAMPLE = (
    "https://www.flashscore.com/match/football/"
    "gamba-osaka-zLQAGOBK/tokushima-IcwgwCCt/?mid=zg0G0pvA"
)


def _split_side_token(token: str) -> tuple[str | None, str | None]:
    """Split ``gamba-osaka-zLQAGOBK`` → ``(slug, hash)``."""
    raw = str(token or "").strip().strip("/")
    if not raw:
        return None, None
    m = _SIDE_SLUG_HASH_RE.match(raw)
    if not m:
        return raw, None
    return str(m.group("slug")), str(m.group("hash"))


def parse_flashscore_match_url(url: str) -> dict[str, str]:
    """Extract Flashscore ``match_id`` (``mid``) and optional path metadata.

    Accepted forms
    --------------
    * ``…/match/football/{home}/{away}/?mid=zg0G0pvA``
    * ``…/match/{mid}/`` (short redirect form)
    * Query ``mid=`` on any flashscore match URL

    Returns
    -------
    dict
        At least ``{"match_id": "…"}``. May include ``home_slug``, ``home_hash``,
        ``away_slug``, ``away_hash``, ``canonical_url``.

    Raises
    ------
    ValueError
        When the URL is empty / not Flashscore / missing a recoverable mid.
    """
    from urllib.parse import parse_qs, urlparse, urlunparse

    raw = str(url or "").strip()
    if not raw:
        raise ValueError("URL trống — hãy dán link trận Flashscore.")

    # Allow scheme-less paste.
    candidate = raw if "://" in raw else f"https://{raw}"
    parsed = urlparse(candidate)
    host = (parsed.netloc or "").lower()
    if "flashscore." not in host and not host.endswith("flashscore.com"):
        raise ValueError(
            "URL không phải Flashscore. Ví dụ: " + _MATCH_URL_EXAMPLE
        )

    qs = parse_qs(parsed.query or "")
    mid = ""
    for key in ("mid", "MID", "eventId", "event_id"):
        vals = qs.get(key) or []
        if vals and str(vals[0]).strip():
            mid = str(vals[0]).strip()
            break

    if not mid:
        m = _MATCH_MID_QUERY_RE.search(candidate)
        if m:
            mid = str(m.group(1)).strip()
    if not mid:
        m = _MATCH_SHORT_PATH_RE.search(parsed.path or "")
        if m:
            mid = str(m.group(1)).strip()

    if not mid or not re.fullmatch(r"[A-Za-z0-9]{6,12}", mid):
        raise ValueError(
            "Không tìm thấy mid trên URL. Ví dụ: " + _MATCH_URL_EXAMPLE
        )

    out: dict[str, str] = {"match_id": mid}
    sides = _MATCH_PATH_SIDES_RE.search(parsed.path or "")
    if sides:
        h_slug, h_hash = _split_side_token(sides.group("home"))
        a_slug, a_hash = _split_side_token(sides.group("away"))
        # Ignore false-positive when path is just /match/{mid}/
        if h_slug and h_slug.casefold() != mid.casefold():
            if h_slug:
                out["home_slug"] = h_slug
            if h_hash:
                out["home_hash"] = h_hash
            if a_slug:
                out["away_slug"] = a_slug
            if a_hash:
                out["away_hash"] = a_hash

    if "home_slug" in out:
        clean = parsed._replace(query=f"mid={mid}", fragment="")
        out["canonical_url"] = urlunparse(clean)
    else:
        out["canonical_url"] = f"{FLASHSCORE_BASE}/match/{mid}/"
    return out


def resolve_league_from_flashscore_path(path: str | None) -> str | None:
    """Map a Flashscore tournament path to a configured league key."""
    from src.league_registry import load_leagues_json

    raw = str(path or "").strip().rstrip("/")
    if not raw:
        return None
    if not raw.startswith("/"):
        raw = "/" + raw
    best: tuple[int, str] | None = None
    for code, entry in load_leagues_json().items():
        cfg_path = str(entry.get("flashscore_path") or "").strip().rstrip("/")
        if not cfg_path:
            continue
        if raw == cfg_path or raw.startswith(cfg_path + "/"):
            score = len(cfg_path)
            if best is None or score > best[0]:
                best = (score, str(code).upper())
    return best[1] if best else None


def _parse_window_environment(html: str) -> dict[str, Any]:
    m = _ENV_JSON_RE.search(str(html or ""))
    if not m:
        return {}
    import json

    try:
        data = json.loads(m.group(1))
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _participant_side(data: dict[str, Any], side: str) -> dict[str, str]:
    block = (data.get("participantsData") or {}).get(side) or []
    if not isinstance(block, list) or not block:
        return {}
    entry = block[0] if isinstance(block[0], dict) else {}
    return {
        "name": str(entry.get("name") or entry.get("seo_name") or "").strip(),
        "hash": str(entry.get("id") or "").strip(),
        "slug": str(entry.get("url_name") or "").strip().strip("/"),
    }


def fetch_flashscore_match_detail(
    match_id: str,
    *,
    page_url: str | None = None,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> dict[str, Any]:
    """Load one Flashscore match page + Odds GraphQL markets.

    Returns a dict with ``match_id``, ``HomeTeam``, ``AwayTeam``, ``Kickoff``,
    odds columns (``B365H``/…), ``league_id`` (best-effort), team hashes, and
    raw ``tournament_path``.
    """
    mid = str(match_id or "").strip()
    if not mid:
        raise ValueError("Thiếu match_id Flashscore.")

    url = str(page_url or "").strip() or f"{FLASHSCORE_BASE}/match/{mid}/"
    try:
        html = _fetch_text_retry(url, timeout=float(timeout_s))
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Không tải được trang trận: {exc}") from exc

    env = _parse_window_environment(html)
    env_mid = str(env.get("event_id_c") or "").strip()
    if env_mid and env_mid != mid:
        mid = env_mid

    home_p = _participant_side(env, "home")
    away_p = _participant_side(env, "away")
    home = home_p.get("name") or ""
    away = away_p.get("name") or ""
    if not home or not away:
        title_m = re.search(
            r'property="og:title"\s+content="([^"]+)"', html, re.IGNORECASE
        )
        if title_m:
            parts = re.split(r"\s+[–\-vV]\s+", title_m.group(1), maxsplit=1)
            if len(parts) == 2:
                home = home or parts[0].strip()
                away = away or parts[1].strip()
    if not home or not away:
        raise RuntimeError("Không đọc được tên đội từ trang Flashscore.")

    kickoff = None
    ts_raw = env.get("eventStageStartTime")
    try:
        ts = int(ts_raw) if ts_raw is not None else 0
    except (TypeError, ValueError):
        ts = 0
    if ts > 0:
        kickoff = datetime.fromtimestamp(ts, tz=timezone.utc).replace(tzinfo=None)

    header = env.get("header") or {}
    tournament = header.get("tournament") if isinstance(header, dict) else {}
    t_link = ""
    t_name = ""
    if isinstance(tournament, dict):
        t_link = str(tournament.get("link") or "").strip()
        t_name = str(tournament.get("tournament") or "").strip()

    league_id = resolve_league_from_flashscore_path(t_link)

    try:
        odds = fetch_flashscore_match_odds(mid)
    except Exception as exc:  # noqa: BLE001
        logger.warning("odds fetch failed for %s: %s", mid, exc)
        odds = {}

    row: dict[str, Any] = {
        "Date": pd.Timestamp(kickoff).normalize() if kickoff is not None else pd.NaT,
        "Kickoff": pd.Timestamp(kickoff) if kickoff is not None else pd.NaT,
        "HomeTeam": home,
        "AwayTeam": away,
        "FlashscoreEventId": mid,
        "Source": "flashscore_match_link",
        "league_id": league_id,
        "HomeFlashscoreHash": home_p.get("hash") or None,
        "AwayFlashscoreHash": away_p.get("hash") or None,
        "HomeFlashscoreSlug": home_p.get("slug") or None,
        "AwayFlashscoreSlug": away_p.get("slug") or None,
        "TournamentPath": t_link or None,
        "TournamentName": t_name or None,
        "OddsProvider": odds.get("OddsProvider") or "Flashscore",
    }
    for key in (
        "B365H",
        "B365D",
        "B365A",
        "OU_Line",
        "OddsOver",
        "OddsUnder",
        "B365_O25",
        "B365_U25",
        "AHh",
        "B365AHH",
        "B365AHA",
    ):
        if key in odds:
            row[key] = odds[key]

    return {
        "match_id": mid,
        "league_id": league_id,
        "row": row,
        "odds": odds,
        "page_url": url,
        "tournament_path": t_link,
        "tournament_name": t_name,
    }


def fetch_and_persist_match_from_url(
    url: str,
    *,
    default_league: str | None = None,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    db_path: Path | str | None = None,
) -> dict[str, Any]:
    """Parse a match link, fetch teams+odds, upsert into ``upcoming_fixtures``.

    Uses the tournament path to pick ``league_id`` when configured; otherwise
    falls back to ``default_league``. Applies league team aliases before upsert.
    """
    from src.global_db import GLOBAL_DB_PATH, upsert_upcoming_fixture

    parsed = parse_flashscore_match_url(url)
    mid = parsed["match_id"]
    page_url = parsed.get("canonical_url") or url

    detail = fetch_flashscore_match_detail(
        mid, page_url=page_url, timeout_s=timeout_s
    )
    row = dict(detail["row"])
    league = detail.get("league_id") or (
        str(default_league).strip().upper() if default_league else None
    )
    if not league:
        raise RuntimeError(
            "Không xác định được giải từ link — chọn giải ở sidebar rồi thử lại."
        )

    home_raw = str(row.get("HomeTeam") or "")
    away_raw = str(row.get("AwayTeam") or "")
    row["HomeTeam"] = resolve_league_team_name(home_raw, league)
    row["AwayTeam"] = resolve_league_team_name(away_raw, league)
    row["league_id"] = league

    hash_entries: list[dict[str, str]] = []
    for side, name_raw in (("Home", home_raw), ("Away", away_raw)):
        h = row.get(f"{side}FlashscoreHash")
        slug = row.get(f"{side}FlashscoreSlug")
        if h:
            hash_entries.append(
                {
                    "hash": str(h),
                    "slug": str(slug or ""),
                    "name": name_raw,
                }
            )
    if hash_entries:
        try:
            persist_team_hashes(hash_entries, league)
        except Exception as exc:  # noqa: BLE001
            logger.warning("persist hashes from match link: %s", exc)

    key = upsert_upcoming_fixture(
        league,
        row,
        db_path=db_path if db_path is not None else GLOBAL_DB_PATH,
    )
    return {
        "match_id": mid,
        "league": league,
        "home": row["HomeTeam"],
        "away": row["AwayTeam"],
        "kickoff": row.get("Kickoff"),
        "row": row,
        "odds": detail.get("odds") or {},
        "upcoming_key": key,
        "tournament_name": detail.get("tournament_name"),
        "page_url": page_url,
    }
