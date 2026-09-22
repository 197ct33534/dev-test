"""Load and clean Premier League match data from football-data.co.uk.

Downloads season CSVs (division code ``E0``), normalises dates/columns,
and returns a single DataFrame ready for Dixon-Coles fitting, value-bet
odds comparison, and corner modelling.
"""

from __future__ import annotations

import json
import re
import sqlite3
import time
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
from typing import Iterable, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BASE_URL = "https://www.football-data.co.uk/mmz4281/{season_code}/E0.csv"
FIXTURES_URL = "https://www.football-data.co.uk/fixtures.csv"
# Flashscore PL fixtures page is JS-rendered; schedule is synced via Fotmob's
# public league API (same calendar as flashscore.com/.../premier-league/fixtures/).
FLASHSCORE_FIXTURES_URL = (
    "https://www.flashscore.com/football/england/premier-league/fixtures/"
)
FLASHSCORE_UWCL_FIXTURES_URL = (
    "https://www.flashscore.com/football/europe/uefa-champions-league-women/fixtures/"
)
FOTMOB_EPL_URL = "https://www.fotmob.com/api/data/leagues?id=47"
FOTMOB_UWCL_URL = "https://www.fotmob.com/api/data/leagues?id=9375"
FOTMOB_LEAGUE_URL = "https://www.fotmob.com/api/data/leagues?id={league_id}"
# Flashscore Odds tab (same feed as flashscore.com/.../odds/...) — GraphQL, no API key.
FLASHSCORE_ODDS_URL = "https://global.ds.lsapp.eu/odds/pq_graphql"
FLASHSCORE_FSIGN = "SW9D1eZo"
# Prefer bet365 as shown on Flashscore GB odds comparison.
FLASHSCORE_BOOKMAKER_PREFER: tuple[str, ...] = (
    "16",  # bet365
    "15",  # William Hill
    "26",  # Betway
    "14",  # 10bet
)
# Live book odds fallback (DraftKings via ESPN public scoreboard JSON).
ESPN_EPL_SCOREBOARD_URL = (
    "https://site.web.api.espn.com/apis/site/v2/sports/soccer/eng.1/scoreboard"
)

# Core result fields required by Dixon-Coles and corner models.
REQUIRED_COLUMNS: list[str] = [
    "Date",
    "HomeTeam",
    "AwayTeam",
    "FTHG",  # Full-time home goals (X)
    "FTAG",  # Full-time away goals (Y)
    "FTR",   # Full-time result: H / D / A
]

# Optional stats / odds kept when present in the source CSV.
OPTIONAL_COLUMNS: list[str] = [
    # Shot / corner volume (AH–OU & Corners feature engineering)
    "HS",   # Home shots
    "AS",   # Away shots
    "HST",  # Home shots on target
    "AST",  # Away shots on target
    "HC",   # Home corners
    "AC",   # Away corners
    # Bet365 match odds (widely available historically)
    "B365H",
    "B365D",
    "B365A",
    # Market average / maximum 1X2 odds
    "AvgH",
    "AvgD",
    "AvgA",
    "MaxH",
    "MaxD",
    "MaxA",
    # Pinnacle (sharp book) when available
    "PSH",
    "PSD",
    "PSA",
    # Totals / Asian Handicap (closing) for walk-forward backtests
    "B365>2.5",
    "B365<2.5",
    "Avg>2.5",
    "Avg<2.5",
    "AHh",
    "B365AHH",
    "B365AHA",
    "AvgAHH",
    "AvgAHA",
]

# Match-stat columns persisted into SQLite ``matches`` (epl_matches.db).
STAT_COLUMNS: tuple[str, ...] = ("HS", "AS", "HST", "AST", "HC", "AC")

# Fixtures feed columns (football-data.co.uk/fixtures.csv) → safe aliases.
FIXTURE_COLUMN_ALIASES: dict[str, str] = {
    "B365>2.5": "B365_O25",
    "B365<2.5": "B365_U25",
    "Avg>2.5": "Avg_O25",
    "Avg<2.5": "Avg_U25",
    "Max>2.5": "Max_O25",
    "Max<2.5": "Max_U25",
}

DEFAULT_N_SEASONS = 3
USER_AGENT = "epl-value-betting/1.0 (educational; +https://github.com/)"

# Local SQLite persistence (offline fallback for historical match data)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_DB_PATH = DATA_DIR / "epl_matches.db"
UWCL_DB_PATH = DATA_DIR / "uwcl_matches.db"
MATCHES_TABLE = "matches"

# Supported competitions (Flashscore / Fotmob IDs).
SUPPORTED_LEAGUES: tuple[str, ...] = ("EPL", "UWCL")
LEAGUE_CONFIG: dict[str, dict[str, object]] = {
    "EPL": {
        "label": "EPL - Premier League",
        "short": "EPL",
        "db_path": DEFAULT_DB_PATH,
        "fotmob_id": 47,
        "flashscore_fixtures_url": FLASHSCORE_FIXTURES_URL,
        "fd_div": "E0",
        "history_source": "football-data",
        "telegram_tag": "EPL",
    },
    "UWCL": {
        "label": "UWCL - UEFA Women's Champions League",
        "short": "UWCL",
        "db_path": UWCL_DB_PATH,
        "fotmob_id": 9375,
        "flashscore_fixtures_url": FLASHSCORE_UWCL_FIXTURES_URL,
        "fd_div": None,
        "history_source": "fotmob",
        "telegram_tag": "UWCL",
        "flashscore_ref": (
            "https://www.flashscore.com/football/europe/uefa-champions-league-women/"
        ),
    },
}


def normalize_league(league: str | None) -> str:
    """Map user input → canonical league code (``EPL`` / ``UWCL``)."""
    raw = str(league or "EPL").strip().upper()
    aliases = {
        "EPL": "EPL",
        "PL": "EPL",
        "PREMIER": "EPL",
        "PREMIER LEAGUE": "EPL",
        "E0": "EPL",
        "UWCL": "UWCL",
        "UWCL - UEFA WOMEN'S CHAMPIONS LEAGUE": "UWCL",
        "WOMEN'S CHAMPIONS LEAGUE": "UWCL",
        "UEFA WOMEN'S CHAMPIONS LEAGUE": "UWCL",
        "WCL": "UWCL",
    }
    key = aliases.get(raw, raw)
    if key not in LEAGUE_CONFIG:
        raise ValueError(
            f"Unsupported league={league!r}. Choose one of {list(LEAGUE_CONFIG)}"
        )
    return key


def league_db_path(league: str | None = "EPL") -> Path:
    """SQLite path for a league (EPL → ``epl_matches.db``, UWCL → ``uwcl_matches.db``)."""
    code = normalize_league(league)
    return Path(LEAGUE_CONFIG[code]["db_path"])  # type: ignore[arg-type]


def league_label(league: str | None = "EPL") -> str:
    code = normalize_league(league)
    return str(LEAGUE_CONFIG[code]["label"])


def league_telegram_tag(league: str | None = "EPL") -> str:
    code = normalize_league(league)
    return str(LEAGUE_CONFIG[code]["telegram_tag"])


# ---------------------------------------------------------------------------
# Season helpers
# ---------------------------------------------------------------------------


def season_code(season_start: int) -> str:
    """Convert a season start year to the football-data.co.uk path code.

    Parameters
    ----------
    season_start:
        Calendar year in which the season begins (e.g. ``2023`` → season
        2023/24 → code ``\"2324\"``).

    Returns
    -------
    str
        Two concatenated two-digit year fragments, e.g. ``\"2324\"``.
    """
    start = season_start % 100
    end = (season_start + 1) % 100
    return f"{start:02d}{end:02d}"


def default_season_starts(n_seasons: int = DEFAULT_N_SEASONS) -> list[int]:
    """Return the most recent ``n_seasons`` EPL season start years.

    Uses a July cut-over: before 1 July, the current season is treated as
    starting in the previous calendar year (EPL runs Aug–May).
    """
    today = datetime.now(timezone.utc).date()
    current_start = today.year if today.month >= 7 else today.year - 1
    return list(range(current_start - n_seasons + 1, current_start + 1))


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------


def _fetch_csv_text(url: str, timeout: float = 30.0) -> str:
    """Download raw CSV text from ``url`` with a polite User-Agent."""
    request = Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urlopen(request, timeout=timeout) as response:
            charset = response.headers.get_content_charset() or "utf-8"
            return response.read().decode(charset, errors="replace")
    except HTTPError as exc:
        raise RuntimeError(f"HTTP {exc.code} when fetching {url}") from exc
    except URLError as exc:
        raise RuntimeError(f"Network error fetching {url}: {exc.reason}") from exc


def _fetch_json(
    url: str,
    *,
    timeout: float = 30.0,
    referer: str | None = None,
    extra_headers: dict[str, str] | None = None,
) -> dict:
    """Download a JSON payload (used for Fotmob / ESPN / Flashscore endpoints)."""
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "application/json",
    }
    if referer:
        headers["Referer"] = referer
    if extra_headers:
        headers.update(extra_headers)
    request = Request(url, headers=headers)
    try:
        with urlopen(request, timeout=timeout) as response:
            charset = response.headers.get_content_charset() or "utf-8"
            text = response.read().decode(charset, errors="replace")
    except HTTPError as exc:
        raise RuntimeError(f"HTTP {exc.code} when fetching {url}") from exc
    except URLError as exc:
        raise RuntimeError(f"Network error fetching {url}: {exc.reason}") from exc

    return json.loads(text)


def _fetch_text(
    url: str,
    *,
    timeout: float = 45.0,
    referer: str | None = None,
    extra_headers: dict[str, str] | None = None,
) -> str:
    """Download raw text (HTML / Flashscore delimited feeds)."""
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "*/*",
    }
    if referer:
        headers["Referer"] = referer
    if extra_headers:
        headers.update(extra_headers)
    request = Request(url, headers=headers)
    try:
        with urlopen(request, timeout=timeout) as response:
            charset = response.headers.get_content_charset() or "utf-8"
            return response.read().decode(charset, errors="replace")
    except HTTPError as exc:
        raise RuntimeError(f"HTTP {exc.code} when fetching {url}") from exc
    except URLError as exc:
        raise RuntimeError(f"Network error fetching {url}: {exc.reason}") from exc


def download_season(season_start: int) -> pd.DataFrame:
    """Download one Premier League season CSV into a DataFrame.

    Parameters
    ----------
    season_start:
        Season start year (e.g. ``2022`` for 2022/23).

    Returns
    -------
    pd.DataFrame
        Raw season table with an added ``Season`` label column
        (e.g. ``\"2022/23\"``).
    """
    code = season_code(season_start)
    url = BASE_URL.format(season_code=code)
    text = _fetch_csv_text(url)
    df = pd.read_csv(StringIO(text))
    meta = pd.DataFrame(
        {
            "Season": f"{season_start}/{str(season_start + 1)[-2:]}",
            "SeasonStart": season_start,
        },
        index=df.index,
    )
    return pd.concat([df, meta], axis=1)


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------


def _parse_dates(series: pd.Series) -> pd.Series:
    """Parse football-data date strings (``dd/mm/yy`` or ``dd/mm/yyyy``)."""
    parsed = pd.to_datetime(series, dayfirst=True, format="mixed", errors="coerce")
    return parsed


def clean_matches(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and standardise a raw football-data match DataFrame.

    Steps
    -----
    1. Keep required + available optional columns (plus ``Season`` metadata),
       including shot/corner stats ``HS``, ``AS``, ``HST``, ``AST``, ``HC``, ``AC``.
    2. Parse ``Date`` and drop rows with invalid dates.
    3. Coerce goal / shot / corner columns to numeric integers.
    4. Drop rows missing home/away team or full-time goals.
    5. Sort chronologically and reset the index.

    Parameters
    ----------
    df:
        Raw concatenated season DataFrame from :func:`download_season`.

    Returns
    -------
    pd.DataFrame
        Clean match-level table suitable for model fitting.
    """
    if df.empty:
        return df.copy()

    keep = [c for c in REQUIRED_COLUMNS + OPTIONAL_COLUMNS if c in df.columns]
    meta = [
        c
        for c in ("Season", "SeasonStart", "league_id", "FotmobMatchId", "Round")
        if c in df.columns
    ]
    out = df[keep + meta].copy()

    missing_required = [c for c in REQUIRED_COLUMNS if c not in out.columns]
    if missing_required:
        raise ValueError(f"Source data missing required columns: {missing_required}")

    out["Date"] = _parse_dates(out["Date"])
    out = out.dropna(subset=["Date", "HomeTeam", "AwayTeam"])

    for col in ("FTHG", "FTAG", "HC", "AC", "HS", "AS", "HST", "AST"):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    out = out.dropna(subset=["FTHG", "FTAG"])
    out["FTHG"] = out["FTHG"].astype(int)
    out["FTAG"] = out["FTAG"].astype(int)

    for col in STAT_COLUMNS:
        if col in out.columns:
            # Stats may be missing on a few older rows; keep as nullable Int64.
            out[col] = out[col].astype("Int64")

    odds_cols = [
        c
        for c in out.columns
        if c.startswith(("B365", "Avg", "Max", "PS")) and c[-1] in "HDA"
    ]
    for col in odds_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    # Totals / AH closing lines (football-data uses B365>2.5 style names)
    rename_lines = {
        "B365>2.5": "B365_O25",
        "B365<2.5": "B365_U25",
        "Avg>2.5": "Avg_O25",
        "Avg<2.5": "Avg_U25",
    }
    out = out.rename(columns={k: v for k, v in rename_lines.items() if k in out.columns})
    for col in (
        "B365_O25",
        "B365_U25",
        "Avg_O25",
        "Avg_U25",
        "AHh",
        "B365AHH",
        "B365AHA",
        "AvgAHH",
        "AvgAHA",
    ):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    out["HomeTeam"] = out["HomeTeam"].astype(str).str.strip()
    out["AwayTeam"] = out["AwayTeam"].astype(str).str.strip()
    out["FTR"] = out["FTR"].astype(str).str.strip().str.upper()
    out = canonicalize_team_columns(out)

    out = out.sort_values(["Date", "HomeTeam", "AwayTeam"]).reset_index(drop=True)
    return out


# ---------------------------------------------------------------------------
# SQLite persistence (offline fallback)
# ---------------------------------------------------------------------------


def _ensure_data_dir(db_path: Path | str = DEFAULT_DB_PATH) -> Path:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def build_match_id(df: pd.DataFrame) -> pd.Series:
    """Primary key: ``YYYY-MM-DD|HomeTeam|AwayTeam``."""
    dates = pd.to_datetime(df["Date"], errors="coerce").dt.strftime("%Y-%m-%d")
    return dates + "|" + df["HomeTeam"].astype(str) + "|" + df["AwayTeam"].astype(str)


def db_has_matches(db_path: Path | str = DEFAULT_DB_PATH) -> bool:
    """True when local SQLite exists and ``matches`` has at least one row."""
    path = Path(db_path)
    if not path.is_file():
        return False
    try:
        with sqlite3.connect(path) as conn:
            row = conn.execute(
                f"SELECT COUNT(*) FROM {MATCHES_TABLE}"
            ).fetchone()
        return bool(row and int(row[0]) > 0)
    except sqlite3.Error:
        return False


def read_matches_from_db(
    db_path: Path | str = DEFAULT_DB_PATH,
    *,
    seasons: Sequence[int] | None = None,
    n_seasons: int | None = None,
) -> pd.DataFrame:
    """Load cleaned matches from local SQLite.

    Optionally filter to the latest ``n_seasons`` (or explicit ``seasons``)
    using the ``SeasonStart`` column when present.
    """
    path = Path(db_path)
    if not path.is_file():
        return pd.DataFrame()

    with sqlite3.connect(path) as conn:
        try:
            df = pd.read_sql(f"SELECT * FROM {MATCHES_TABLE}", conn)
        except (sqlite3.Error, pd.errors.DatabaseError):
            return pd.DataFrame()

    if df.empty:
        return df

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    for col in ("FTHG", "FTAG"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=[c for c in ("Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG") if c in df.columns])
    if "FTHG" in df.columns:
        df["FTHG"] = df["FTHG"].astype(int)
    if "FTAG" in df.columns:
        df["FTAG"] = df["FTAG"].astype(int)
    for col in STAT_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    odds_cols = [
        c
        for c in df.columns
        if c.startswith(("B365", "Avg", "Max", "PS")) and str(c)[-1] in "HDA"
    ]
    for col in odds_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if "SeasonStart" in df.columns:
        df["SeasonStart"] = pd.to_numeric(df["SeasonStart"], errors="coerce").astype(
            "Int64"
        )

    if seasons is not None and "SeasonStart" in df.columns:
        starts = {int(s) for s in seasons}
        df = df.loc[df["SeasonStart"].isin(starts)].copy()
    elif n_seasons is not None and "SeasonStart" in df.columns:
        starts = set(default_season_starts(n_seasons))
        df = df.loc[df["SeasonStart"].isin(starts)].copy()

    df = df.sort_values(["Date", "HomeTeam", "AwayTeam"]).reset_index(drop=True)
    return canonicalize_team_columns(df)


def save_matches_to_db(
    df: pd.DataFrame,
    db_path: Path | str = DEFAULT_DB_PATH,
) -> int:
    """Upsert match rows into SQLite (PK = Match_ID).

    Returns
    -------
    int
        Total row count stored in ``matches`` after the write.
    """
    if df.empty:
        return 0

    path = _ensure_data_dir(db_path)
    incoming = df.copy()
    if "Match_ID" not in incoming.columns:
        incoming["Match_ID"] = build_match_id(incoming)

    # Store dates as ISO strings for portable SQLite reads
    to_write = incoming.copy()
    if "Date" in to_write.columns:
        to_write["Date"] = pd.to_datetime(to_write["Date"], errors="coerce").dt.strftime(
            "%Y-%m-%d"
        )

    with sqlite3.connect(path) as conn:
        try:
            existing = pd.read_sql(f"SELECT * FROM {MATCHES_TABLE}", conn)
        except (sqlite3.Error, pd.errors.DatabaseError):
            existing = pd.DataFrame()

        if not existing.empty and "Match_ID" in existing.columns:
            combined = pd.concat([existing, to_write], ignore_index=True)
            combined = combined.drop_duplicates(subset=["Match_ID"], keep="last")
        else:
            combined = to_write

        combined = combined.sort_values(
            ["Date", "HomeTeam", "AwayTeam"], kind="mergesort"
        ).reset_index(drop=True)
        combined.to_sql(MATCHES_TABLE, conn, if_exists="replace", index=False)

        # Enforce PK for future tooling / integrity
        conn.execute(
            f"""
            CREATE UNIQUE INDEX IF NOT EXISTS idx_matches_pk
            ON {MATCHES_TABLE} (Match_ID)
            """
        )
        conn.commit()
        return len(combined)


def _download_epl_seasons(
    season_starts: Iterable[int],
) -> tuple[pd.DataFrame, list[str]]:
    """Download + clean seasons from football-data.co.uk.

    Returns
    -------
    (cleaned_df, errors)
        ``errors`` lists per-season download failures (partial success OK).
    """
    frames: list[pd.DataFrame] = []
    errors: list[str] = []

    for start in season_starts:
        try:
            frames.append(download_season(int(start)))
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{start}/{start + 1}: {exc}")

    if not frames:
        detail = "; ".join(errors) if errors else "no seasons requested"
        raise RuntimeError(f"Failed to download any EPL season data ({detail})")

    cleaned = clean_matches(pd.concat(frames, ignore_index=True))
    return cleaned, errors


def _parse_fotmob_score(score_str: str | None) -> tuple[int, int] | None:
    """Parse Fotmob ``scoreStr`` like ``\"3 - 0\"`` → ``(3, 0)``."""
    if not score_str:
        return None
    m = re.match(r"^\s*(\d+)\s*[-:]\s*(\d+)\s*$", str(score_str))
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def _normalize_uwcl_team(name: str) -> str:
    """Strip women's suffixes then apply :func:`normalize_team_name` aliases."""
    return normalize_team_name(name, known_teams=None)


def _fotmob_season_start(season_name: str) -> int:
    """``2024/2025`` → ``2024``."""
    m = re.match(r"^(\d{4})", str(season_name))
    return int(m.group(1)) if m else int(datetime.now(timezone.utc).year)


def _fetch_fotmob_league_payload(fotmob_id: int, season: str | None = None) -> dict:
    """Download Fotmob league JSON (optional season ``YYYY/YYYY``)."""
    url = FOTMOB_LEAGUE_URL.format(league_id=int(fotmob_id))
    if season:
        url = f"{url}&season={quote(str(season), safe='')}"
    return _fetch_json(
        url,
        referer="https://www.fotmob.com/",
        extra_headers={"Accept": "application/json"},
    )


def _matches_from_fotmob_payload(
    payload: dict,
    *,
    league_id: str,
    season_name: str,
    finished_only: bool = True,
) -> pd.DataFrame:
    """Convert Fotmob ``fixtures.allMatches`` → cleaned match rows."""
    fixtures_block = payload.get("fixtures") or {}
    matches = fixtures_block.get("allMatches") or []
    season_start = _fotmob_season_start(season_name)
    rows: list[dict] = []
    for m in matches:
        status = m.get("status") or {}
        if status.get("cancelled"):
            continue
        finished = bool(status.get("finished"))
        if finished_only and not finished:
            continue
        utc = status.get("utcTime")
        kickoff = pd.to_datetime(utc, utc=True, errors="coerce")
        if pd.isna(kickoff):
            continue
        kickoff_naive = kickoff.tz_convert("UTC").tz_localize(None)
        home = _normalize_uwcl_team((m.get("home") or {}).get("name", ""))
        away = _normalize_uwcl_team((m.get("away") or {}).get("name", ""))
        if not home or not away:
            continue
        score = _parse_fotmob_score(status.get("scoreStr"))
        if finished_only and score is None:
            continue
        fthg = int(score[0]) if score else None
        ftag = int(score[1]) if score else None
        if fthg is None or ftag is None:
            continue
        if fthg > ftag:
            ftr = "H"
        elif fthg < ftag:
            ftr = "A"
        else:
            ftr = "D"
        rows.append(
            {
                "Date": kickoff_naive.normalize(),
                "HomeTeam": home,
                "AwayTeam": away,
                "FTHG": fthg,
                "FTAG": ftag,
                "FTR": ftr,
                "Season": season_name,
                "SeasonStart": season_start,
                "league_id": league_id,
                "FotmobMatchId": str(m.get("id", "")),
                "Round": m.get("round") or m.get("roundName"),
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def _download_uwcl_seasons(
    n_seasons: int = DEFAULT_N_SEASONS,
    *,
    fotmob_id: int = 9375,
) -> tuple[pd.DataFrame, list[str]]:
    """Download finished UWCL matches from Fotmob across recent seasons.

    Flashscore UWCL page is JS-rendered; Fotmob league id ``9375`` mirrors the
    same calendar (reference:
    https://www.flashscore.com/football/europe/uefa-champions-league-women/).
    """
    errors: list[str] = []
    try:
        meta = _fetch_fotmob_league_payload(fotmob_id)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Failed to reach Fotmob UWCL endpoint: {exc}") from exc

    available = list(meta.get("allAvailableSeasons") or [])
    if not available:
        details = meta.get("details") or {}
        latest = details.get("latestSeason") or details.get("selectedSeason")
        available = [str(latest)] if latest else []

    frames: list[pd.DataFrame] = []
    for season in available:
        if len(frames) >= int(n_seasons):
            break
        try:
            payload = _fetch_fotmob_league_payload(fotmob_id, season=season)
            frame = _matches_from_fotmob_payload(
                payload, league_id="UWCL", season_name=str(season), finished_only=True
            )
            if frame.empty:
                errors.append(f"{season}: no finished matches")
                continue
            frames.append(frame)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{season}: {exc}")

    if not frames:
        detail = "; ".join(errors) if errors else "no seasons"
        raise RuntimeError(f"Failed to download UWCL history ({detail})")

    raw = pd.concat(frames, ignore_index=True)
    cleaned = clean_matches(raw)
    if "league_id" not in cleaned.columns:
        cleaned["league_id"] = "UWCL"
    return cleaned, errors


def load_league_data(
    league: str = "EPL",
    seasons: Sequence[int] | None = None,
    n_seasons: int = DEFAULT_N_SEASONS,
    *,
    force_refresh: bool = False,
    db_path: Path | str | None = None,
) -> pd.DataFrame:
    """Load historical matches for ``EPL`` or ``UWCL`` with SQLite fallback.

    Parameters
    ----------
    league:
        ``EPL`` (football-data.co.uk) or ``UWCL`` (Fotmob id 9375).
    n_seasons / seasons / force_refresh:
        Same semantics as :func:`load_epl_data`.
    db_path:
        Override SQLite path (default per-league: ``epl_matches.db`` /
        ``uwcl_matches.db``).
    """
    code = normalize_league(league)
    path = Path(db_path) if db_path is not None else league_db_path(code)
    cfg = LEAGUE_CONFIG[code]

    if seasons is None:
        season_starts = list(default_season_starts(n_seasons))
    else:
        season_starts = [int(s) for s in seasons]

    def _attach_meta(
        df: pd.DataFrame,
        *,
        source: str,
        warnings: list[str] | None = None,
    ) -> pd.DataFrame:
        out = df.copy()
        if "league_id" not in out.columns:
            out["league_id"] = code
        out.attrs["data_source"] = source
        out.attrs["db_path"] = str(path)
        out.attrs["league"] = code
        out.attrs["league_label"] = str(cfg["label"])
        out.attrs["n_matches"] = int(len(out))
        if warnings:
            out.attrs["download_warnings"] = warnings
        return out

    if not force_refresh and db_has_matches(path):
        local = read_matches_from_db(
            path,
            seasons=season_starts if seasons is not None else None,
            n_seasons=n_seasons,
        )
        if not local.empty:
            return _attach_meta(local, source="sqlite_local")

    try:
        if code == "EPL":
            cleaned, errors = _download_epl_seasons(season_starts)
            cleaned["league_id"] = "EPL"
            source = "football-data.co.uk"
        else:
            cleaned, errors = _download_uwcl_seasons(
                n_seasons=n_seasons if seasons is None else len(season_starts),
                fotmob_id=int(cfg["fotmob_id"]),  # type: ignore[arg-type]
            )
            source = "fotmob"
        save_matches_to_db(cleaned, path)
        return _attach_meta(cleaned, source=source, warnings=errors or None)
    except Exception as exc:  # noqa: BLE001
        if db_has_matches(path):
            local = read_matches_from_db(
                path,
                seasons=season_starts if seasons is not None else None,
                n_seasons=n_seasons,
            )
            if not local.empty:
                out = _attach_meta(local, source="sqlite_fallback")
                out.attrs["download_warnings"] = [f"network/refresh failed: {exc}"]
                return out
        raise RuntimeError(
            f"Cannot load {code} data (network failed and no local DB at {path}): {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_epl_data(
    seasons: Sequence[int] | None = None,
    n_seasons: int = DEFAULT_N_SEASONS,
    *,
    force_refresh: bool = False,
    db_path: Path | str = DEFAULT_DB_PATH,
) -> pd.DataFrame:
    """Load and clean Premier League seasons with SQLite offline fallback.

    Thin wrapper around :func:`load_league_data` with ``league=\"EPL\"``.
    """
    return load_league_data(
        "EPL",
        seasons=seasons,
        n_seasons=n_seasons,
        force_refresh=force_refresh,
        db_path=db_path,
    )


def list_teams(df: pd.DataFrame) -> list[str]:
    """Return sorted unique team names appearing as home or away."""
    teams = set(df["HomeTeam"].dropna().unique()) | set(df["AwayTeam"].dropna().unique())
    return sorted(teams)


# ---------------------------------------------------------------------------
# Upcoming fixtures
# Schedule: Fotmob (đồng bộ lịch Flashscore PL)
# Odds: ESPN scoreboard (DraftKings) + football-data.co.uk/fixtures.csv
# ---------------------------------------------------------------------------

# Unified team-name aliases (Flashscore / Fotmob / ESPN → DB spelling).
TEAM_ALIASES: dict[str, str] = {
    # --- EPL / football-data.co.uk ---
    "Manchester United": "Man United",
    "Manchester Utd": "Man United",
    "Manchester City": "Man City",
    "Tottenham Hotspur": "Tottenham",
    "Nottingham Forest": "Nott'm Forest",
    "Nottingham": "Nott'm Forest",
    "Brighton & Hove Albion": "Brighton",
    "Brighton and Hove Albion": "Brighton",
    "AFC Bournemouth": "Bournemouth",
    "Newcastle United": "Newcastle",
    "West Ham United": "West Ham",
    "Wolverhampton Wanderers": "Wolves",
    "Leicester City": "Leicester",
    "Leeds United": "Leeds",
    "Ipswich Town": "Ipswich",
    "Hull City": "Hull",
    "Coventry City": "Coventry",
    "Sheffield United": "Sheffield United",
    "Luton Town": "Luton",
    # --- UWCL / women's clubs ---
    "Hacken": "BK Hacken",
    "Häcken": "BK Hacken",
    "BK Häcken": "BK Hacken",
    "Bayern München": "Bayern Munich",
    "Bayern Munchen": "Bayern Munich",
    "Austria Wien": "Austria Vienna",
    "HB Koge": "Koge",
    "HB Køge": "Koge",
    "Køge": "Koge",
    "Paris Saint-Germain": "PSG",
    "Paris Saint Germain": "PSG",
    "Paris SG": "PSG",
    "FC Barcelona": "Barcelona",
    "Atletico Madrid": "Atletico Madrid",
    "Atlético Madrid": "Atletico Madrid",
    "SL Benfica": "Benfica",
    "AS Roma": "Roma",
    "OH Leuven": "Leuven",
    "Oud-Heverlee Leuven": "Leuven",
    "Oud Heverlee Leuven": "Leuven",
    "OHL": "Leuven",
    "OHL Leuven": "Leuven",
    "Leuven Women": "Leuven",
    "Servette": "Servette Geneve FC",
    "Servette FC": "Servette Geneve FC",
    # Inter (UWCL newcomers / alternate spellings)
    "FC Internazionale": "Inter",
    "Internazionale": "Inter",
    "Inter Milan": "Inter",
    "Inter Women": "Inter",
    "FC Inter": "Inter",
}

# Back-compat aliases used by older call sites.
TEAM_NAME_TO_FD: dict[str, str] = dict(TEAM_ALIASES)
UWCL_TEAM_ALIASES: dict[str, str] = {
    k: v
    for k, v in TEAM_ALIASES.items()
    if k
    in {
        "Hacken",
        "Häcken",
        "BK Häcken",
        "Bayern München",
        "Bayern Munchen",
        "Austria Wien",
        "HB Koge",
        "HB Køge",
        "Køge",
        "Paris Saint-Germain",
        "Paris Saint Germain",
        "Paris SG",
        "Manchester City",
        "FC Barcelona",
        "Atletico Madrid",
        "Atlético Madrid",
        "SL Benfica",
        "AS Roma",
        "OH Leuven",
        "Oud-Heverlee Leuven",
        "Oud Heverlee Leuven",
        "OHL",
        "OHL Leuven",
        "Leuven Women",
        "Servette",
        "Servette FC",
        "FC Internazionale",
        "Internazionale",
        "Inter Milan",
        "Inter Women",
        "FC Inter",
    }
}


def _strip_accents(text: str) -> str:
    import unicodedata

    norm = unicodedata.normalize("NFKD", str(text or ""))
    return "".join(c for c in norm if not unicodedata.combining(c))


def _fold_team_key(name: str) -> str:
    """Accent-insensitive, prefix-stripped key for odds↔fixture merges.

    ``BK Häcken`` / ``Hacken`` → ``hacken``; ``Bayern München`` /
    ``Bayern Munich`` → ``bayernmunich``.
    """
    s = _strip_accents(name).lower().strip()
    s = re.sub(r"\s+", " ", s)
    for prefix in (
        "bk ",
        "fc ",
        "ac ",
        "as ",
        "sl ",
        "hb ",
        "afc ",
        "rsc ",
        "vfl ",
        "tsv ",
        "oh ",
    ):
        if s.startswith(prefix):
            s = s[len(prefix) :]
            break
    return re.sub(r"[^a-z0-9]+", "", s)


def normalize_team_name(name: str, known_teams: Sequence[str] | None = None) -> str:
    """Normalize a scraped team name via ``TEAM_ALIASES`` (+ optional fuzzy match).

    Steps
    -----
    1. Strip whitespace / women's ``(W)`` suffix.
    2. Look up ``TEAM_ALIASES`` (exact, then accent-folded).
    3. If ``known_teams`` is given, map onto a historical spelling:
       exact → reverse-alias (``Leuven`` ↔ ``Oud-Heverlee Leuven``) →
       last-token / containment → startswith → folded key.
    """
    raw = str(name or "").strip()
    for suf in (" (W)", " W", " Women", " Ladies"):
        if raw.endswith(suf):
            raw = raw[: -len(suf)].strip()
            break

    mapped = TEAM_ALIASES.get(raw, raw)
    if mapped == raw:
        folded_alias = {
            _strip_accents(k).lower(): v for k, v in TEAM_ALIASES.items()
        }
        mapped = folded_alias.get(_strip_accents(raw).lower(), raw)

    if known_teams is None:
        return mapped

    known = [str(t) for t in known_teams]
    known_set = set(known)
    if mapped in known_set:
        return mapped
    if raw in known_set:
        return raw

    # Reverse alias: DB has "Oud-Heverlee Leuven", fixture canonical is "Leuven".
    reverse: dict[str, str] = {}
    for team in known:
        canon = TEAM_ALIASES.get(team)
        if canon is None:
            folded = {
                _strip_accents(k).lower(): v for k, v in TEAM_ALIASES.items()
            }
            canon = folded.get(_strip_accents(team).lower())
        if canon:
            reverse.setdefault(str(canon), team)
            reverse.setdefault(_strip_accents(str(canon)).lower(), team)
    if mapped in reverse:
        return reverse[mapped]
    mapped_fold = _strip_accents(mapped).lower()
    if mapped_fold in reverse:
        return reverse[mapped_fold]

    mapped_l, raw_l = mapped.lower(), raw.lower()
    for team in known:
        tl = team.lower()
        if mapped_l == tl or raw_l == tl:
            return team
        tokens = re.split(r"[\s\-]+", tl)
        if mapped_l in tokens or raw_l in tokens:
            return team
        if mapped_l.startswith(tl) or tl.startswith(mapped_l):
            return team
        if raw_l.startswith(tl) or tl.startswith(raw_l):
            return team

    raw_key = _fold_team_key(mapped)
    for team in known:
        if _fold_team_key(team) == raw_key:
            return team
        team_key = _fold_team_key(team)
        if len(raw_key) >= 5 and raw_key in team_key:
            return team
    return mapped


def canonicalize_team_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Apply :func:`normalize_team_name` to ``HomeTeam`` / ``AwayTeam``.

    Ensures SQLite history and fresh downloads share Flashscore spelling
    (e.g. ``Oud-Heverlee Leuven`` → ``Leuven``).
    """
    if df.empty:
        return df.copy()
    out = df.copy()
    for col in ("HomeTeam", "AwayTeam"):
        if col in out.columns:
            out[col] = out[col].map(
                lambda x: normalize_team_name(str(x)) if pd.notna(x) else x
            )
    return out


def _parse_kickoff(date_series: pd.Series, time_series: pd.Series | None) -> pd.Series:
    """Combine Date + Time into a timezone-naive kickoff Timestamp."""
    dates = pd.to_datetime(date_series, dayfirst=True, format="mixed", errors="coerce")
    if time_series is None:
        return dates
    times = time_series.fillna("00:00").astype(str).str.strip()
    combined = dates.dt.strftime("%Y-%m-%d") + " " + times
    kickoff = pd.to_datetime(combined, errors="coerce")
    missing = kickoff.isna() & dates.notna()
    kickoff = kickoff.where(~missing, dates)
    return kickoff


def clean_fixtures(df: pd.DataFrame, div: str = "E0") -> pd.DataFrame:
    """Clean the multi-league fixtures feed and keep one division (default EPL)."""
    if df.empty:
        return df.copy()

    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    out = out.rename(columns={k: v for k, v in FIXTURE_COLUMN_ALIASES.items() if k in out.columns})

    if "Div" in out.columns:
        out = out[out["Div"].astype(str).str.upper() == div.upper()].copy()

    required = {"Date", "HomeTeam", "AwayTeam"}
    missing = required - set(out.columns)
    if missing:
        raise ValueError(f"Fixtures feed missing columns: {sorted(missing)}")

    out["HomeTeam"] = out["HomeTeam"].astype(str).str.strip()
    out["AwayTeam"] = out["AwayTeam"].astype(str).str.strip()
    time_col = out["Time"] if "Time" in out.columns else None
    out["Kickoff"] = _parse_kickoff(out["Date"], time_col)
    out["Date"] = pd.to_datetime(out["Date"], dayfirst=True, format="mixed", errors="coerce")
    out = out.dropna(subset=["Kickoff", "HomeTeam", "AwayTeam"])

    odds_like = [
        c
        for c in out.columns
        if c.startswith(("B365", "Avg", "Max", "PS", "AHh", "AH"))
        or c.endswith(("H", "D", "A", "O25", "U25", "AHH", "AHA"))
    ]
    for col in odds_like:
        if col in ("HomeTeam", "AwayTeam"):
            continue
        out[col] = pd.to_numeric(out[col], errors="coerce")

    if "AHh" in out.columns:
        out["AHh"] = pd.to_numeric(out["AHh"], errors="coerce")

    out = out.sort_values(["Kickoff", "HomeTeam"]).reset_index(drop=True)
    return out


def _fixture_key(df: pd.DataFrame) -> pd.Series:
    """Match key: calendar date + home + away (string)."""
    dates = pd.to_datetime(df["Date"], errors="coerce").dt.strftime("%Y-%m-%d")
    return dates + "|" + df["HomeTeam"].astype(str) + "|" + df["AwayTeam"].astype(str)


def _pair_key(df: pd.DataFrame) -> pd.Series:
    """Home|Away only (for odds merge when dates differ slightly)."""
    return df["HomeTeam"].astype(str) + "|" + df["AwayTeam"].astype(str)


def _pair_key_folded(df: pd.DataFrame) -> pd.Series:
    """Accent/prefix-insensitive pair key for UWCL Flashscore↔Fotmob merges."""
    return (
        df["HomeTeam"].map(_fold_team_key) + "|" + df["AwayTeam"].map(_fold_team_key)
    )


def fetch_fotmob_league_schedule(
    *,
    fotmob_id: int = 47,
    known_teams: Sequence[str] | None = None,
    only_unplayed: bool = True,
    normalize_women: bool = False,
) -> pd.DataFrame:
    """Download league schedule from Fotmob (mirrors Flashscore calendar).

    Parameters
    ----------
    fotmob_id:
        ``47`` = EPL, ``9375`` = UWCL.
    normalize_women:
        Strip ``(W)`` suffixes (UWCL).
    """
    payload = _fetch_fotmob_league_payload(int(fotmob_id))
    fixtures_block = payload.get("fixtures") or {}
    matches = fixtures_block.get("allMatches") or []
    if not matches:
        return pd.DataFrame(
            columns=["Date", "Kickoff", "HomeTeam", "AwayTeam", "Round", "Source"]
        )

    first_idx = 0
    if only_unplayed:
        meta = fixtures_block.get("firstUnplayedMatch") or {}
        first_idx = int(meta.get("firstUnplayedMatchIndex") or 0)

    rows: list[dict] = []
    for m in matches[first_idx:]:
        status = m.get("status") or {}
        if only_unplayed and status.get("finished"):
            continue
        if status.get("cancelled"):
            continue
        utc = status.get("utcTime")
        kickoff = pd.to_datetime(utc, utc=True, errors="coerce")
        if pd.isna(kickoff):
            continue
        kickoff_naive = kickoff.tz_convert("UTC").tz_localize(None)
        raw_home = (m.get("home") or {}).get("name", "")
        raw_away = (m.get("away") or {}).get("name", "")
        if normalize_women:
            raw_home = _normalize_uwcl_team(raw_home)
            raw_away = _normalize_uwcl_team(raw_away)
        home = normalize_team_name(raw_home, known_teams)
        away = normalize_team_name(raw_away, known_teams)
        rows.append(
            {
                "Date": kickoff_naive.normalize(),
                "Kickoff": kickoff_naive,
                "HomeTeam": home,
                "AwayTeam": away,
                "Round": m.get("round") or m.get("roundName"),
                "Source": "flashscore/fotmob",
                "MatchId": str(m.get("id", "")),
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["Kickoff", "HomeTeam"]).reset_index(drop=True)


def fetch_fotmob_epl_schedule(
    *,
    known_teams: Sequence[str] | None = None,
    only_unplayed: bool = True,
) -> pd.DataFrame:
    """Download the full EPL schedule from Fotmob (same calendar as Flashscore)."""
    return fetch_fotmob_league_schedule(
        fotmob_id=47,
        known_teams=known_teams,
        only_unplayed=only_unplayed,
        normalize_women=False,
    )


def _load_fd_odds_fixtures(div: str = "E0") -> pd.DataFrame:
    """Raw football-data fixtures.csv (may be stale) for odds columns."""
    try:
        text = _fetch_csv_text(FIXTURES_URL)
        raw = pd.read_csv(StringIO(text))
        return clean_fixtures(raw, div=div)
    except Exception:  # noqa: BLE001
        return pd.DataFrame()


def american_to_decimal(american: float | int | str | None) -> float:
    """Convert American moneyline / price to European decimal odds.

    Examples
    --------
    ``+150`` → ``2.50``, ``-200`` → ``1.50``, ``100`` → ``2.00``.
    """
    if american is None or (isinstance(american, float) and pd.isna(american)):
        return float("nan")
    if isinstance(american, str):
        s = american.strip().replace("−", "-")
        if not s or s.lower() in {"none", "null", "nan"}:
            return float("nan")
        try:
            american = float(s)
        except ValueError:
            return float("nan")
    a = float(american)
    if a == 0:
        return float("nan")
    if a > 0:
        return round(1.0 + a / 100.0, 3)
    return round(1.0 + 100.0 / abs(a), 3)


# ---------------------------------------------------------------------------
# Flashscore Odds tab
# https://www.flashscore.com/match/.../odds/...  (GraphQL hash=oce)
# ---------------------------------------------------------------------------

_FS_SEP_FIELD = "\xac"
_FS_SEP_KV = "\xf7"


def _flashscore_parse_fields(section: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for part in section.split(_FS_SEP_FIELD):
        if _FS_SEP_KV in part:
            k, _, v = part.partition(_FS_SEP_KV)
            out[k] = v
    return out


def _flashscore_extract_initial_feed(html: str, key: str) -> str:
    """Extract ``cjs.initialFeeds["key"].data`` template string from page HTML."""
    pat = rf'cjs\.initialFeeds\["{re.escape(key)}"\]\s*=\s*\{{\s*data:\s*`([^`]*)`'
    m = re.search(pat, html, re.S)
    return m.group(1) if m else ""


def fetch_flashscore_fixture_events(
    *,
    fixtures_url: str = FLASHSCORE_FIXTURES_URL,
    known_teams: Sequence[str] | None = None,
    only_unplayed: bool = True,
    normalize_women: bool = False,
) -> pd.DataFrame:
    """Parse upcoming events (with Flashscore ``event_id``) from a fixtures page."""
    html = _fetch_text(
        fixtures_url,
        referer="https://www.flashscore.com/",
        extra_headers={"Accept": "text/html", "x-fsign": FLASHSCORE_FSIGN},
    )
    raw = _flashscore_extract_initial_feed(html, "summary-fixtures")
    if not raw:
        return pd.DataFrame(
            columns=["Date", "Kickoff", "HomeTeam", "AwayTeam", "FlashscoreEventId", "Source"]
        )

    rows: list[dict] = []
    for sec in raw.split("~"):
        fields = _flashscore_parse_fields(sec)
        if "AA" not in fields or "AE" not in fields:
            continue
        status = fields.get("AB", "")
        # Flashscore: 1=scheduled, 2=live, 3=finished
        if only_unplayed and status == "3":
            continue
        ts = int(fields.get("AD") or 0)
        if not ts:
            continue
        kickoff = datetime.fromtimestamp(ts, tz=timezone.utc).replace(tzinfo=None)
        home_raw = fields.get("AE", "")
        away_raw = fields.get("AF", "")
        if normalize_women:
            home_raw = _normalize_uwcl_team(home_raw)
            away_raw = _normalize_uwcl_team(away_raw)
        home = normalize_team_name(home_raw, known_teams)
        away = normalize_team_name(away_raw, known_teams)
        rows.append(
            {
                "Date": pd.Timestamp(kickoff).normalize(),
                "Kickoff": pd.Timestamp(kickoff),
                "HomeTeam": home,
                "AwayTeam": away,
                "FlashscoreEventId": fields["AA"],
                "Source": "flashscore",
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["Kickoff", "HomeTeam"]).drop_duplicates(
        subset=["FlashscoreEventId"], keep="last"
    ).reset_index(drop=True)


def fetch_flashscore_epl_fixture_events(
    *,
    known_teams: Sequence[str] | None = None,
    only_unplayed: bool = True,
) -> pd.DataFrame:
    """Parse upcoming EPL events (with Flashscore ``event_id``) from fixtures page."""
    return fetch_flashscore_fixture_events(
        fixtures_url=FLASHSCORE_FIXTURES_URL,
        known_teams=known_teams,
        only_unplayed=only_unplayed,
        normalize_women=False,
    )


def _fs_handicap_value(item: dict) -> float:
    hc = item.get("handicap")
    if isinstance(hc, dict):
        try:
            return float(hc.get("value"))
        except (TypeError, ValueError):
            return float("nan")
    if hc is None:
        return float("nan")
    try:
        return float(hc)
    except (TypeError, ValueError):
        return float("nan")


def _fs_pick_bookmaker(
    markets: list[dict], prefer: Sequence[str] = FLASHSCORE_BOOKMAKER_PREFER
) -> tuple[str | None, dict | None, str | None]:
    by_bk: dict[str, dict] = {}
    for m in markets:
        by_bk[str(m.get("bookmakerId"))] = m
    for pid in prefer:
        if pid in by_bk:
            return pid, by_bk[pid], None
    if by_bk:
        pid = next(iter(by_bk))
        return pid, by_bk[pid], None
    return None, None, None


def _fs_parse_1x2(market: dict) -> tuple[float, float, float]:
    """HOME_DRAW_AWAY selections → (home, draw, away) decimal odds."""
    sels = market.get("odds") or []
    if len(sels) < 3:
        return float("nan"), float("nan"), float("nan")
    # Flashscore order: home participant, away participant, draw (null id)
    try:
        home = float(sels[0]["value"])
        away = float(sels[1]["value"])
        draw = float(sels[2]["value"])
        return home, draw, away
    except (TypeError, ValueError, KeyError):
        return float("nan"), float("nan"), float("nan")


def _fs_parse_ou(market: dict) -> tuple[float, float, float]:
    """OVER_UNDER → (line, over, under).

    Chọn **mốc chính** = cặp Over/Under có biên độ hai cửa nhỏ nhất
    (``|odds_over - odds_under|`` min). Ví dụ Arsenal–Leeds thường là **2.75**
    (≈1.95/1.90) chứ không hardcode 2.5.
    """
    overs: dict[float, float] = {}
    unders: dict[float, float] = {}
    for s in market.get("odds") or []:
        line = _fs_handicap_value(s)
        if pd.isna(line):
            continue
        try:
            val = float(s["value"])
        except (TypeError, ValueError, KeyError):
            continue
        sel = str(s.get("selection") or "").upper()
        if sel == "OVER":
            overs[line] = val
        elif sel == "UNDER":
            unders[line] = val
    common = sorted(set(overs) & set(unders))
    if not common:
        return float("nan"), float("nan"), float("nan")

    def _score(line: float) -> tuple[float, float]:
        o, u = overs[line], unders[line]
        # 1) biên độ hai cửa nhỏ nhất; 2) tie-break: min(o,u) càng cao càng tốt
        return (abs(o - u), -min(o, u))

    line = min(common, key=_score)
    return float(line), float(overs[line]), float(unders[line])


def _fs_parse_ah(
    market: dict,
    *,
    home_pid: str | None = None,
    away_pid: str | None = None,
) -> tuple[float, float, float]:
    """ASIAN_HANDICAP → (home_line, home_odds, away_odds).

    Chọn **mốc chính** = cặp chấp đối xứng có biên độ hai cửa nhỏ nhất
    (``|odds_home - odds_away|`` min), không hardcode ``-0.5``.
    Dùng participant id từ 1X2 để gắn đúng cửa nhà / khách.
    """
    home_quotes: list[tuple[float, float]] = []
    away_quotes: list[tuple[float, float]] = []
    for s in market.get("odds") or []:
        pid = s.get("eventParticipantId")
        line = _fs_handicap_value(s)
        if pid is None or pd.isna(line):
            continue
        try:
            val = float(s["value"])
        except (TypeError, ValueError, KeyError):
            continue
        pid_s = str(pid)
        if home_pid and pid_s == home_pid:
            home_quotes.append((line, val))
        elif away_pid and pid_s == away_pid:
            away_quotes.append((line, val))

    if not home_quotes or not away_quotes:
        return float("nan"), float("nan"), float("nan")

    away_by_line = {lb: ob for lb, ob in away_quotes}
    candidates: list[tuple[float, float, float, float]] = []
    for la, oa in home_quotes:
        lb = -la
        if lb not in away_by_line:
            match = min(away_by_line.keys(), key=lambda x: abs(x - lb), default=None)
            if match is None or abs(match - lb) > 1e-6:
                continue
            lb = match
        ob = away_by_line[lb]
        # score: biên độ nhỏ nhất; tie-break min(odds) cao hơn
        bal = abs(oa - ob)
        candidates.append((bal, -min(oa, ob), la, oa, ob))

    if not candidates:
        return float("nan"), float("nan"), float("nan")
    _, _, la, oa, ob = min(candidates, key=lambda x: (x[0], x[1]))
    return float(la), float(oa), float(ob)


def fetch_flashscore_match_odds(
    event_id: str,
    *,
    geo_code: str = "GB",
    geo_sub: str = "GBENG",
    project_id: str = "2",
    bookmaker_names: dict[str, str] | None = None,
) -> dict[str, float | str]:
    """Fetch Flashscore Odds-tab prices for one event (``mid`` / ``eventId``).

    Endpoint mirrors the browser tab, e.g.
    https://www.flashscore.com/match/football/.../odds/over-under/full-time/?mid=xtmHKGT0
    """
    params = {
        "_hash": "oce",
        "eventId": event_id,
        "projectId": project_id,
        "geoIpCode": geo_code,
        "geoIpSubdivisionCode": geo_sub,
    }
    url = f"{FLASHSCORE_ODDS_URL}?{urlencode(params)}"
    payload = _fetch_json(
        url,
        referer="https://www.flashscore.com/",
        extra_headers={
            "Origin": "https://www.flashscore.com",
            "x-fsign": FLASHSCORE_FSIGN,
            "X-Fsign": FLASHSCORE_FSIGN,
        },
    )
    event = (payload.get("data") or {}).get("findOddsByEventId") or {}
    if not event:
        return {}

    bk_map = bookmaker_names or {}
    for entry in (event.get("settings") or {}).get("bookmakers") or []:
        inner = entry.get("bookmaker") or {}
        bid = str(inner.get("id", ""))
        if bid:
            bk_map[bid] = str(inner.get("name", bid))

    odds_list = event.get("odds") or []
    ft = [o for o in odds_list if o.get("bettingScope") == "FULL_TIME"]

    hda = [o for o in ft if o.get("bettingType") == "HOME_DRAW_AWAY"]
    ou = [o for o in ft if o.get("bettingType") == "OVER_UNDER"]
    ah = [o for o in ft if o.get("bettingType") == "ASIAN_HANDICAP"]

    _, hda_m, _ = _fs_pick_bookmaker(hda)
    _, ou_m, _ = _fs_pick_bookmaker(ou)
    _, ah_m, _ = _fs_pick_bookmaker(ah)

    out: dict[str, float | str] = {"FlashscoreEventId": event_id}
    provider_bits: list[str] = []

    home_pid = away_pid = None
    if hda_m:
        h, d, a = _fs_parse_1x2(hda_m)
        out["B365H"], out["B365D"], out["B365A"] = h, d, a
        provider_bits.append(bk_map.get(str(hda_m.get("bookmakerId")), "book"))
        sels = hda_m.get("odds") or []
        if len(sels) >= 2:
            home_pid = sels[0].get("eventParticipantId")
            away_pid = sels[1].get("eventParticipantId")
            if home_pid is not None:
                home_pid = str(home_pid)
            if away_pid is not None:
                away_pid = str(away_pid)
    if ou_m:
        line, over, under = _fs_parse_ou(ou_m)
        out["OU_Line"], out["OddsOver"], out["OddsUnder"] = line, over, under
        if pd.notna(line) and abs(float(line) - 2.5) < 1e-9:
            out["B365_O25"], out["B365_U25"] = over, under
    if ah_m:
        ahh, oh, oa = _fs_parse_ah(ah_m, home_pid=home_pid, away_pid=away_pid)
        out["AHh"], out["B365AHH"], out["B365AHA"] = ahh, oh, oa

    label = provider_bits[0] if provider_bits else "Flashscore"
    out["OddsProvider"] = f"Flashscore/{label}"
    return out


def fetch_flashscore_odds(
    *,
    fixtures_url: str = FLASHSCORE_FIXTURES_URL,
    known_teams: Sequence[str] | None = None,
    max_events: int = 24,
    request_delay_s: float = 0.25,
    normalize_women: bool = False,
) -> pd.DataFrame:
    """Load Flashscore fixture event IDs then pull Odds-tab markets for each.

    Prefers **bet365** (bookmaker id 16) as listed on the Flashscore GB comparison.
    """
    empty_cols = [
        "HomeTeam",
        "AwayTeam",
        "Kickoff",
        "FlashscoreEventId",
        "B365H",
        "B365D",
        "B365A",
        "B365_O25",
        "B365_U25",
        "OU_Line",
        "OddsOver",
        "OddsUnder",
        "AHh",
        "B365AHH",
        "B365AHA",
        "OddsProvider",
    ]
    events = fetch_flashscore_fixture_events(
        fixtures_url=fixtures_url,
        known_teams=known_teams,
        only_unplayed=True,
        normalize_women=normalize_women,
    )
    if events.empty:
        return pd.DataFrame(columns=empty_cols)

    events = events.head(max_events)
    rows: list[dict] = []
    for idx, (_, ev) in enumerate(events.iterrows()):
        if idx and request_delay_s > 0:
            time.sleep(request_delay_s)
        try:
            odds = fetch_flashscore_match_odds(str(ev["FlashscoreEventId"]))
        except Exception:  # noqa: BLE001
            continue
        if not odds or pd.isna(odds.get("B365H", float("nan"))):
            continue
        row = {
            "HomeTeam": ev["HomeTeam"],
            "AwayTeam": ev["AwayTeam"],
            "Kickoff": ev["Kickoff"],
            **odds,
        }
        rows.append(row)

    if not rows:
        return pd.DataFrame(columns=empty_cols)
    return pd.DataFrame(rows).reset_index(drop=True)


def fetch_flashscore_epl_odds(
    *,
    known_teams: Sequence[str] | None = None,
    max_events: int = 24,
    request_delay_s: float = 0.25,
) -> pd.DataFrame:
    """EPL convenience wrapper around :func:`fetch_flashscore_odds`."""
    return fetch_flashscore_odds(
        fixtures_url=FLASHSCORE_FIXTURES_URL,
        known_teams=known_teams,
        max_events=max_events,
        request_delay_s=request_delay_s,
        normalize_women=False,
    )


def _espn_close_american(block: dict | None) -> float | None:
    """Read close (fallback open) American price from ESPN moneyline/total/spread node."""
    if not isinstance(block, dict):
        return None
    for key in ("close", "open"):
        node = block.get(key) or {}
        odds = node.get("odds")
        if odds is not None:
            return odds
    return None


def _espn_close_line(block: dict | None) -> float:
    """Parse close line string like ``'-1.5'``, ``'+0.5'``, ``'o2.5'``."""
    if not isinstance(block, dict):
        return float("nan")
    for key in ("close", "open"):
        node = block.get(key) or {}
        line = node.get("line")
        if line is None:
            continue
        s = str(line).strip().lstrip("ouOU")
        try:
            return float(s)
        except ValueError:
            continue
    return float("nan")


def fetch_espn_epl_odds(
    dates: Iterable[pd.Timestamp | datetime | str] | None = None,
    *,
    known_teams: Sequence[str] | None = None,
    days_ahead: int = 28,
    max_dates: int = 24,
) -> pd.DataFrame:
    """Fetch live EPL book odds from ESPN scoreboard (DraftKings lines).

    Endpoint (no API key)::

        site.web.api.espn.com/.../soccer/eng.1/scoreboard?dates=YYYYMMDD

    Returns rows with football-data-compatible columns
    (``B365H/D/A``, ``B365_O25/U25``, ``AHh``, ``B365AHH/AHA``) plus
    ``OU_Line``, ``OddsOver``, ``OddsUnder``, ``OddsProvider``.
    """
    empty_cols = [
        "HomeTeam",
        "AwayTeam",
        "Kickoff",
        "B365H",
        "B365D",
        "B365A",
        "B365_O25",
        "B365_U25",
        "OU_Line",
        "OddsOver",
        "OddsUnder",
        "AHh",
        "B365AHH",
        "B365AHA",
        "OddsProvider",
        "EspnEventId",
    ]
    def _naive_day(value: pd.Timestamp | datetime | str) -> pd.Timestamp:
        ts = pd.Timestamp(value)
        if getattr(ts, "tzinfo", None) is not None:
            ts = ts.tz_convert("UTC").tz_localize(None)
        return ts.normalize()

    ref_now = _naive_day(pd.Timestamp.now(tz="UTC"))
    if dates is None:
        date_list = [ref_now + pd.Timedelta(days=i) for i in range(max(1, days_ahead))]
    else:
        date_list = sorted(
            {
                _naive_day(d)
                for d in dates
                if d is not None and not pd.isna(pd.Timestamp(d))
            }
        )
        if days_ahead is not None and date_list:
            cutoff = ref_now + pd.Timedelta(days=days_ahead)
            date_list = [d for d in date_list if d <= cutoff]

    date_list = date_list[:max_dates]
    rows: list[dict] = []

    for day in date_list:
        ymd = pd.Timestamp(day).strftime("%Y%m%d")
        url = f"{ESPN_EPL_SCOREBOARD_URL}?dates={ymd}"
        try:
            payload = _fetch_json(url, referer="https://www.espn.com/")
        except Exception:  # noqa: BLE001 — skip one bad day, keep others
            continue

        for ev in payload.get("events") or []:
            comps = ev.get("competitions") or []
            if not comps:
                continue
            comp = comps[0]
            odds_list = comp.get("odds") or []
            odds = next((o for o in odds_list if isinstance(o, dict)), None)
            if not odds:
                continue

            home_name = away_name = ""
            for c in comp.get("competitors") or []:
                team = (c.get("team") or {}).get("displayName") or ""
                if c.get("homeAway") == "home":
                    home_name = team
                elif c.get("homeAway") == "away":
                    away_name = team
            if not home_name or not away_name:
                continue

            home = normalize_team_name(home_name, known_teams)
            away = normalize_team_name(away_name, known_teams)

            ml = odds.get("moneyline") or {}
            odds_h = american_to_decimal(_espn_close_american(ml.get("home")))
            odds_a = american_to_decimal(_espn_close_american(ml.get("away")))
            odds_d = american_to_decimal(_espn_close_american(ml.get("draw")))
            if pd.isna(odds_d):
                draw_block = odds.get("drawOdds") or {}
                odds_d = american_to_decimal(draw_block.get("moneyLine"))

            ou_line = odds.get("overUnder")
            try:
                ou_line_f = float(ou_line) if ou_line is not None else float("nan")
            except (TypeError, ValueError):
                ou_line_f = float("nan")

            total = odds.get("total") or {}
            odds_over = american_to_decimal(_espn_close_american(total.get("over")))
            odds_under = american_to_decimal(_espn_close_american(total.get("under")))
            if pd.isna(ou_line_f):
                ou_line_f = _espn_close_line(total.get("over"))

            ps = odds.get("pointSpread") or {}
            ah_home = _espn_close_line(ps.get("home"))
            odds_ahh = american_to_decimal(_espn_close_american(ps.get("home")))
            odds_aha = american_to_decimal(_espn_close_american(ps.get("away")))

            kickoff = pd.to_datetime(comp.get("date") or ev.get("date"), errors="coerce")
            if pd.notna(kickoff) and getattr(kickoff, "tzinfo", None) is not None:
                kickoff = kickoff.tz_convert("UTC").tz_localize(None)

            provider = (odds.get("provider") or {}).get("name") or "DraftKings"
            row = {
                "HomeTeam": home,
                "AwayTeam": away,
                "Kickoff": kickoff,
                "B365H": odds_h,
                "B365D": odds_d,
                "B365A": odds_a,
                "OU_Line": ou_line_f,
                "OddsOver": odds_over,
                "OddsUnder": odds_under,
                "AHh": ah_home,
                "B365AHH": odds_ahh,
                "B365AHA": odds_aha,
                "OddsProvider": f"ESPN/{provider}",
                "EspnEventId": str(ev.get("id", "")),
                "B365_O25": float("nan"),
                "B365_U25": float("nan"),
            }
            # Keep O25 aliases when the market is the classic 2.5 line.
            if pd.notna(ou_line_f) and abs(float(ou_line_f) - 2.5) < 1e-9:
                row["B365_O25"] = odds_over
                row["B365_U25"] = odds_under
            rows.append(row)

    if not rows:
        return pd.DataFrame(columns=empty_cols)

    out = pd.DataFrame(rows)
    out = out.sort_values(["Kickoff", "HomeTeam"]).drop_duplicates(
        subset=["HomeTeam", "AwayTeam"], keep="last"
    )
    return out.reset_index(drop=True)


def _merge_odds_onto_fixtures(
    fixtures: pd.DataFrame,
    odds_df: pd.DataFrame,
    *,
    prefer_new: bool = False,
) -> pd.DataFrame:
    """Left-merge odds onto fixtures by team pair (+ kickoff fallback).

    Primary key is accent/prefix-folded ``Home|Away`` so UWCL names like
    ``BK Häcken`` (Fotmob) match ``Hacken`` (Flashscore). Exact string pairs
    and kickoff (±1 min) fill any remaining gaps.
    """
    if fixtures.empty or odds_df.empty:
        return fixtures

    skip = {
        "HomeTeam",
        "AwayTeam",
        "Kickoff",
        "Date",
        "Time",
        "Div",
        "Source",
        "Round",
        "MatchId",
        "is_future",
        "is_played",
        "_pair",
        "_pair_exact",
        "_kick",
    }
    odds_cols = [c for c in odds_df.columns if c not in skip]

    def _apply_merge(
        base: pd.DataFrame,
        donor: pd.DataFrame,
        key: str,
    ) -> pd.DataFrame:
        right = donor.dropna(subset=[key]).drop_duplicates(key, keep="last")
        cols = [key] + [c for c in odds_cols if c in right.columns]
        incoming = right[cols]
        if prefer_new:
            drop_cols = [c for c in odds_cols if c in base.columns and c in incoming.columns]
            left = base.drop(columns=drop_cols, errors="ignore")
            return left.merge(incoming, on=key, how="left")
        rename = {c: f"{c}__new" for c in odds_cols if c in incoming.columns}
        incoming = incoming.rename(columns=rename)
        out = base.merge(incoming, on=key, how="left")
        for col, new_col in rename.items():
            if col in out.columns:
                out[col] = out[col].fillna(out[new_col])
            else:
                out[col] = out[new_col]
            out = out.drop(columns=[new_col], errors="ignore")
        return out

    left = fixtures.copy()
    right = odds_df.copy()
    left["_pair"] = _pair_key_folded(left)
    right["_pair"] = _pair_key_folded(right)
    out = _apply_merge(left, right, "_pair")

    # Exact-name pass for rows still missing 1X2
    if "B365H" in out.columns and out["B365H"].isna().any():
        left_e = out.copy()
        right_e = odds_df.copy()
        left_e["_pair_exact"] = _pair_key(left_e)
        right_e["_pair_exact"] = _pair_key(right_e)
        # Merge only onto missing rows via fillna
        tmp = _apply_merge(
            left_e.drop(columns=[c for c in odds_cols if c in left_e.columns], errors="ignore"),
            right_e,
            "_pair_exact",
        )
        for col in odds_cols:
            if col in tmp.columns:
                if col in out.columns:
                    out[col] = out[col].fillna(tmp[col])
                else:
                    out[col] = tmp[col]

    # Kickoff fallback
    if (
        "Kickoff" in out.columns
        and "Kickoff" in odds_df.columns
        and "B365H" in out.columns
        and out["B365H"].isna().any()
    ):
        right_k = odds_df.copy()
        right_k["_kick"] = pd.to_datetime(right_k["Kickoff"], errors="coerce").dt.floor(
            "min"
        )
        miss_idx = out.index[out["B365H"].isna()]
        if len(miss_idx):
            sub = out.loc[miss_idx].copy()
            sub["_kick"] = pd.to_datetime(sub["Kickoff"], errors="coerce").dt.floor("min")
            sub = sub.drop(columns=[c for c in odds_cols if c in sub.columns], errors="ignore")
            got = _apply_merge(sub, right_k, "_kick")
            for col in odds_cols:
                if col in got.columns:
                    out.loc[miss_idx, col] = got[col].to_numpy()

    return out.drop(columns=["_pair", "_pair_exact", "_kick"], errors="ignore")


def load_upcoming_fixtures(
    *,
    league: str = "EPL",
    div: str = "E0",
    results: pd.DataFrame | None = None,
    only_future: bool = True,
    exclude_played: bool = True,
    now: datetime | None = None,
    include_odds: bool = True,
) -> pd.DataFrame:
    """Load upcoming fixtures + bookmaker odds for ``EPL`` or ``UWCL``.

    Schedule source
        Fotmob league API — mirrors Flashscore calendars
        (EPL / UEFA Women's Champions League).

    Odds sources (merged in order)
        1. **Flashscore Odds tab** (GraphQL ``oce``) — prefer bet365.
        2. ESPN scoreboard JSON (DraftKings) — **EPL only**.
        3. football-data.co.uk/fixtures.csv — **EPL only**.
    """
    code = normalize_league(league)
    cfg = LEAGUE_CONFIG[code]
    fotmob_id = int(cfg["fotmob_id"])  # type: ignore[arg-type]
    fs_url = str(cfg["flashscore_fixtures_url"])
    is_uwcl = code == "UWCL"
    fd_div = cfg.get("fd_div") or div

    known = list_teams(results) if results is not None and not results.empty else None
    schedule = fetch_fotmob_league_schedule(
        fotmob_id=fotmob_id,
        known_teams=known,
        only_unplayed=exclude_played,
        normalize_women=is_uwcl,
    )

    ref = now or datetime.now(timezone.utc).replace(tzinfo=None)
    if getattr(ref, "tzinfo", None) is not None:
        ref = ref.replace(tzinfo=None)
    ref_ts = pd.Timestamp(ref)

    odds_sources: list[str] = []

    if schedule.empty:
        # Fallback: football-data feed only (EPL); UWCL has no FD division.
        if is_uwcl:
            fixtures = pd.DataFrame()
        else:
            fixtures = _load_fd_odds_fixtures(div=str(fd_div))
            fixtures["Source"] = "football-data"
            if not fixtures.empty and "B365H" in fixtures.columns:
                odds_sources.append("football-data")
    else:
        fixtures = schedule.copy()
        if include_odds:
            # 1) Flashscore Odds tab
            try:
                fs_df = fetch_flashscore_odds(
                    fixtures_url=fs_url,
                    known_teams=known,
                    max_events=40 if is_uwcl else 24,
                    normalize_women=is_uwcl,
                )
            except Exception:  # noqa: BLE001
                fs_df = pd.DataFrame()
            if not fs_df.empty:
                fixtures = _merge_odds_onto_fixtures(fixtures, fs_df, prefer_new=True)
                odds_sources.append("Flashscore")

            # 2) ESPN / DraftKings — EPL only
            if not is_uwcl:
                try:
                    espn_df = fetch_espn_epl_odds(
                        dates=fixtures["Kickoff"],
                        known_teams=known,
                        days_ahead=28,
                    )
                except Exception:  # noqa: BLE001
                    espn_df = pd.DataFrame()
                if not espn_df.empty:
                    before = (
                        int(fixtures["B365H"].notna().sum())
                        if "B365H" in fixtures.columns
                        else 0
                    )
                    fixtures = _merge_odds_onto_fixtures(
                        fixtures, espn_df, prefer_new=False
                    )
                    after = (
                        int(fixtures["B365H"].notna().sum())
                        if "B365H" in fixtures.columns
                        else 0
                    )
                    if after > before:
                        odds_sources.append("ESPN/DraftKings")
                    if "OddsProvider" in fixtures.columns:
                        mask = fixtures["OddsProvider"].isna() & fixtures["B365H"].notna()
                        fixtures.loc[mask, "OddsProvider"] = "ESPN/DraftKings"

                # 3) football-data fixtures.csv
                odds_df = _load_fd_odds_fixtures(div=str(fd_div))
                if not odds_df.empty:
                    before = (
                        fixtures["B365H"].notna().sum()
                        if "B365H" in fixtures.columns
                        else 0
                    )
                    fixtures = _merge_odds_onto_fixtures(
                        fixtures, odds_df, prefer_new=False
                    )
                    after = (
                        fixtures["B365H"].notna().sum()
                        if "B365H" in fixtures.columns
                        else 0
                    )
                    if after > before:
                        odds_sources.append("football-data")
                if "OddsProvider" in fixtures.columns:
                    mask_fd = fixtures["OddsProvider"].isna() & fixtures["B365H"].notna()
                    fixtures.loc[mask_fd, "OddsProvider"] = "football-data"
                elif "B365H" in fixtures.columns:
                    fixtures["OddsProvider"] = pd.NA
                    fixtures.loc[fixtures["B365H"].notna(), "OddsProvider"] = (
                        "football-data"
                    )

    if fixtures.empty:
        fixtures.attrs["source_url"] = fs_url
        fixtures.attrs["league"] = code
        fixtures.attrs["as_of"] = ref_ts
        fixtures.attrs["odds_sources"] = odds_sources
        return fixtures

    fixtures["is_future"] = fixtures["Kickoff"] >= ref_ts
    fixtures["is_played"] = False
    if results is not None and not results.empty:
        played = set(_fixture_key(results.dropna(subset=["HomeTeam", "AwayTeam"])))
        keys = _fixture_key(fixtures)
        fixtures["is_played"] = keys.isin(played)

    if exclude_played:
        fixtures = fixtures.loc[~fixtures["is_played"]].reset_index(drop=True)
    if only_future:
        fixtures = fixtures.loc[fixtures["is_future"]].reset_index(drop=True)

    fixtures.attrs["source_url"] = fs_url
    fixtures.attrs["odds_url"] = FLASHSCORE_ODDS_URL
    fixtures.attrs["odds_sources"] = odds_sources or ["none"]
    fixtures.attrs["league"] = code
    fixtures.attrs["as_of"] = ref_ts
    return fixtures


if __name__ == "__main__":
    data = load_epl_data(n_seasons=1, force_refresh=False)
    print(f"Loaded {len(data)} matches · source={data.attrs.get('data_source')}")
    print(f"DB: {data.attrs.get('db_path')}")
    print(f"Date range: {data['Date'].min().date()} → {data['Date'].max().date()}")
    print(f"Teams ({len(list_teams(data))}): {', '.join(list_teams(data)[:5])}…")
    print(data[["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]].head())

    upcoming = load_upcoming_fixtures(results=data, only_future=True)
    print(f"\nUpcoming EPL fixtures: {len(upcoming)}")
    print("odds_sources:", upcoming.attrs.get("odds_sources"))
    if not upcoming.empty:
        cols = [
            c
            for c in (
                "Kickoff",
                "HomeTeam",
                "AwayTeam",
                "Round",
                "B365H",
                "B365D",
                "B365A",
                "OU_Line",
                "OddsProvider",
                "Source",
            )
            if c in upcoming.columns
        ]
        with_odds = upcoming["B365H"].notna().sum() if "B365H" in upcoming.columns else 0
        print(f"fixtures with 1X2 odds: {with_odds}/{len(upcoming)}")
        print(upcoming[cols].head(15).to_string(index=False))