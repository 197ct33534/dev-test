r"""Walk-forward backtester for EPL value bets.

Simulates betting day-by-day: at each match date ``t``, fit Dixon–Coles
(and optionally LightGBM) on matches **strictly before** ``t``, calibrate
probabilities on past scored fixtures, apply safety filters (odds band +
stake cap), place Kelly-sized bets when ``EV >= min_ev``, then settle PnL.

Persists simulated bets in SQLite table ``bet_history``.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Literal, Sequence

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from src.calibration import (
    apply_commission,
    closing_line_value,
    format_calibration_report,
    summarize_predictions,
)
from src.config import (
    DEFAULT_KELLY_FRACTION,
    DEFAULT_MIN_EV,
    DEFAULT_W_ML,
    MAX_STAKE_PCT,
)
from src.data_loader import DEFAULT_DB_PATH
from src.dixon_coles import (
    DixonColesModel,
    _combine_settlements,
    _mirror_settlement,
    _settle_ah_home,
    _settle_over,
    _settle_under,
)
from src.models import ensemble_weight_for_sample
from src.recommender import (
    ODDS_COLUMN_MAPS,
    ValueBetRecommender,
    expected_value,
    fractional_kelly,
)

try:
    from src.ml_model import EPLMachineLearningModel
except ImportError:  # pragma: no cover
    EPLMachineLearningModel = None  # type: ignore[misc, assignment]

BET_HISTORY_TABLE = "bet_history"
ResultLabel = Literal["WIN", "LOSS", "PUSH"]
CalibMethod = Literal["none", "isotonic", "platt"]
MarketName = Literal["1X2", "OU", "AH"]

_SELECTION_TO_FTR: dict[str, str] = {
    "Home": "H",
    "Draw": "D",
    "Away": "A",
    "H": "H",
    "D": "D",
    "A": "A",
}

DEFAULT_MIN_ODDS = 1.40
DEFAULT_MAX_ODDS = 3.50
DEFAULT_MAX_STAKE_PCT = MAX_STAKE_PCT  # 1% bankroll cap (from config)
DEFAULT_COMMISSION_PCT = 0.0  # exchange-style cut on winning PnL


# ---------------------------------------------------------------------------
# Probability calibration (Isotonic / Platt)
# ---------------------------------------------------------------------------


@dataclass
class ProbabilityCalibrator:
    """Online calibrator for binary event probabilities.

    Fits on past ``(p_raw, y)`` pairs scored in earlier walk-forward days
    (no look-ahead). Shrinks over-confident longshot probabilities.
    """

    method: CalibMethod = "isotonic"
    min_samples: int = 80

    _iso: IsotonicRegression | None = field(default=None, init=False, repr=False)
    _platt: LogisticRegression | None = field(default=None, init=False, repr=False)
    fitted_: bool = field(default=False, init=False)
    n_samples_: int = field(default=0, init=False)

    def fit(self, p: Sequence[float], y: Sequence[int | float]) -> ProbabilityCalibrator:
        """Fit calibrator on historical predictions vs outcomes."""
        p_arr = np.asarray(p, dtype=float)
        y_arr = np.asarray(y, dtype=float)
        mask = np.isfinite(p_arr) & np.isfinite(y_arr)
        p_arr, y_arr = p_arr[mask], y_arr[mask]
        self.n_samples_ = int(len(p_arr))
        self.fitted_ = False
        self._iso = None
        self._platt = None

        if self.method == "none" or self.n_samples_ < self.min_samples:
            return self
        if len(np.unique(y_arr)) < 2:
            return self

        p_clip = np.clip(p_arr, 1e-4, 1.0 - 1e-4)
        if self.method == "isotonic":
            iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
            iso.fit(p_clip, y_arr)
            self._iso = iso
            self.fitted_ = True
        elif self.method == "platt":
            # Platt = logistic regression on logit(p)
            logit = np.log(p_clip / (1.0 - p_clip)).reshape(-1, 1)
            clf = LogisticRegression(solver="lbfgs", max_iter=500)
            clf.fit(logit, y_arr.astype(int))
            self._platt = clf
            self.fitted_ = True
        return self

    def transform(self, p: float) -> float:
        """Map raw probability → calibrated probability in (0, 1)."""
        p = float(np.clip(p, 1e-6, 1.0 - 1e-6))
        if not self.fitted_ or self.method == "none":
            return p
        if self.method == "isotonic" and self._iso is not None:
            return float(np.clip(self._iso.predict([p])[0], 1e-6, 1.0 - 1e-6))
        if self.method == "platt" and self._platt is not None:
            logit = np.log(p / (1.0 - p)).reshape(1, -1)
            return float(np.clip(self._platt.predict_proba(logit)[0, 1], 1e-6, 1.0 - 1e-6))
        return p

    def transform_dict(self, probs: dict[str, float]) -> dict[str, float]:
        """Calibrate each 1X2 margin then renormalise."""
        raw = {k: self.transform(float(probs[k])) for k in ("H", "D", "A") if k in probs}
        total = sum(raw.values())
        if total <= 0:
            return {"H": 1 / 3, "D": 1 / 3, "A": 1 / 3}
        return {k: v / total for k, v in raw.items()}


# ---------------------------------------------------------------------------
# SQLite: bet_history
# ---------------------------------------------------------------------------


def ensure_bet_history_table(db_path: Path | str = DEFAULT_DB_PATH) -> Path:
    """Create ``bet_history`` if missing. Returns resolved DB path."""
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as conn:
        conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {BET_HISTORY_TABLE} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                match_date TEXT NOT NULL,
                home_team TEXT NOT NULL,
                away_team TEXT NOT NULL,
                market TEXT NOT NULL,
                selection TEXT NOT NULL,
                p_model REAL,
                odds REAL,
                ev REAL,
                stake_pct REAL,
                stake_amount REAL,
                result TEXT,
                pnl REAL,
                created_at TEXT
            )
            """
        )
        conn.execute(
            f"""
            CREATE INDEX IF NOT EXISTS idx_bet_history_date
            ON {BET_HISTORY_TABLE} (match_date)
            """
        )
        conn.commit()
    return path


def clear_bet_history(db_path: Path | str = DEFAULT_DB_PATH) -> None:
    """Delete all rows from ``bet_history`` (table kept)."""
    path = ensure_bet_history_table(db_path)
    with sqlite3.connect(path) as conn:
        conn.execute(f"DELETE FROM {BET_HISTORY_TABLE}")
        conn.commit()


def save_bets_to_db(
    bets: pd.DataFrame,
    db_path: Path | str = DEFAULT_DB_PATH,
    *,
    replace: bool = False,
) -> int:
    """Append (or replace) backtest bets into ``bet_history``."""
    if bets is None or bets.empty:
        return 0

    path = ensure_bet_history_table(db_path)
    cols = [
        "match_date",
        "home_team",
        "away_team",
        "market",
        "selection",
        "p_model",
        "odds",
        "ev",
        "stake_pct",
        "stake_amount",
        "result",
        "pnl",
    ]
    missing = [c for c in cols if c not in bets.columns]
    if missing:
        raise ValueError(f"bets missing columns: {missing}")

    to_write = bets[cols].copy()
    to_write["match_date"] = pd.to_datetime(
        to_write["match_date"], errors="coerce"
    ).dt.strftime("%Y-%m-%d")
    to_write["created_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    with sqlite3.connect(path) as conn:
        if replace:
            conn.execute(f"DELETE FROM {BET_HISTORY_TABLE}")
        to_write.to_sql(BET_HISTORY_TABLE, conn, if_exists="append", index=False)
        conn.commit()
    return int(len(to_write))


def load_bets_from_db(db_path: Path | str = DEFAULT_DB_PATH) -> pd.DataFrame:
    """Load all rows from ``bet_history`` ordered by date."""
    path = Path(db_path)
    if not path.is_file():
        return pd.DataFrame()
    ensure_bet_history_table(path)
    with sqlite3.connect(path) as conn:
        try:
            df = pd.read_sql(
                f"SELECT * FROM {BET_HISTORY_TABLE} ORDER BY match_date, id",
                conn,
            )
        except (sqlite3.Error, pd.errors.DatabaseError):
            return pd.DataFrame()
    if df.empty:
        return df
    df["match_date"] = pd.to_datetime(df["match_date"], errors="coerce")
    return df


# ---------------------------------------------------------------------------
# Settlement & odds helpers
# ---------------------------------------------------------------------------


def settle_1x2(selection: str, ftr: str) -> ResultLabel:
    """Settle a 1X2 bet against full-time result ``H``/``D``/``A``."""
    want = _SELECTION_TO_FTR.get(str(selection).strip(), "")
    got = str(ftr).strip().upper()
    if want not in {"H", "D", "A"} or got not in {"H", "D", "A"}:
        return "PUSH"
    return "WIN" if want == got else "LOSS"


def _settlement_to_result_pnl(
    settlement: str, odds: float, stake_amount: float
) -> tuple[ResultLabel, float]:
    """Map Dixon–Coles settlement label → (WIN/LOSS/PUSH, cash PnL)."""
    stake = float(stake_amount)
    o = float(odds)
    if settlement == "win":
        return "WIN", stake * (o - 1.0)
    if settlement == "half_win":
        return "WIN", 0.5 * stake * (o - 1.0)
    if settlement == "push":
        return "PUSH", 0.0
    if settlement == "half_lose":
        return "LOSS", -0.5 * stake
    return "LOSS", -stake


def settle_ou(
    selection: str, home_goals: int, away_goals: int, line: float, odds: float, stake: float
) -> tuple[ResultLabel, float]:
    """Settle Over/Under including quarter-line half-win/half-lose."""
    total = int(home_goals) + int(away_goals)
    if str(selection).lower().startswith("over"):
        s = _combine_settlements(_settle_over(total, float(line)))
    else:
        s = _combine_settlements(_settle_under(total, float(line)))
    return _settlement_to_result_pnl(s, odds, stake)


def settle_ah(
    selection: str,
    home_goals: int,
    away_goals: int,
    handicap: float,
    odds: float,
    stake: float,
) -> tuple[ResultLabel, float]:
    """Settle Asian Handicap (home-line convention)."""
    home_s = _combine_settlements(
        _settle_ah_home(int(home_goals), int(away_goals), float(handicap))
    )
    if "away" in str(selection).lower():
        home_s = _mirror_settlement(home_s)
    return _settlement_to_result_pnl(home_s, odds, stake)


def pnl_from_result(
    result: ResultLabel,
    odds: float,
    stake_amount: float,
) -> float:
    """Cash PnL for a settled 1X2-style bet (no half outcomes)."""
    stake = float(stake_amount)
    if result == "WIN":
        return stake * (float(odds) - 1.0)
    if result == "LOSS":
        return -stake
    return 0.0


def _resolve_1x2_odds(
    row: pd.Series,
    odds_family: str = "B365",
) -> dict[str, float] | None:
    """Extract closing 1X2 odds; fall back B365 → Avg → Max → PS."""
    families = [odds_family] + [
        f for f in ("B365", "Avg", "Max", "PS") if f != odds_family
    ]
    for fam in families:
        cols = ODDS_COLUMN_MAPS.get(fam)
        if not cols:
            continue
        if not all(c in row.index for c in cols.values()):
            continue
        odds = {k: float(row[cols[k]]) for k in ("H", "D", "A")}
        if any(pd.isna(v) or v <= 1.0 for v in odds.values()):
            continue
        return odds
    return None


def _closing_ref_1x2(
    row: pd.Series,
    selection: str,
    *,
    ref_family: str = "Avg",
    entered_family: str = "B365",
) -> float | None:
    """Closing reference price for a 1X2 selection (for CLV).

    Prefers ``ref_family`` (Avg = market consensus close on football-data).
    Falls back to other families excluding the entry book when needed.
    """
    key = _SELECTION_TO_FTR.get(str(selection).strip(), str(selection).strip().upper())
    if key not in {"H", "D", "A"}:
        return None
    families = [ref_family] + [
        f for f in ("Avg", "B365", "Max", "PS") if f not in {ref_family, entered_family}
    ]
    # Allow ref == entered only as last resort (CLV → 0).
    families.append(entered_family)
    for fam in families:
        cols = ODDS_COLUMN_MAPS.get(fam)
        if not cols or cols[key] not in row.index:
            continue
        val = float(row[cols[key]])
        if pd.isna(val) or val <= 1.0:
            continue
        return val
    return None


def _season_label(row: pd.Series, day: pd.Timestamp) -> str:
    if "Season" in row.index and pd.notna(row["Season"]):
        return str(row["Season"])
    if "SeasonStart" in row.index and pd.notna(row["SeasonStart"]):
        start = int(row["SeasonStart"])
        return f"{start}/{start + 1}"
    year = int(pd.Timestamp(day).year)
    # EPL season straddles calendar years; Aug+ → year/year+1
    if pd.Timestamp(day).month >= 7:
        return f"{year}/{year + 1}"
    return f"{year - 1}/{year}"


def _resolve_ou_odds(row: pd.Series, line: float = 2.5) -> dict[str, float] | None:
    """OU odds from ``B365_O25``/``U25`` or raw ``B365>2.5`` columns."""
    candidates = [
        ("B365_O25", "B365_U25"),
        ("B365>2.5", "B365<2.5"),
        ("Avg_O25", "Avg_U25"),
        ("Avg>2.5", "Avg<2.5"),
    ]
    for over_c, under_c in candidates:
        if over_c not in row.index or under_c not in row.index:
            continue
        over, under = float(row[over_c]), float(row[under_c])
        if pd.isna(over) or pd.isna(under) or over <= 1.0 or under <= 1.0:
            continue
        return {"line": float(line), "over": over, "under": under}
    return None


def _resolve_ah_odds(row: pd.Series) -> dict[str, float] | None:
    """AH odds from ``AHh`` + ``B365AHH``/``B365AHA`` (or Avg)."""
    if "AHh" not in row.index or pd.isna(row["AHh"]):
        return None
    handicap = float(row["AHh"])
    for home_c, away_c in (("B365AHH", "B365AHA"), ("AvgAHH", "AvgAHA")):
        if home_c not in row.index or away_c not in row.index:
            continue
        h, a = float(row[home_c]), float(row[away_c])
        if pd.isna(h) or pd.isna(a) or h <= 1.0 or a <= 1.0:
            continue
        return {"handicap": handicap, "home": h, "away": a}
    return None


def _blend_1x2(
    p_dc: dict[str, float],
    p_ml: dict[str, float],
    w_ml: float,
) -> dict[str, float]:
    w = float(max(0.0, min(1.0, w_ml)))
    out = {k: (1.0 - w) * float(p_dc[k]) + w * float(p_ml[k]) for k in ("H", "D", "A")}
    total = sum(out.values())
    if total <= 0:
        return {"H": 1 / 3, "D": 1 / 3, "A": 1 / 3}
    return {k: v / total for k, v in out.items()}


def _as_date(ts: Any) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    if t.tzinfo is not None:
        t = t.tz_convert(None)
    return t.normalize()


def odds_in_band(odds: float, min_odds: float, max_odds: float) -> bool:
    """True when book odds sit inside the allowed band."""
    o = float(odds)
    return (not pd.isna(o)) and (min_odds <= o <= max_odds)


def capped_stake_pct(kelly_pct: float, max_stake_pct: float) -> float:
    """Clip Kelly fraction to the safety cap."""
    return float(max(0.0, min(float(kelly_pct), float(max_stake_pct))))


# ---------------------------------------------------------------------------
# Walk-forward engine
# ---------------------------------------------------------------------------


@dataclass
class BacktestConfig:
    """Parameters for :func:`run_backtest`."""

    min_ev: float = DEFAULT_MIN_EV
    kelly_fraction: float = DEFAULT_KELLY_FRACTION
    initial_bankroll: float = 1000.0
    min_train_matches: int = 100
    xi: float = 0.0018
    odds_family: str = "B365"
    use_ml: bool = False  # False = Dixon–Coles only; True = Ensemble
    ensemble_w_ml: float = DEFAULT_W_ML
    # Market filter — only these markets are bet (default all three).
    allowed_markets: tuple[str, ...] | list[str] = ("1X2", "OU", "AH")
    # Per-market EV floors (None → fall back to ``min_ev``).
    min_ev_1x2: float | None = None
    min_ev_ou: float | None = None
    min_ev_ah: float | None = None
    persist: bool = True
    replace_history: bool = True
    db_path: Path | str = DEFAULT_DB_PATH
    refit_every: int = 1
    # Calibration
    calibration: CalibMethod = "isotonic"
    min_cal_samples: int = 80
    # Safety filters
    min_odds: float = DEFAULT_MIN_ODDS
    max_odds: float = DEFAULT_MAX_ODDS
    max_stake_pct: float = DEFAULT_MAX_STAKE_PCT
    # Closing-line discipline: bets priced at ``odds_family`` (default B365
    # closing). CLV compares that price to ``closing_ref_family`` (Avg).
    closing_ref_family: str = "Avg"
    # Exchange-style commission on winning PnL only (0.02 = 2%).
    commission_pct: float = DEFAULT_COMMISSION_PCT
    # Shrink ensemble w_ML when train sample is thin (UWCL-friendly).
    adapt_w_ml: bool = True

    def normalized_markets(self) -> tuple[str, ...]:
        """Canonical market codes from ``allowed_markets``."""
        alias = {
            "1X2": "1X2",
            "OU": "OU",
            "OVER/UNDER": "OU",
            "OVERUNDER": "OU",
            "AH": "AH",
            "ASIAN HANDICAP": "AH",
            "ASIAN": "AH",
        }
        out: list[str] = []
        for m in self.allowed_markets:
            key = alias.get(str(m).strip().upper())
            if key and key not in out:
                out.append(key)
        return tuple(out) if out else ("1X2", "OU", "AH")

    def min_ev_map(self) -> dict[str, float]:
        """Per-market EV thresholds used by the recommender / backtester."""
        return {
            "1X2": float(self.min_ev if self.min_ev_1x2 is None else self.min_ev_1x2),
            "OU": float(self.min_ev if self.min_ev_ou is None else self.min_ev_ou),
            "AH": float(self.min_ev if self.min_ev_ah is None else self.min_ev_ah),
        }


@dataclass
class BacktestResult:
    """Output of a completed walk-forward run."""

    bets: pd.DataFrame
    equity_curve: pd.DataFrame
    summary: dict[str, Any]
    config: BacktestConfig
    n_match_days: int = 0
    n_train_final: int = 0
    predictions: pd.DataFrame = field(default_factory=pd.DataFrame)
    calibration: dict[str, Any] = field(default_factory=dict)


ProgressFn = Callable[[int, int, str], None]


def run_backtest(
    matches: pd.DataFrame,
    *,
    config: BacktestConfig | None = None,
    progress: ProgressFn | None = None,
) -> BacktestResult:
    """Run walk-forward value-bet simulation on historical matches."""
    cfg = config or BacktestConfig()
    required = {"Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"}
    missing = required - set(matches.columns)
    if missing:
        raise ValueError(f"matches missing columns: {sorted(missing)}")

    df = matches.dropna(subset=list(required)).copy()
    df["Date"] = df["Date"].map(_as_date)
    df["HomeTeam"] = df["HomeTeam"].astype(str).str.strip()
    df["AwayTeam"] = df["AwayTeam"].astype(str).str.strip()
    df["FTHG"] = pd.to_numeric(df["FTHG"], errors="coerce").astype(int)
    df["FTAG"] = pd.to_numeric(df["FTAG"], errors="coerce").astype(int)
    if "FTR" not in df.columns:
        df["FTR"] = np.where(
            df["FTHG"] > df["FTAG"],
            "H",
            np.where(df["FTHG"] < df["FTAG"], "A", "D"),
        )
    else:
        df["FTR"] = df["FTR"].astype(str).str.upper().str.strip()

    df = df.sort_values(["Date", "HomeTeam", "AwayTeam"]).reset_index(drop=True)
    match_days = sorted(df["Date"].unique())

    if len(df) < cfg.min_train_matches + 1:
        raise ValueError(
            f"Need > {cfg.min_train_matches} matches for walk-forward; got {len(df)}"
        )

    bankroll = float(cfg.initial_bankroll)
    bet_rows: list[dict[str, Any]] = []
    pred_rows: list[dict[str, Any]] = []
    equity_rows: list[dict[str, Any]] = [
        {"date": match_days[0], "bankroll": bankroll, "daily_pnl": 0.0, "n_bets": 0}
    ]

    dc_model: DixonColesModel | None = None
    ml_model: Any = None
    days_since_refit = cfg.refit_every
    total_days = len(match_days)
    effective_w_ml = float(cfg.ensemble_w_ml)

    # Online calibration memory: past scored 1X2 margins
    cal_p: list[float] = []
    cal_y: list[int] = []
    calibrator = ProbabilityCalibrator(
        method=cfg.calibration, min_samples=cfg.min_cal_samples
    )

    for day_i, day in enumerate(match_days):
        train = df.loc[df["Date"] < day]
        today = df.loc[df["Date"] == day]
        if len(train) < cfg.min_train_matches or today.empty:
            continue

        days_since_refit += 1
        need_refit = dc_model is None or days_since_refit >= max(1, int(cfg.refit_every))

        if need_refit:
            if progress:
                progress(
                    day_i + 1,
                    total_days,
                    f"Fit models @ {day.date()} (train={len(train)})",
                )
            try:
                dc_model = DixonColesModel(xi=float(cfg.xi)).fit(train)
            except Exception:
                continue
            ml_model = None
            effective_w_ml = float(cfg.ensemble_w_ml)
            if cfg.use_ml and EPLMachineLearningModel is not None:
                if cfg.adapt_w_ml:
                    effective_w_ml, _ = ensemble_weight_for_sample(
                        len(train), float(cfg.ensemble_w_ml)
                    )
                if effective_w_ml > 0:
                    try:
                        ml_model = EPLMachineLearningModel().fit(train)
                    except Exception:
                        ml_model = None
            days_since_refit = 0
        elif progress and day_i % 5 == 0:
            progress(day_i + 1, total_days, f"Score @ {day.date()}")

        # Refresh calibrator from all past scored fixtures (causal).
        if cfg.calibration != "none":
            calibrator.fit(cal_p, cal_y)

        assert dc_model is not None
        known = set(dc_model.teams)
        allowed = cfg.normalized_markets()
        ev_by_mkt = cfg.min_ev_map()
        rec = ValueBetRecommender(
            dc_model,
            min_ev=cfg.min_ev,
            kelly_fraction=cfg.kelly_fraction,
            allowed_markets=allowed,
            min_ev_by_market=ev_by_mkt,
        )
        day_bankroll = bankroll
        day_pnl = 0.0
        day_bets = 0

        for _, row in today.iterrows():
            home = str(row["HomeTeam"])
            away = str(row["AwayTeam"])
            if home not in known or away not in known:
                continue

            ftr = str(row["FTR"])
            fthg, ftag = int(row["FTHG"]), int(row["FTAG"])

            odds_1x2 = _resolve_1x2_odds(row, cfg.odds_family) if "1X2" in allowed else None
            ou_odds = _resolve_ou_odds(row) if "OU" in allowed else None
            ah_odds = _resolve_ah_odds(row) if "AH" in allowed else None

            probs_1x2: dict[str, float] | None = None
            if odds_1x2 is not None:
                try:
                    p_dc = dc_model.predict_match_probs(home, away)
                except Exception:
                    continue
                if cfg.use_ml and ml_model is not None and effective_w_ml > 0:
                    try:
                        p_ml = ml_model.predict_proba(home, away, match_date=day)
                        probs_1x2 = _blend_1x2(p_dc, p_ml, effective_w_ml)
                    except Exception:
                        probs_1x2 = p_dc
                else:
                    probs_1x2 = p_dc

                season = _season_label(row, day)
                # Multiclass snapshot (no binary y — used for MC-Brier / MC-LogLoss)
                pred_rows.append(
                    {
                        "match_date": day,
                        "season": season,
                        "market": "1X2",
                        "home_team": home,
                        "away_team": away,
                        "ftr": ftr,
                        "p_H": float(probs_1x2["H"]),
                        "p_D": float(probs_1x2["D"]),
                        "p_A": float(probs_1x2["A"]),
                        "model": "ensemble" if cfg.use_ml and ml_model is not None else "dc",
                        "w_ml": float(effective_w_ml) if cfg.use_ml else 0.0,
                    }
                )
                # Binary margins for reliability / Brier / log-loss
                for key in ("H", "D", "A"):
                    cal_p.append(float(probs_1x2[key]))
                    cal_y.append(1 if ftr == key else 0)
                    pred_rows.append(
                        {
                            "match_date": day,
                            "season": season,
                            "market": "1X2",
                            "selection": key,
                            "home_team": home,
                            "away_team": away,
                            "ftr": ftr,
                            "p_model": float(probs_1x2[key]),
                            "y": 1.0 if ftr == key else 0.0,
                            "model": (
                                "ensemble"
                                if cfg.use_ml and ml_model is not None
                                else "dc"
                            ),
                            "w_ml": float(effective_w_ml) if cfg.use_ml else 0.0,
                        }
                    )

                if calibrator.fitted_:
                    probs_1x2 = calibrator.transform_dict(probs_1x2)

            try:
                candidates = rec.evaluate_match(
                    home,
                    away,
                    odds_1x2=odds_1x2,
                    over_under=ou_odds,
                    asian_handicap=ah_odds,
                    only_value=False,  # apply odds/stake filters below
                    match_date=day,
                    probs_1x2=probs_1x2,
                )
            except Exception:
                continue

            season = _season_label(row, day)
            # Score OU / AH model probs for calibration (all fixtures with odds)
            if ou_odds is not None:
                try:
                    line = float(ou_odds["line"])
                    ou_p = dc_model.predict_over_under(home, away, line=line)
                    total_g = fthg + ftag
                    # .5 lines: over wins iff total > line
                    y_over = 1.0 if total_g > line else 0.0
                    pred_rows.append(
                        {
                            "match_date": day,
                            "season": season,
                            "market": "OU",
                            "selection": f"Over {line:g}",
                            "p_model": float(ou_p["over"]),
                            "y": y_over,
                            "model": "dc",
                            "w_ml": 0.0,
                        }
                    )
                    pred_rows.append(
                        {
                            "match_date": day,
                            "season": season,
                            "market": "OU",
                            "selection": f"Under {line:g}",
                            "p_model": float(ou_p["under"]),
                            "y": 1.0 - y_over,
                            "model": "dc",
                            "w_ml": 0.0,
                        }
                    )
                except Exception:
                    pass
            if ah_odds is not None:
                try:
                    hand = float(ah_odds["handicap"])
                    ah_p = dc_model.predict_asian_handicap(
                        home, away, handicap=hand
                    )
                    # Approximate binary: home covers when settlement is win
                    home_s = _combine_settlements(
                        _settle_ah_home(fthg, ftag, hand)
                    )
                    y_home = (
                        1.0
                        if home_s in {"win", "half_win"}
                        else (0.0 if home_s in {"lose", "half_lose"} else float("nan"))
                    )
                    if y_home == y_home:
                        pred_rows.append(
                            {
                                "match_date": day,
                                "season": season,
                                "market": "AH",
                                "selection": f"AH Home {hand:+g}",
                                "p_model": float(ah_p["home"]),
                                "y": y_home,
                                "model": "dc",
                                "w_ml": 0.0,
                            }
                        )
                        pred_rows.append(
                            {
                                "match_date": day,
                                "season": season,
                                "market": "AH",
                                "selection": f"AH Away {-hand:+g}",
                                "p_model": float(ah_p["away"]),
                                "y": 1.0 - y_home,
                                "model": "dc",
                                "w_ml": 0.0,
                            }
                        )
                except Exception:
                    pass

            for bet in candidates:
                if bet.market not in allowed:
                    continue
                book_odds = float(bet.bookmaker_odds)
                if not odds_in_band(book_odds, cfg.min_odds, cfg.max_odds):
                    continue

                p_use = float(bet.p_model)
                # Re-calibrate OU/AH single-event probs when calibrator is warm.
                if bet.market != "1X2" and calibrator.fitted_:
                    p_use = calibrator.transform(p_use)

                ev = expected_value(p_use, book_odds)
                min_ev_m = float(ev_by_mkt.get(bet.market, cfg.min_ev))
                if ev < min_ev_m:
                    continue

                kelly = fractional_kelly(p_use, book_odds, cfg.kelly_fraction)
                stake_pct = capped_stake_pct(kelly, cfg.max_stake_pct)
                stake_amount = day_bankroll * stake_pct
                if stake_amount <= 0:
                    continue

                if bet.market == "1X2":
                    result = settle_1x2(bet.selection, ftr)
                    pnl = pnl_from_result(result, book_odds, stake_amount)
                elif bet.market == "OU":
                    line = float(bet.line) if bet.line is not None else 2.5
                    result, pnl = settle_ou(
                        bet.selection, fthg, ftag, line, book_odds, stake_amount
                    )
                elif bet.market == "AH":
                    hand = float(bet.line) if bet.line is not None else 0.0
                    result, pnl = settle_ah(
                        bet.selection, fthg, ftag, hand, book_odds, stake_amount
                    )
                else:
                    continue

                pnl = apply_commission(pnl, cfg.commission_pct)

                # CLV vs closing reference (Avg consensus on football-data)
                odds_close = None
                clv = float("nan")
                if bet.market == "1X2":
                    odds_close = _closing_ref_1x2(
                        row,
                        bet.selection,
                        ref_family=cfg.closing_ref_family,
                        entered_family=cfg.odds_family,
                    )
                    if odds_close is not None:
                        clv = closing_line_value(book_odds, odds_close)

                bet_rows.append(
                    {
                        "match_date": day,
                        "home_team": home,
                        "away_team": away,
                        "market": bet.market,
                        "selection": bet.selection,
                        "p_model": p_use,
                        "odds": book_odds,
                        "odds_closing": odds_close,
                        "clv": clv,
                        "ev": ev,
                        "stake_pct": stake_pct,
                        "stake_amount": stake_amount,
                        "result": result,
                        "pnl": pnl,
                        "commission_pct": float(cfg.commission_pct),
                        "w_ml": float(effective_w_ml) if cfg.use_ml else 0.0,
                    }
                )
                day_pnl += pnl
                day_bets += 1

        bankroll = max(0.0, bankroll + day_pnl)
        equity_rows.append(
            {
                "date": day,
                "bankroll": bankroll,
                "daily_pnl": day_pnl,
                "n_bets": day_bets,
            }
        )
        if bankroll <= 1e-9:
            if progress:
                progress(day_i + 1, total_days, "Bankroll depleted — stop")
            break

    bets_df = pd.DataFrame(bet_rows)
    equity_df = pd.DataFrame(equity_rows)
    preds_df = pd.DataFrame(pred_rows)
    if not equity_df.empty:
        equity_df["date"] = pd.to_datetime(equity_df["date"])

    summary = get_backtest_summary(
        bets_df,
        equity_curve=equity_df,
        initial_bankroll=cfg.initial_bankroll,
    )
    calibration = summarize_predictions(preds_df)
    summary["calibration"] = calibration
    summary["calibration_report"] = format_calibration_report(calibration)
    if not bets_df.empty and "clv" in bets_df.columns:
        clv_s = bets_df["clv"].astype(float)
        clv_s = clv_s[np.isfinite(clv_s)]
        summary["avg_clv"] = float(clv_s.mean()) if len(clv_s) else float("nan")
        summary["avg_clv_pct"] = (
            float(clv_s.mean() * 100.0) if len(clv_s) else float("nan")
        )
        summary["n_clv"] = int(len(clv_s))
    else:
        summary["avg_clv"] = float("nan")
        summary["avg_clv_pct"] = float("nan")
        summary["n_clv"] = 0
    summary["commission_pct"] = float(cfg.commission_pct)

    if cfg.persist and not bets_df.empty:
        save_bets_to_db(bets_df, cfg.db_path, replace=cfg.replace_history)

    return BacktestResult(
        bets=bets_df,
        equity_curve=equity_df,
        summary=summary,
        config=cfg,
        n_match_days=int(len(match_days)),
        n_train_final=int(len(df)),
        predictions=preds_df,
        calibration=calibration,
    )


# ---------------------------------------------------------------------------
# Performance report
# ---------------------------------------------------------------------------


def _max_drawdown_pct(equity: pd.Series) -> float:
    """Peak-to-trough drawdown as a fraction of the running peak (≤ 0)."""
    if equity.empty:
        return 0.0
    values = equity.astype(float).to_numpy()
    peak = values[0]
    max_dd = 0.0
    for v in values:
        peak = max(peak, v)
        if peak > 0:
            dd = (v - peak) / peak
            max_dd = min(max_dd, dd)
    return float(max_dd)


def market_breakdown(bets: pd.DataFrame) -> pd.DataFrame:
    """Group PnL / hit-rate / ROI by market (1X2, OU, AH).

    Always returns one row per market (zeros when no bets).
    """
    markets = ["1X2", "OU", "AH"]
    rows: list[dict[str, Any]] = []
    empty = bets is None or bets.empty or "market" not in getattr(bets, "columns", [])

    for mkt in markets:
        if empty:
            sub = pd.DataFrame()
        else:
            sub = bets.loc[bets["market"] == mkt]
        n = int(len(sub))
        if n == 0:
            rows.append(
                {
                    "market": mkt,
                    "n_bets": 0,
                    "wins": 0,
                    "losses": 0,
                    "hit_rate_pct": 0.0,
                    "total_pnl": 0.0,
                    "total_staked": 0.0,
                    "roi_pct": 0.0,
                    "avg_odds": float("nan"),
                    "brier_score": float("nan"),
                }
            )
            continue
        decided = sub.loc[sub["result"].isin(["WIN", "LOSS"])]
        wins = int((decided["result"] == "WIN").sum())
        losses = int((decided["result"] == "LOSS").sum())
        n_dec = wins + losses
        hit = (wins / n_dec) if n_dec else 0.0
        pnl = float(sub["pnl"].sum())
        staked = float(sub["stake_amount"].sum())
        roi = (pnl / staked) if staked > 0 else 0.0
        brier = float("nan")
        if n_dec and "p_model" in decided.columns:
            y = (decided["result"] == "WIN").astype(float).to_numpy()
            p = decided["p_model"].astype(float).to_numpy()
            brier = float(np.mean((p - y) ** 2))
        rows.append(
            {
                "market": mkt,
                "n_bets": n,
                "wins": wins,
                "losses": losses,
                "hit_rate_pct": hit * 100.0,
                "total_pnl": pnl,
                "total_staked": staked,
                "roi_pct": roi * 100.0,
                "avg_odds": float(sub["odds"].mean()),
                "brier_score": brier,
            }
        )
    return pd.DataFrame(rows)


def get_backtest_summary(
    bets: pd.DataFrame | None = None,
    *,
    equity_curve: pd.DataFrame | None = None,
    initial_bankroll: float = 1000.0,
    db_path: Path | str | None = None,
) -> dict[str, Any]:
    """Compute performance metrics from bets and/or equity curve."""
    if bets is None:
        bets = load_bets_from_db(db_path) if db_path is not None else pd.DataFrame()
    if bets is None:
        bets = pd.DataFrame()

    decided = bets
    if not bets.empty and "result" in bets.columns:
        decided = bets.loc[bets["result"].isin(["WIN", "LOSS"])].copy()

    n_bets = int(len(bets)) if not bets.empty else 0
    n_decided = int(len(decided))
    wins = int((decided["result"] == "WIN").sum()) if n_decided else 0
    losses = int((decided["result"] == "LOSS").sum()) if n_decided else 0
    pushes = (
        int((bets["result"] == "PUSH").sum())
        if n_bets and "result" in bets.columns
        else 0
    )
    hit_rate = (wins / n_decided) if n_decided else 0.0

    total_pnl = float(bets["pnl"].sum()) if n_bets and "pnl" in bets.columns else 0.0
    total_staked = (
        float(bets["stake_amount"].sum())
        if n_bets and "stake_amount" in bets.columns
        else 0.0
    )
    roi = (total_pnl / total_staked) if total_staked > 0 else 0.0

    brier = float("nan")
    if n_decided and "p_model" in decided.columns:
        y = (decided["result"] == "WIN").astype(float).to_numpy()
        p = decided["p_model"].astype(float).to_numpy()
        brier = float(np.mean((p - y) ** 2))

    if equity_curve is None or equity_curve.empty:
        if n_bets and "match_date" in bets.columns:
            tmp = bets.copy()
            tmp["match_date"] = pd.to_datetime(tmp["match_date"], errors="coerce")
            daily = (
                tmp.groupby(tmp["match_date"].dt.normalize(), as_index=False)["pnl"]
                .sum()
                .rename(columns={"match_date": "date", "pnl": "daily_pnl"})
            )
            daily = daily.sort_values("date")
            daily["bankroll"] = float(initial_bankroll) + daily["daily_pnl"].cumsum()
            equity_curve = daily
        else:
            equity_curve = pd.DataFrame(
                {"date": [], "bankroll": [], "daily_pnl": [], "n_bets": []}
            )

    final_bankroll = (
        float(equity_curve["bankroll"].iloc[-1])
        if not equity_curve.empty and "bankroll" in equity_curve.columns
        else float(initial_bankroll) + total_pnl
    )
    max_dd = (
        _max_drawdown_pct(equity_curve["bankroll"])
        if not equity_curve.empty and "bankroll" in equity_curve.columns
        else 0.0
    )
    bankroll_return = (
        (final_bankroll / initial_bankroll - 1.0) if initial_bankroll > 0 else 0.0
    )

    by_market = market_breakdown(bets)

    metrics = {
        "total_bets": n_bets,
        "wins": wins,
        "losses": losses,
        "pushes": pushes,
        "hit_rate": hit_rate,
        "hit_rate_pct": hit_rate * 100.0,
        "total_pnl": total_pnl,
        "total_staked": total_staked,
        "roi": roi,
        "roi_pct": roi * 100.0,
        "initial_bankroll": float(initial_bankroll),
        "final_bankroll": final_bankroll,
        "bankroll_return": bankroll_return,
        "bankroll_return_pct": bankroll_return * 100.0,
        "max_drawdown": max_dd,
        "max_drawdown_pct": max_dd * 100.0,
        "brier_score": brier,
        "avg_ev": float(bets["ev"].mean()) if n_bets and "ev" in bets.columns else float("nan"),
        "avg_odds": float(bets["odds"].mean()) if n_bets and "odds" in bets.columns else float("nan"),
        "avg_stake_pct": (
            float(bets["stake_pct"].mean())
            if n_bets and "stake_pct" in bets.columns
            else float("nan")
        ),
    }

    metrics_df = pd.DataFrame(
        [
            {
                "Total Bets": metrics["total_bets"],
                "Win Rate %": round(metrics["hit_rate_pct"], 2),
                "Total PnL": round(metrics["total_pnl"], 2),
                "ROI %": round(metrics["roi_pct"], 2),
                "Final Bankroll": round(metrics["final_bankroll"], 2),
                "Max Drawdown %": round(metrics["max_drawdown_pct"], 2),
                "Brier Score": (
                    round(metrics["brier_score"], 4)
                    if metrics["brier_score"] == metrics["brier_score"]
                    else None
                ),
                "Avg Odds": (
                    round(metrics["avg_odds"], 2)
                    if metrics["avg_odds"] == metrics["avg_odds"]
                    else None
                ),
            }
        ]
    )

    metrics["metrics_df"] = metrics_df
    metrics["equity_curve"] = equity_curve
    metrics["bets"] = bets
    metrics["market_breakdown"] = by_market
    return metrics


def format_backtest_report(summary: dict[str, Any]) -> str:
    """Human-readable console report."""
    brier = summary.get("brier_score")
    brier_s = f"{brier:.4f}" if brier == brier else "n/a"
    clv = summary.get("avg_clv_pct")
    clv_s = f"{clv:+.2f}%" if clv is not None and clv == clv else "n/a"
    lines = [
        "=== Walk-Forward Backtest Summary ===",
        f"Total Bets     : {summary.get('total_bets', 0)}",
        f"Wins / Losses  : {summary.get('wins', 0)} / {summary.get('losses', 0)}"
        f"  (Push {summary.get('pushes', 0)})",
        f"Hit Rate       : {summary.get('hit_rate_pct', 0):.2f}%",
        f"Total PnL      : {summary.get('total_pnl', 0):+.2f}",
        f"ROI (on staked): {summary.get('roi_pct', 0):+.2f}%",
        f"Bankroll       : {summary.get('initial_bankroll', 0):.0f}"
        f" -> {summary.get('final_bankroll', 0):.2f}"
        f" ({summary.get('bankroll_return_pct', 0):+.2f}%)",
        f"Max Drawdown   : {summary.get('max_drawdown_pct', 0):.2f}%",
        f"Brier Score    : {brier_s}",
        f"Avg CLV        : {clv_s}  (n={summary.get('n_clv', 0)})",
        f"Commission     : {float(summary.get('commission_pct', 0) or 0):.1%} on wins",
        f"Avg EV / Odds  : {summary.get('avg_ev', float('nan')):+.3f}"
        f" / {summary.get('avg_odds', float('nan')):.2f}",
        f"Avg stake %    : {summary.get('avg_stake_pct', float('nan')):.2%}",
    ]
    mb = summary.get("market_breakdown")
    if isinstance(mb, pd.DataFrame) and not mb.empty:
        lines.append("--- Market breakdown ---")
        for _, r in mb.iterrows():
            lines.append(
                f"  {r['market']:<4} bets={int(r['n_bets']):3d}  "
                f"hit={r['hit_rate_pct']:5.1f}%  "
                f"ROI={r['roi_pct']:+6.1f}%  "
                f"PnL={r['total_pnl']:+8.2f}"
            )
    cal_txt = summary.get("calibration_report")
    if cal_txt:
        lines.append("")
        lines.append(str(cal_txt))
    return "\n".join(lines)


def run_ablation(
    matches: pd.DataFrame,
    *,
    base_config: BacktestConfig | None = None,
    w_ml_grid: Sequence[float] = (0.0, 0.2, 0.4, 0.6),
    progress: ProgressFn | None = None,
) -> pd.DataFrame:
    """Compare Dixon–Coles-only vs Ensemble across ``w_ml`` values.

    Returns a comparison table (one row per variant) with ROI, Brier, CLV,
    drawdown, and calibration ECE — useful to see when LightGBM helps or
    hurts (esp. thin samples / UWCL).
    """
    base = base_config or BacktestConfig()
    rows: list[dict[str, Any]] = []
    grid = list(w_ml_grid)
    if 0.0 not in grid:
        grid = [0.0, *grid]

    for i, w in enumerate(grid):
        use_ml = float(w) > 0.0
        label = "DC-only" if not use_ml else f"Ensemble w_ML={float(w):.0%}"
        if progress:
            progress(i + 1, len(grid), f"Ablation: {label}")

        cfg = BacktestConfig(
            min_ev=base.min_ev,
            kelly_fraction=base.kelly_fraction,
            initial_bankroll=base.initial_bankroll,
            min_train_matches=base.min_train_matches,
            xi=base.xi,
            odds_family=base.odds_family,
            use_ml=use_ml,
            ensemble_w_ml=float(w),
            allowed_markets=base.allowed_markets,
            min_ev_1x2=base.min_ev_1x2,
            min_ev_ou=base.min_ev_ou,
            min_ev_ah=base.min_ev_ah,
            persist=False,
            replace_history=False,
            db_path=base.db_path,
            refit_every=base.refit_every,
            calibration=base.calibration,
            min_cal_samples=base.min_cal_samples,
            min_odds=base.min_odds,
            max_odds=base.max_odds,
            max_stake_pct=base.max_stake_pct,
            closing_ref_family=base.closing_ref_family,
            commission_pct=base.commission_pct,
            adapt_w_ml=base.adapt_w_ml,
        )
        try:
            result = run_backtest(matches, config=cfg, progress=None)
        except ValueError as exc:
            rows.append(
                {
                    "variant": label,
                    "use_ml": use_ml,
                    "w_ml": float(w),
                    "error": str(exc),
                }
            )
            continue

        s = result.summary
        cal = (result.calibration or {}).get("overall") or {}
        rows.append(
            {
                "variant": label,
                "use_ml": use_ml,
                "w_ml": float(w),
                "n_bets": s.get("total_bets", 0),
                "roi_pct": s.get("roi_pct", float("nan")),
                "hit_rate_pct": s.get("hit_rate_pct", float("nan")),
                "max_drawdown_pct": s.get("max_drawdown_pct", float("nan")),
                "avg_clv_pct": s.get("avg_clv_pct", float("nan")),
                "brier_bets": s.get("brier_score", float("nan")),
                "brier_preds": cal.get("brier", float("nan")),
                "log_loss": cal.get("log_loss", float("nan")),
                "ece": cal.get("ece", float("nan")),
                "mc_brier": cal.get("multiclass_brier", float("nan")),
                "final_bankroll": s.get("final_bankroll", float("nan")),
            }
        )
    return pd.DataFrame(rows)
