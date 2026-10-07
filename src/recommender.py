r"""Value Betting engine.

Fair Odds = 1 / P_model
EV        = (P_model × Odds_bookmaker) - 1
Recommend only when EV ≥ min_ev (default 0.05 = 5%).

Fractional Kelly (default 10% of full Kelly, capped at 1% bankroll):
    f* = min( kelly_fraction × EV / (Odds − 1), MAX_STAKE_PCT )
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Literal, Mapping, Sequence

import pandas as pd

from src.config import (
    DEFAULT_KELLY_FRACTION,
    DEFAULT_MIN_EV,
    MAX_STAKE_PCT,
)
from src.dixon_coles import DixonColesModel
from src.models import NEW_TEAM_KELLY_CAP, apply_new_team_kelly_cap
from src.strategy import capped_kelly_fraction

MarketKind = Literal["1X2", "OU", "AH"]

MARKET_1X2_LABELS: dict[str, str] = {"H": "Home", "D": "Draw", "A": "Away"}

ODDS_COLUMN_MAPS: dict[str, dict[str, str]] = {
    "B365": {"H": "B365H", "D": "B365D", "A": "B365A"},
    "Avg": {"H": "AvgH", "D": "AvgD", "A": "AvgA"},
    "Max": {"H": "MaxH", "D": "MaxD", "A": "MaxA"},
    "PS": {"H": "PSH", "D": "PSD", "A": "PSA"},
}


# ---------------------------------------------------------------------------
# Core math
# ---------------------------------------------------------------------------


def fair_odds(p_model: float) -> float:
    """Fair Odds = 1 / P_model."""
    if p_model <= 0.0:
        raise ValueError(f"p_model must be positive, got {p_model}")
    return 1.0 / p_model


def expected_value(p_model: float, odds_bookmaker: float) -> float:
    """EV = (P_model × Odds_bookmaker) - 1."""
    if p_model < 0.0 or p_model > 1.0:
        raise ValueError(f"p_model must be in [0, 1], got {p_model}")
    if odds_bookmaker <= 0.0:
        raise ValueError(f"odds_bookmaker must be positive, got {odds_bookmaker}")
    return (p_model * odds_bookmaker) - 1.0


def fractional_kelly(
    p_model: float,
    odds_bookmaker: float,
    fraction: float = DEFAULT_KELLY_FRACTION,
) -> float:
    """Fractional Kelly stake as a fraction of bankroll.

    f* = fraction × (P_model × Odds - 1) / (Odds - 1)

    Returns 0 when EV ≤ 0 or odds ≤ 1. Clipped to [0, 1].
    Prefer :func:`src.strategy.capped_kelly_fraction` for production stakes
    (applies ``MAX_STAKE_PCT``).
    """
    if odds_bookmaker <= 1.0 or p_model <= 0.0:
        return 0.0
    ev = expected_value(p_model, odds_bookmaker)
    if ev <= 0.0:
        return 0.0
    stake = fraction * ev / (odds_bookmaker - 1.0)
    return float(max(0.0, min(1.0, stake)))


def ev_pct(p_model: float, odds_bookmaker: float) -> float:
    """EV as a percentage (0.05 → 5.0)."""
    return expected_value(p_model, odds_bookmaker) * 100.0


# ---------------------------------------------------------------------------
# Recommendation record
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ValueBet:
    """One evaluated (or recommended) selection across 1X2 / OU / AH."""

    home_team: str
    away_team: str
    market: MarketKind
    selection: str
    p_model: float
    fair_odds: float
    bookmaker_odds: float
    ev: float
    ev_pct: float
    kelly_fraction: float
    kelly_pct: float
    recommended: bool
    line: float | None = None
    lambda_home: float | None = None
    lambda_away: float | None = None
    odds_source: str | None = None
    match_date: object | None = None
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Recommender
# ---------------------------------------------------------------------------


@dataclass
class ValueBetRecommender:
    """Scan 1X2, Over/Under, and Asian Handicap for value bets.

    Parameters
    ----------
    model:
        Fitted :class:`~src.dixon_coles.DixonColesModel`.
    min_ev:
        Default minimum EV to recommend (default 0.05 = 5%).
    kelly_fraction:
        Fraction of full Kelly (default 0.10). Stake is also clipped to
        ``MAX_STAKE_PCT`` (1% bankroll).
    allowed_markets:
        Markets to evaluate (``1X2``, ``OU``, ``AH``). ``None`` / empty → all.
    min_ev_by_market:
        Optional per-market EV floors, e.g. ``{\"1X2\": 0.05, \"OU\": 0.10}``.
        Falls back to ``min_ev`` when a market key is absent.
    max_stake_pct:
        Hard stake ceiling as a fraction of bankroll (default 1%).
    """

    model: DixonColesModel
    min_ev: float = DEFAULT_MIN_EV
    kelly_fraction: float = DEFAULT_KELLY_FRACTION
    allowed_markets: Sequence[str] | None = None
    min_ev_by_market: Mapping[str, float] | None = None
    max_stake_pct: float = MAX_STAKE_PCT

    def _market_allowed(self, market: str) -> bool:
        if not self.allowed_markets:
            return True
        allowed = {str(m).upper() for m in self.allowed_markets}
        # Accept common aliases from the UI.
        aliases = {"OVER/UNDER": "OU", "ASIAN HANDICAP": "AH", "ASIAN": "AH"}
        allowed = {aliases.get(m, m) for m in allowed}
        return market.upper() in allowed

    def min_ev_for(self, market: str) -> float:
        """EV threshold for ``market`` (per-market override or global default)."""
        if self.min_ev_by_market:
            key = str(market).upper()
            if key in self.min_ev_by_market:
                return float(self.min_ev_by_market[key])
            # soft alias
            if key == "OU" and "Over/Under" in self.min_ev_by_market:
                return float(self.min_ev_by_market["Over/Under"])
        return float(self.min_ev)

    # -- builders -----------------------------------------------------------

    def _make_bet(
        self,
        *,
        home_team: str,
        away_team: str,
        market: MarketKind,
        selection: str,
        p_model: float,
        bookmaker_odds: float,
        line: float | None = None,
        lambda_home: float | None = None,
        lambda_away: float | None = None,
        odds_source: str | None = None,
        match_date: object | None = None,
        detail: dict[str, Any] | None = None,
        has_new_team: bool = False,
    ) -> ValueBet | None:
        if p_model <= 0.0 or bookmaker_odds <= 1.0:
            return None
        if pd.isna(p_model) or pd.isna(bookmaker_odds):
            return None
        ev = expected_value(p_model, bookmaker_odds)
        kelly = capped_kelly_fraction(
            p_model,
            bookmaker_odds,
            kelly_fraction=self.kelly_fraction,
            max_stake_pct=self.max_stake_pct,
        )
        kelly = apply_new_team_kelly_cap(
            kelly, has_new_team=has_new_team, cap=NEW_TEAM_KELLY_CAP
        )
        detail_out = dict(detail or {})
        if has_new_team:
            detail_out.setdefault("new_team_fallback", True)
            detail_out.setdefault("kelly_cap", NEW_TEAM_KELLY_CAP)
        return ValueBet(
            home_team=home_team,
            away_team=away_team,
            market=market,
            selection=selection,
            p_model=float(p_model),
            fair_odds=fair_odds(p_model),
            bookmaker_odds=float(bookmaker_odds),
            ev=ev,
            ev_pct=ev * 100.0,
            kelly_fraction=kelly,
            kelly_pct=kelly * 100.0,
            recommended=ev >= self.min_ev_for(market),
            line=line,
            lambda_home=lambda_home,
            lambda_away=lambda_away,
            odds_source=odds_source,
            match_date=match_date,
            detail=detail_out,
        )

    # -- market evaluators --------------------------------------------------

    def _eval_1x2(
        self,
        home_team: str,
        away_team: str,
        odds_1x2: Mapping[str, float],
        *,
        lam: float,
        mu: float,
        odds_source: str | None,
        match_date: object | None,
        probs_1x2: Mapping[str, float] | None = None,
        has_new_team: bool = False,
    ) -> list[ValueBet]:
        # Optional override (e.g. Ensemble blend of Dixon–Coles + LightGBM).
        probs = (
            {k: float(probs_1x2[k]) for k in ("H", "D", "A") if k in probs_1x2}
            if probs_1x2 is not None
            else self.model.predict_match_probs(home_team, away_team)
        )
        out: list[ValueBet] = []
        for key, label in MARKET_1X2_LABELS.items():
            if key not in odds_1x2 or key not in probs:
                continue
            bet = self._make_bet(
                home_team=home_team,
                away_team=away_team,
                market="1X2",
                selection=label,
                p_model=probs[key],
                bookmaker_odds=float(odds_1x2[key]),
                lambda_home=lam,
                lambda_away=mu,
                odds_source=odds_source,
                match_date=match_date,
                detail={"1x2_probs": probs},
                has_new_team=has_new_team,
            )
            if bet is not None:
                out.append(bet)
        return out

    def _eval_ou(
        self,
        home_team: str,
        away_team: str,
        ou: Mapping[str, float],
        *,
        lam: float,
        mu: float,
        odds_source: str | None,
        match_date: object | None,
        has_new_team: bool = False,
    ) -> list[ValueBet]:
        line = float(ou["line"])
        preds = self.model.predict_over_under(home_team, away_team, line=line)
        out: list[ValueBet] = []
        for side, label in (("over", f"Over {line:g}"), ("under", f"Under {line:g}")):
            if side not in ou:
                continue
            bet = self._make_bet(
                home_team=home_team,
                away_team=away_team,
                market="OU",
                selection=label,
                p_model=preds[side],
                bookmaker_odds=float(ou[side]),
                line=line,
                lambda_home=lam,
                lambda_away=mu,
                odds_source=odds_source,
                match_date=match_date,
                detail={k: preds[k] for k in preds if k.startswith(side)},
                has_new_team=has_new_team,
            )
            if bet is not None:
                out.append(bet)
        return out

    def _eval_ah(
        self,
        home_team: str,
        away_team: str,
        ah: Mapping[str, float],
        *,
        lam: float,
        mu: float,
        odds_source: str | None,
        match_date: object | None,
        has_new_team: bool = False,
    ) -> list[ValueBet]:
        handicap = float(ah["handicap"])
        preds = self.model.predict_asian_handicap(
            home_team, away_team, handicap=handicap
        )
        out: list[ValueBet] = []
        sides = (
            ("home", f"AH Home {handicap:+g}", preds["home"]),
            ("away", f"AH Away {-handicap:+g}", preds["away"]),
        )
        for key, label, p in sides:
            if key not in ah:
                continue
            bet = self._make_bet(
                home_team=home_team,
                away_team=away_team,
                market="AH",
                selection=label,
                p_model=p,
                bookmaker_odds=float(ah[key]),
                line=handicap,
                lambda_home=lam,
                lambda_away=mu,
                odds_source=odds_source,
                match_date=match_date,
                detail={k: preds[k] for k in preds if k.startswith(key)},
                has_new_team=has_new_team,
            )
            if bet is not None:
                out.append(bet)
        return out

    # -- public API ---------------------------------------------------------

    def evaluate_match(
        self,
        home_team: str,
        away_team: str,
        *,
        odds_1x2: Mapping[str, float] | None = None,
        over_under: Mapping[str, float] | Sequence[Mapping[str, float]] | None = None,
        asian_handicap: Mapping[str, float]
        | Sequence[Mapping[str, float]]
        | None = None,
        only_value: bool = True,
        odds_source: str | None = None,
        match_date: object | None = None,
        probs_1x2: Mapping[str, float] | None = None,
    ) -> list[ValueBet]:
        """Evaluate 1X2, Over/Under, and Asian Handicap vs bookmaker odds.

        Parameters
        ----------
        odds_1x2:
            ``{\"H\", \"D\", \"A\"}`` decimal odds.
        over_under:
            One dict or a list of dicts:
            ``{\"line\": 2.5, \"over\": 1.90, \"under\": 1.95}``.
        asian_handicap:
            One dict or a list of dicts (home-team handicap convention):
            ``{\"handicap\": -0.25, \"home\": 1.95, \"away\": 1.95}``.
        only_value:
            If True, return only bets with EV ≥ ``min_ev``.
        probs_1x2:
            Optional ``{\"H\", \"D\", \"A\"}`` model probabilities for 1X2
            (e.g. Ensemble blend). When omitted, uses Dixon–Coles.

        Returns
        -------
        list[ValueBet]
            Sorted by EV descending. Each bet includes Quarter-Kelly stake
            (``kelly_pct`` = % of bankroll). New-team fixtures are capped at
            ``NEW_TEAM_KELLY_CAP`` (1% bankroll).
        """
        # Detect BEFORE expected_goals injects priors into attack/defence.
        has_new = bool(self.model.has_new_team(home_team, away_team))
        lam, mu = self.model.expected_goals(home_team, away_team)
        bets: list[ValueBet] = []

        if odds_1x2 and self._market_allowed("1X2"):
            bets.extend(
                self._eval_1x2(
                    home_team,
                    away_team,
                    odds_1x2,
                    lam=lam,
                    mu=mu,
                    odds_source=odds_source,
                    match_date=match_date,
                    probs_1x2=probs_1x2,
                    has_new_team=has_new,
                )
            )

        if over_under and self._market_allowed("OU"):
            ou_list: Sequence[Mapping[str, float]]
            if isinstance(over_under, Mapping) and "line" in over_under:
                ou_list = [over_under]  # type: ignore[list-item]
            else:
                ou_list = over_under  # type: ignore[assignment]
            for ou in ou_list:
                bets.extend(
                    self._eval_ou(
                        home_team,
                        away_team,
                        ou,
                        lam=lam,
                        mu=mu,
                        odds_source=odds_source,
                        match_date=match_date,
                        has_new_team=has_new,
                    )
                )

        if asian_handicap and self._market_allowed("AH"):
            ah_list: Sequence[Mapping[str, float]]
            if isinstance(asian_handicap, Mapping) and "handicap" in asian_handicap:
                ah_list = [asian_handicap]  # type: ignore[list-item]
            else:
                ah_list = asian_handicap  # type: ignore[assignment]
            for ah in ah_list:
                bets.extend(
                    self._eval_ah(
                        home_team,
                        away_team,
                        ah,
                        lam=lam,
                        mu=mu,
                        odds_source=odds_source,
                        match_date=match_date,
                        has_new_team=has_new,
                    )
                )

        if only_value:
            bets = [b for b in bets if b.recommended]
        bets.sort(key=lambda b: b.ev, reverse=True)
        return bets

    def evaluate_match_dataframe(
        self,
        home_team: str,
        away_team: str,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Same as :meth:`evaluate_match` but returns a DataFrame."""
        bets = self.evaluate_match(home_team, away_team, **kwargs)
        if not bets:
            return pd.DataFrame(
                columns=[
                    "home_team",
                    "away_team",
                    "market",
                    "selection",
                    "line",
                    "p_model",
                    "fair_odds",
                    "bookmaker_odds",
                    "ev",
                    "ev_pct",
                    "kelly_fraction",
                    "kelly_pct",
                    "recommended",
                ]
            )
        return pd.DataFrame([b.to_dict() for b in bets])


# ---------------------------------------------------------------------------
# Backward-compatible helpers (Streamlit / scanners)
# ---------------------------------------------------------------------------


def recommend_match(
    model: DixonColesModel,
    home_team: str,
    away_team: str,
    odds: Mapping[str, float],
    *,
    min_ev: float = DEFAULT_MIN_EV,
    kelly_fraction: float = DEFAULT_KELLY_FRACTION,
    odds_source: str | None = None,
    match_date: object | None = None,
    only_value: bool = True,
    over_under: Mapping[str, float] | Sequence[Mapping[str, float]] | None = None,
    asian_handicap: Mapping[str, float] | Sequence[Mapping[str, float]] | None = None,
) -> list[ValueBet]:
    """Convenience wrapper around :class:`ValueBetRecommender` for 1X2 (+ optional sides)."""
    rec = ValueBetRecommender(model, min_ev=min_ev, kelly_fraction=kelly_fraction)
    return rec.evaluate_match(
        home_team,
        away_team,
        odds_1x2=odds,
        over_under=over_under,
        asian_handicap=asian_handicap,
        only_value=only_value,
        odds_source=odds_source,
        match_date=match_date,
    )


def _resolve_odds_columns(odds_family: str) -> dict[str, str]:
    if odds_family not in ODDS_COLUMN_MAPS:
        known = ", ".join(sorted(ODDS_COLUMN_MAPS))
        raise ValueError(f"Unknown odds_family={odds_family!r}. Choose from: {known}")
    return ODDS_COLUMN_MAPS[odds_family]


def recommend_fixtures(
    model: DixonColesModel,
    fixtures: pd.DataFrame,
    *,
    odds_family: str = "B365",
    min_ev: float = DEFAULT_MIN_EV,
    kelly_fraction: float = DEFAULT_KELLY_FRACTION,
    only_value: bool = True,
) -> pd.DataFrame:
    """Scan a fixtures table for 1X2 value bets (football-data odds columns)."""
    required = {"HomeTeam", "AwayTeam"}
    missing = required - set(fixtures.columns)
    if missing:
        raise ValueError(f"fixtures missing columns: {sorted(missing)}")

    col_map = _resolve_odds_columns(odds_family)
    for col in col_map.values():
        if col not in fixtures.columns:
            raise ValueError(
                f"fixtures missing odds column {col!r} for family {odds_family!r}"
            )

    rec = ValueBetRecommender(model, min_ev=min_ev, kelly_fraction=kelly_fraction)
    rows: list[dict[str, Any]] = []

    for _, row in fixtures.iterrows():
        home = str(row["HomeTeam"])
        away = str(row["AwayTeam"])
        # Newcomers use Dixon–Coles league priors (has_new_team / Kelly cap).
        odds = {k: row[col_map[k]] for k in ("H", "D", "A")}
        if any(pd.isna(v) for v in odds.values()):
            continue
        match_date = row["Date"] if "Date" in fixtures.columns else None
        bets = rec.evaluate_match(
            home,
            away,
            odds_1x2={k: float(v) for k, v in odds.items()},
            only_value=False,
            odds_source=odds_family,
            match_date=match_date,
        )
        for bet in bets:
            if only_value and not bet.recommended:
                continue
            rows.append(bet.to_dict())

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("ev", ascending=False).reset_index(drop=True)


def _fixture_side_markets(
    row: pd.Series,
    odds_family: str,
) -> tuple[
    Mapping[str, float] | None,
    Mapping[str, float] | None,
]:
    """Extract O/U and Asian Handicap dicts from a fixtures row.

    Prefers classic ``{family}_O25/U25`` (line 2.5). Falls back to Flashscore
    ``OU_Line`` / ``OddsOver`` / ``OddsUnder`` when O25 aliases are missing
    (common for Japan cups where the main total is 2.75 / 3.5 / …).
    """
    prefix = odds_family  # B365 / Avg / Max
    ou = None
    over_col, under_col = f"{prefix}_O25", f"{prefix}_U25"
    if over_col in row.index and under_col in row.index:
        if pd.notna(row[over_col]) and pd.notna(row[under_col]):
            ou = {
                "line": 2.5,
                "over": float(row[over_col]),
                "under": float(row[under_col]),
            }
    if ou is None:
        # Flashscore / ESPN style dynamic totals.
        line_ok = "OU_Line" in row.index and pd.notna(row["OU_Line"])
        over_ok = "OddsOver" in row.index and pd.notna(row["OddsOver"])
        under_ok = "OddsUnder" in row.index and pd.notna(row["OddsUnder"])
        if line_ok and over_ok and under_ok:
            try:
                ou = {
                    "line": float(row["OU_Line"]),
                    "over": float(row["OddsOver"]),
                    "under": float(row["OddsUnder"]),
                }
            except (TypeError, ValueError):
                ou = None

    ah = None
    home_col, away_col = f"{prefix}AHH", f"{prefix}AHA"
    if odds_family == "B365":
        home_col, away_col = "B365AHH", "B365AHA"
    elif odds_family == "Avg":
        home_col, away_col = "AvgAHH", "AvgAHA"
    elif odds_family == "Max":
        home_col, away_col = "MaxAHH", "MaxAHA"

    if (
        "AHh" in row.index
        and home_col in row.index
        and away_col in row.index
        and pd.notna(row["AHh"])
        and pd.notna(row[home_col])
        and pd.notna(row[away_col])
    ):
        ah = {
            "handicap": float(row["AHh"]),
            "home": float(row[home_col]),
            "away": float(row[away_col]),
        }
    return ou, ah


def recommend_upcoming(
    model: DixonColesModel,
    fixtures: pd.DataFrame,
    *,
    odds_family: str = "B365",
    min_ev: float = DEFAULT_MIN_EV,
    kelly_fraction: float = DEFAULT_KELLY_FRACTION,
    only_value: bool = True,
    include_ou: bool = True,
    include_ah: bool = True,
) -> pd.DataFrame:
    """Scan upcoming fixtures for 1X2 + O/U 2.5 + Asian Handicap value bets.

    Expects columns produced by :func:`src.data_loader.load_upcoming_fixtures`
    (aliased ``B365_O25`` / ``B365_U25``, ``AHh``, ``B365AHH``, …).
    """
    required = {"HomeTeam", "AwayTeam"}
    missing = required - set(fixtures.columns)
    if missing:
        raise ValueError(f"fixtures missing columns: {sorted(missing)}")

    col_map = _resolve_odds_columns(odds_family)
    rec = ValueBetRecommender(model, min_ev=min_ev, kelly_fraction=kelly_fraction)
    rows: list[dict[str, Any]] = []

    for _, row in fixtures.iterrows():
        home = str(row["HomeTeam"])
        away = str(row["AwayTeam"])
        # Newcomers use Dixon–Coles league priors (has_new_team / Kelly cap).
        if any(c not in row.index or pd.isna(row[c]) for c in col_map.values()):
            continue

        odds_1x2 = {k: float(row[col_map[k]]) for k in ("H", "D", "A")}
        ou, ah = _fixture_side_markets(row, odds_family)
        match_date = row["Kickoff"] if "Kickoff" in row.index else row.get("Date")

        bets = rec.evaluate_match(
            home,
            away,
            odds_1x2=odds_1x2,
            over_under=ou if include_ou else None,
            asian_handicap=ah if include_ah else None,
            only_value=False,
            odds_source=odds_family,
            match_date=match_date,
        )
        for bet in bets:
            if only_value and not bet.recommended:
                continue
            payload = bet.to_dict()
            payload["kickoff"] = match_date
            rows.append(payload)

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("ev", ascending=False).reset_index(drop=True)


def format_recommendations(
    bets: Iterable[ValueBet] | pd.DataFrame,
    *,
    top_n: int | None = None,
) -> str:
    """Tóm tắt nhiều dòng cho CLI / Streamlit (tiếng Việt)."""
    if isinstance(bets, pd.DataFrame):
        records = bets.to_dict(orient="records")
    else:
        records = [b.to_dict() if isinstance(b, ValueBet) else dict(b) for b in bets]

    if top_n is not None:
        records = records[:top_n]
    if not records:
        return "Không có value bet (EV ≥ ngưỡng)."

    market_vi = {"1X2": "1X2", "OU": "Tài/Xỉu", "AH": "Chấp Á"}

    def _sel_vi(text: str) -> str:
        if text == "Home":
            return "Chủ nhà"
        if text == "Draw":
            return "Hòa"
        if text == "Away":
            return "Khách"
        if text.startswith("Over "):
            return "Tài " + text[5:]
        if text.startswith("Under "):
            return "Xỉu " + text[6:]
        if text.startswith("AH Home "):
            return "Chấp chủ " + text[8:]
        if text.startswith("AH Away "):
            return "Chấp khách " + text[8:]
        return text

    lines: list[str] = []
    for r in records:
        fixture = f"{r['home_team']} vs {r['away_team']}"
        kelly = r.get("kelly_pct")
        kelly_s = f" · Kelly={kelly:.2f}%" if kelly is not None else ""
        mkt = market_vi.get(str(r.get("market", "")), r.get("market", ""))
        lines.append(
            f"{fixture} | [{mkt}] {_sel_vi(str(r['selection']))} "
            f"@ {r['bookmaker_odds']:.2f} (công bằng {r['fair_odds']:.2f}) | "
            f"P={r['p_model']:.1%} · EV={r['ev_pct']:+.1f}%{kelly_s}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Upcoming-match prediction (live bookmaker odds → Match Card)
# ---------------------------------------------------------------------------


def implied_probability(odds: float) -> float:
    """P_implied = 1 / Odds (xác suất ngầm định từ nhà cái, chưa bỏ margin)."""
    if odds <= 0.0:
        raise ValueError(f"odds must be positive, got {odds}")
    return 1.0 / odds


def _parse_bookmaker_odds(bookmaker_odds: Mapping[str, float]) -> tuple[
    dict[str, float] | None,
    dict[str, float] | None,
    dict[str, float] | None,
]:
    """Normalize the user-facing odds dict into evaluate_match payloads."""
    odds_1x2 = None
    if all(k in bookmaker_odds for k in ("odds_home", "odds_draw", "odds_away")):
        odds_1x2 = {
            "H": float(bookmaker_odds["odds_home"]),
            "D": float(bookmaker_odds["odds_draw"]),
            "A": float(bookmaker_odds["odds_away"]),
        }

    over_under = None
    if all(k in bookmaker_odds for k in ("line", "odds_over", "odds_under")):
        over_under = {
            "line": float(bookmaker_odds["line"]),
            "over": float(bookmaker_odds["odds_over"]),
            "under": float(bookmaker_odds["odds_under"]),
        }

    asian = None
    if all(
        k in bookmaker_odds
        for k in ("handicap", "odds_home_handicap", "odds_away_handicap")
    ):
        asian = {
            "handicap": float(bookmaker_odds["handicap"]),
            "home": float(bookmaker_odds["odds_home_handicap"]),
            "away": float(bookmaker_odds["odds_away_handicap"]),
        }

    return odds_1x2, over_under, asian


def _selection_label_vi(text: str) -> str:
    if text == "Home":
        return "Chủ nhà"
    if text == "Draw":
        return "Hòa"
    if text == "Away":
        return "Khách"
    if text.startswith("Over "):
        return "Tài " + text[5:]
    if text.startswith("Under "):
        return "Xỉu " + text[6:]
    if text.startswith("AH Home "):
        return "Chấp chủ " + text[8:]
    if text.startswith("AH Away "):
        return "Chấp khách " + text[8:]
    return text


@dataclass
class MatchPrediction:
    """Structured result of :func:`predict_upcoming_match`."""

    home_team: str
    away_team: str
    lambda_home: float
    lambda_away: float
    all_bets: list[ValueBet]
    value_bets: list[ValueBet]
    comparison_rows: list[dict[str, Any]]
    model_summary: dict[str, Any]
    min_ev: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "home_team": self.home_team,
            "away_team": self.away_team,
            "lambda_home": self.lambda_home,
            "lambda_away": self.lambda_away,
            "all_bets": [b.to_dict() for b in self.all_bets],
            "value_bets": [b.to_dict() for b in self.value_bets],
            "comparison_rows": self.comparison_rows,
            "model_summary": self.model_summary,
            "min_ev": self.min_ev,
        }


def predict_upcoming_match(
    home_team: str,
    away_team: str,
    bookmaker_odds: Mapping[str, float],
    *,
    model: DixonColesModel | None = None,
    n_seasons: int = 3,
    xi: float = 0.0018,
    min_ev: float = DEFAULT_MIN_EV,
    kelly_fraction: float = DEFAULT_KELLY_FRACTION,
) -> MatchPrediction:
    """Predict an upcoming fixture vs live bookmaker odds.

    Parameters
    ----------
    home_team, away_team:
        Team names as in football-data.co.uk (e.g. ``\"Arsenal\"``, ``\"Chelsea\"``).
    bookmaker_odds:
        Dict with keys:
        - 1X2: ``odds_home``, ``odds_draw``, ``odds_away``
        - O/U: ``line``, ``odds_over``, ``odds_under``
        - AH: ``handicap``, ``odds_home_handicap``, ``odds_away_handicap``
    model:
        Optional pre-fitted Dixon–Coles model. If ``None``, loads ``n_seasons``
        of EPL data and fits a fresh model.
    min_ev:
        Value-bet threshold (default 0.05 = 5%).
    kelly_fraction:
        Fractional Kelly (default 0.10, capped at 1% bankroll).

    Returns
    -------
    MatchPrediction
        xG, full comparison table, and filtered value bets.
    """
    if model is None:
        from src.data_loader import load_epl_data

        data = load_epl_data(n_seasons=n_seasons)
        model = DixonColesModel(xi=xi).fit(data)

    odds_1x2, over_under, asian = _parse_bookmaker_odds(bookmaker_odds)
    if odds_1x2 is None and over_under is None and asian is None:
        raise ValueError(
            "bookmaker_odds must include 1X2 and/or O/U and/or Asian Handicap keys"
        )

    rec = ValueBetRecommender(model, min_ev=min_ev, kelly_fraction=kelly_fraction)
    lam, mu = model.expected_goals(home_team, away_team)
    all_bets = rec.evaluate_match(
        home_team,
        away_team,
        odds_1x2=odds_1x2,
        over_under=over_under,
        asian_handicap=asian,
        only_value=False,
    )
    value_bets = [b for b in all_bets if b.recommended]

    comparison_rows: list[dict[str, Any]] = []
    for bet in all_bets:
        p_imp = implied_probability(bet.bookmaker_odds)
        comparison_rows.append(
            {
                "market": bet.market,
                "selection": bet.selection,
                "line": bet.line,
                "odds": bet.bookmaker_odds,
                "p_model": bet.p_model,
                "p_implied": p_imp,
                "edge": bet.p_model - p_imp,
                "ev": bet.ev,
                "ev_pct": bet.ev_pct,
                "kelly_pct": bet.kelly_pct,
                "recommended": bet.recommended,
            }
        )

    return MatchPrediction(
        home_team=home_team,
        away_team=away_team,
        lambda_home=lam,
        lambda_away=mu,
        all_bets=all_bets,
        value_bets=value_bets,
        comparison_rows=comparison_rows,
        model_summary=model.summary(),
        min_ev=min_ev,
    )


def format_match_card(prediction: MatchPrediction) -> str:
    """Render a console Match Card report (tiếng Việt)."""
    home = prediction.home_team
    away = prediction.away_team
    lam = prediction.lambda_home
    mu = prediction.lambda_away
    min_ev_pct = prediction.min_ev * 100.0

    w = 64
    bar = "=" * w
    thin = "-" * w
    lines: list[str] = [
        bar,
        f"  BÁO CÁO PHÂN TÍCH TRẬN ĐẤU".center(w),
        f"  {home}  vs  {away}".center(w),
        bar,
        "",
        "⚽  CHỈ SỐ xG KỲ VỌNG (Dixon–Coles)",
        thin,
        f"  xG chủ nhà ({home}):  {lam:.2f}",
        f"  xG đội khách ({away}): {mu:.2f}",
        f"  Tổng xG dự kiến:       {lam + mu:.2f}",
        "",
        "📊  SO SÁNH XÁC SUẤT  (P_model vs P_implied = 1/Odds)",
        thin,
        f"  {'Cửa cược':<22} {'Odds':>6} {'P_model':>9} {'P_nhà cái':>10} {'EV%':>8}",
        f"  {'-'*22} {'-'*6} {'-'*9} {'-'*10} {'-'*8}",
    ]

    market_vi = {"1X2": "1X2", "OU": "Tài/Xỉu", "AH": "Chấp Á"}
    for row in prediction.comparison_rows:
        label = _selection_label_vi(str(row["selection"]))
        mkt = market_vi.get(str(row["market"]), str(row["market"]))
        name = f"[{mkt}] {label}"
        if len(name) > 22:
            name = name[:21] + "…"
        lines.append(
            f"  {name:<22} {row['odds']:>6.2f} {row['p_model']:>8.1%} "
            f"{row['p_implied']:>9.1%} {row['ev_pct']:>+7.1f}%"
        )

    lines.extend(["", "🎯  DANH SÁCH ĐỀ XUẤT VALUE BET  (EV ≥ {:.0f}%)".format(min_ev_pct), thin])

    if not prediction.value_bets:
        lines.extend(
            [
                "  ⚠️  KHÔNG NÊN ĐẶT CƯỢC",
                "      Nhà cái ra kèo rất chuẩn xác (mọi cửa có EV < {:.0f}%).".format(
                    min_ev_pct
                ),
            ]
        )
    else:
        for i, bet in enumerate(prediction.value_bets, start=1):
            label = _selection_label_vi(bet.selection)
            mkt = market_vi.get(bet.market, bet.market)
            lines.extend(
                [
                    f"  #{i}  [{mkt}] {label}",
                    f"      Odds nhà cái : {bet.bookmaker_odds:.2f}"
                    f"  (công bằng {bet.fair_odds:.2f})",
                    f"      P_model      : {bet.p_model:.1%}",
                    f"      EV           : {bet.ev_pct:+.1f}%",
                    f"      Kelly (25%)  : {bet.kelly_pct:.2f}% vốn",
                    "",
                ]
            )
        # drop trailing blank if present
        if lines[-1] == "":
            lines.pop()

    lines.extend(
        [
            "",
            thin,
            "  Ghi chú: EV = P_model × Odds − 1 · Kelly = 0.10 × EV / (Odds − 1) "
            f"(trần {MAX_STAKE_PCT:.0%} bankroll)",
            "  Chỉ đặt cược khi EV ≥ {:.0f}% (theo .cursorrules).".format(min_ev_pct),
            bar,
        ]
    )
    # Remove the accidental empty line from the False branch
    return "\n".join(line for line in lines if line is not None)
