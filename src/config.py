r"""Central quantitative defaults for the EPL / UWCL value-betting stack.

Tuned from walk-forward ablation (2026-09):
- Ensemble ``w_ML=0.2`` beat higher ML weights (LightGBM overconfidence).
- Fractional Kelly 10% + 1% bankroll stake cap cut drawdowns vs Quarter-Kelly.
"""

from __future__ import annotations

# Ensemble blend: P = (1 − w)·P_DC + w·P_ML
DEFAULT_W_ML: float = 0.20

# Fractional Kelly multiplier (full Kelly × this). Was 0.25 (Quarter Kelly).
DEFAULT_KELLY_FRACTION: float = 0.10
MAX_KELLY_FRACTION: float = 0.10  # alias / hard policy ceiling for UI & strategy

# Hard cap on stake as a fraction of bankroll (1.0% per bet).
MAX_STAKE_PCT: float = 0.01

# System-wide paper / alert exposure: at most this many bets per VN calendar day
# after per-match correlation dedupe (highest EV only).
MAX_BETS_PER_DAY: int = 5

# Near-kickoff window (minutes before KO) to snapshot live book odds → closing_odds.
CLOSE_ODDS_WINDOW_MIN_MINUTES: int = 15
CLOSE_ODDS_WINDOW_MAX_MINUTES: int = 30

# Value-bet floor (unchanged).
DEFAULT_MIN_EV: float = 0.05
# Hard sanity cap — drop absurd EV (overconfident longshots / bad probs).
MAX_EV: float = 0.50
# 1X2 longshot guard: odds above this with model p above threshold → skip.
LONGSHOT_ODDS_MIN: float = 15.0
LONGSHOT_P_MAX: float = 0.10
# Clubs with fewer historical matches use weak-tier Dixon–Coles priors.
MIN_TEAM_MATCHES: int = 5
# Alias kept for older call sites / docs.
MIN_TEAM_MATCHES_FOR_STRENGTH: int = MIN_TEAM_MATCHES

# LightGBM probability calibration (sklearn CalibratedClassifierCV).
ML_CALIBRATION_METHOD: str = "sigmoid"  # or "isotonic"
ML_CALIBRATION_MIN_ROWS: int = 40
