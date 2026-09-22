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

# Value-bet floor (unchanged).
DEFAULT_MIN_EV: float = 0.05

# LightGBM probability calibration (sklearn CalibratedClassifierCV).
ML_CALIBRATION_METHOD: str = "sigmoid"  # or "isotonic"
ML_CALIBRATION_MIN_ROWS: int = 40
