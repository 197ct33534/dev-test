r"""Calibration & scoring metrics for probabilistic football models.

Provides Brier score, log-loss, and reliability (calibration) tables for
1X2 / Over-Under / Asian Handicap predictions, optionally sliced by season.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd


def brier_score(y_true: Sequence[float | int], p_pred: Sequence[float]) -> float:
    """Mean squared error between outcomes and predicted probabilities."""
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(p_pred, dtype=float)
    mask = np.isfinite(y) & np.isfinite(p)
    if not mask.any():
        return float("nan")
    return float(np.mean((p[mask] - y[mask]) ** 2))


def log_loss(
    y_true: Sequence[float | int],
    p_pred: Sequence[float],
    *,
    eps: float = 1e-15,
) -> float:
    """Binary cross-entropy (natural log). Lower is better."""
    y = np.asarray(y_true, dtype=float)
    p = np.clip(np.asarray(p_pred, dtype=float), eps, 1.0 - eps)
    mask = np.isfinite(y) & np.isfinite(p)
    if not mask.any():
        return float("nan")
    y_m, p_m = y[mask], p[mask]
    return float(-np.mean(y_m * np.log(p_m) + (1.0 - y_m) * np.log(1.0 - p_m)))


def multiclass_brier(
    y_labels: Sequence[str],
    probs: pd.DataFrame | dict[str, Sequence[float]],
    classes: Sequence[str] = ("H", "D", "A"),
) -> float:
    """Multi-class Brier: mean over classes of (p_k - 1_{y=k})^2."""
    y = np.asarray(list(y_labels), dtype=object)
    if isinstance(probs, dict):
        p_df = pd.DataFrame({c: np.asarray(probs[c], dtype=float) for c in classes})
    else:
        p_df = probs[list(classes)].astype(float)
    if len(y) != len(p_df):
        raise ValueError("y_labels and probs length mismatch")
    total = 0.0
    for c in classes:
        y_bin = (y == c).astype(float)
        total += float(np.mean((p_df[c].to_numpy() - y_bin) ** 2))
    return total / float(len(classes))


def multiclass_log_loss(
    y_labels: Sequence[str],
    probs: pd.DataFrame | dict[str, Sequence[float]],
    classes: Sequence[str] = ("H", "D", "A"),
    *,
    eps: float = 1e-15,
) -> float:
    """Multi-class log-loss: -mean log p(y)."""
    y = np.asarray(list(y_labels), dtype=object)
    if isinstance(probs, dict):
        p_df = pd.DataFrame({c: np.asarray(probs[c], dtype=float) for c in classes})
    else:
        p_df = probs[list(classes)].astype(float)
    rows: list[float] = []
    for i, label in enumerate(y):
        if label not in classes:
            continue
        p = float(np.clip(p_df.iloc[i][label], eps, 1.0))
        rows.append(-np.log(p))
    if not rows:
        return float("nan")
    return float(np.mean(rows))


def reliability_table(
    y_true: Sequence[float | int],
    p_pred: Sequence[float],
    *,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Reliability diagram data: mean predicted vs empirical frequency per bin."""
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(p_pred, dtype=float)
    mask = np.isfinite(y) & np.isfinite(p)
    y, p = y[mask], p[mask]
    if len(p) == 0:
        return pd.DataFrame(
            columns=["bin", "p_low", "p_high", "p_mean", "y_freq", "n", "gap"]
        )

    edges = np.linspace(0.0, 1.0, int(n_bins) + 1)
    rows: list[dict[str, Any]] = []
    for i in range(len(edges) - 1):
        lo, hi = float(edges[i]), float(edges[i + 1])
        if i == len(edges) - 2:
            sel = (p >= lo) & (p <= hi)
        else:
            sel = (p >= lo) & (p < hi)
        n = int(sel.sum())
        if n == 0:
            rows.append(
                {
                    "bin": i,
                    "p_low": lo,
                    "p_high": hi,
                    "p_mean": float("nan"),
                    "y_freq": float("nan"),
                    "n": 0,
                    "gap": float("nan"),
                }
            )
            continue
        p_mean = float(p[sel].mean())
        y_freq = float(y[sel].mean())
        rows.append(
            {
                "bin": i,
                "p_low": lo,
                "p_high": hi,
                "p_mean": p_mean,
                "y_freq": y_freq,
                "n": n,
                "gap": abs(p_mean - y_freq),
            }
        )
    return pd.DataFrame(rows)


def expected_calibration_error(
    y_true: Sequence[float | int],
    p_pred: Sequence[float],
    *,
    n_bins: int = 10,
) -> float:
    """Weighted ECE = Σ (n_bin / N) · |p_mean - y_freq|."""
    table = reliability_table(y_true, p_pred, n_bins=n_bins)
    if table.empty or int(table["n"].sum()) == 0:
        return float("nan")
    n_total = float(table["n"].sum())
    ece = 0.0
    for _, row in table.iterrows():
        if int(row["n"]) == 0 or not np.isfinite(row["gap"]):
            continue
        ece += (float(row["n"]) / n_total) * float(row["gap"])
    return float(ece)


def closing_line_value(odds_entered: float, odds_closing: float) -> float:
    """CLV = (odds_entered / odds_closing) − 1.

    Positive ⇒ got a better (higher) price than the closing reference.
    """
    o_in = float(odds_entered)
    o_cl = float(odds_closing)
    if o_in <= 1.0 or o_cl <= 1.0 or not np.isfinite(o_in) or not np.isfinite(o_cl):
        return float("nan")
    return (o_in / o_cl) - 1.0


def apply_commission(pnl: float, commission_pct: float) -> float:
    """Subtract commission from **winning** PnL only (exchange-style)."""
    c = float(max(0.0, min(1.0, commission_pct)))
    p = float(pnl)
    if p > 0 and c > 0:
        return p * (1.0 - c)
    return p


def summarize_predictions(
    preds: pd.DataFrame,
    *,
    n_bins: int = 10,
    season_col: str = "season",
    market_col: str = "market",
) -> dict[str, Any]:
    """Build calibration summary from a prediction log.

    Expected columns for binary rows: ``p_model``, ``y``, optional ``market``,
    ``season``. Optional multiclass: ``p_H``, ``p_D``, ``p_A``, ``ftr``.
    """
    out: dict[str, Any] = {
        "overall": {},
        "by_market": pd.DataFrame(),
        "by_season": pd.DataFrame(),
        "reliability": pd.DataFrame(),
    }
    if preds is None or preds.empty:
        return out

    df = preds.copy()
    if "p_model" in df.columns and "y" in df.columns:
        binary = df.dropna(subset=["p_model", "y"])
        if not binary.empty:
            out["overall"] = {
                "n": int(len(binary)),
                "brier": brier_score(binary["y"], binary["p_model"]),
                "log_loss": log_loss(binary["y"], binary["p_model"]),
                "ece": expected_calibration_error(
                    binary["y"], binary["p_model"], n_bins=n_bins
                ),
            }
            out["reliability"] = reliability_table(
                binary["y"], binary["p_model"], n_bins=n_bins
            )

    if market_col in df.columns and "p_model" in df.columns and "y" in df.columns:
        rows = []
        for mkt, sub in df.groupby(market_col, dropna=False):
            sub = sub.dropna(subset=["p_model", "y"])
            if sub.empty:
                continue
            rows.append(
                {
                    "market": mkt,
                    "n": int(len(sub)),
                    "brier": brier_score(sub["y"], sub["p_model"]),
                    "log_loss": log_loss(sub["y"], sub["p_model"]),
                    "ece": expected_calibration_error(
                        sub["y"], sub["p_model"], n_bins=n_bins
                    ),
                }
            )
        out["by_market"] = pd.DataFrame(rows)

    if season_col in df.columns and "p_model" in df.columns and "y" in df.columns:
        rows = []
        for season, sub in df.groupby(season_col, dropna=False):
            sub = sub.dropna(subset=["p_model", "y"])
            if sub.empty:
                continue
            rows.append(
                {
                    "season": season,
                    "n": int(len(sub)),
                    "brier": brier_score(sub["y"], sub["p_model"]),
                    "log_loss": log_loss(sub["y"], sub["p_model"]),
                    "ece": expected_calibration_error(
                        sub["y"], sub["p_model"], n_bins=n_bins
                    ),
                }
            )
        out["by_season"] = pd.DataFrame(rows)

    need = {"p_H", "p_D", "p_A", "ftr"}
    if need.issubset(df.columns):
        mc = df.dropna(subset=list(need))
        if not mc.empty:
            probs = mc[["p_H", "p_D", "p_A"]].rename(
                columns={"p_H": "H", "p_D": "D", "p_A": "A"}
            )
            out["overall"]["multiclass_brier"] = multiclass_brier(mc["ftr"], probs)
            out["overall"]["multiclass_log_loss"] = multiclass_log_loss(
                mc["ftr"], probs
            )

    return out


def format_calibration_report(summary: dict[str, Any]) -> str:
    """Plain-text calibration report."""
    overall = summary.get("overall") or {}
    lines = ["=== Calibration Report ==="]
    if overall:
        lines.append(f"N predictions  : {overall.get('n', 0)}")
        b = overall.get("brier")
        ll = overall.get("log_loss")
        ece = overall.get("ece")
        lines.append(f"Brier          : {b:.4f}" if b == b else "Brier          : n/a")
        lines.append(
            f"Log-loss       : {ll:.4f}" if ll == ll else "Log-loss       : n/a"
        )
        lines.append(
            f"ECE (10 bins)  : {ece:.4f}" if ece == ece else "ECE            : n/a"
        )
        mb = overall.get("multiclass_brier")
        mll = overall.get("multiclass_log_loss")
        if mb is not None and mb == mb:
            lines.append(f"1X2 MC-Brier   : {mb:.4f}")
        if mll is not None and mll == mll:
            lines.append(f"1X2 MC-LogLoss : {mll:.4f}")

    by_m = summary.get("by_market")
    if isinstance(by_m, pd.DataFrame) and not by_m.empty:
        lines.append("")
        lines.append("-- By market --")
        for _, r in by_m.iterrows():
            lines.append(
                f"  {r['market']:<4} n={int(r['n']):4d}  "
                f"Brier={r['brier']:.4f}  LogL={r['log_loss']:.4f}  "
                f"ECE={r['ece']:.4f}"
            )

    by_s = summary.get("by_season")
    if isinstance(by_s, pd.DataFrame) and not by_s.empty:
        lines.append("")
        lines.append("-- By season --")
        for _, r in by_s.iterrows():
            lines.append(
                f"  {r['season']}: n={int(r['n']):4d}  "
                f"Brier={r['brier']:.4f}  LogL={r['log_loss']:.4f}"
            )
    return "\n".join(lines)
