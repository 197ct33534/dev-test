"""Quant report dashboard for walk-forward backtests."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from src.validation.metrics import QuantMetrics

try:
    from rich import box
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table

    _HAS_RICH = True
except ImportError:  # pragma: no cover
    _HAS_RICH = False
    box = None  # type: ignore[assignment]


@dataclass
class QuantReport:
    """Aggregated probability / financial / CLV / risk / tier metrics."""

    # Probability
    brier: float = float("nan")
    log_loss: float = float("nan")
    ece: float = float("nan")
    n_prob_samples: int = 0

    # Financial
    total_bets: int = 0
    win_rate: float = float("nan")
    net_pnl: float = 0.0
    total_staked: float = 0.0
    roi: float = float("nan")
    yield_: float = float("nan")
    final_bankroll: float = 0.0
    initial_bankroll: float = 0.0

    # Market edge
    mean_clv: float = float("nan")
    pct_positive_clv: float = float("nan")
    n_clv: int = 0

    # Risk
    max_drawdown_pct: float = float("nan")
    sharpe_annualized: float = float("nan")
    max_consecutive_losses: int = 0

    # Model tiers
    tier_counts: dict[str, int] = field(default_factory=dict)

    # Meta
    n_matches: int = 0
    n_priced: int = 0
    n_skipped_gate: int = 0
    mode: str = "live"


def build_quant_report(
    *,
    settled_bets: Sequence[Mapping[str, Any]],
    equity_curve: Sequence[float],
    y_true: Sequence[int],
    y_prob: Sequence[float],
    tier_counts: Mapping[str, int],
    initial_bankroll: float,
    final_bankroll: float,
    n_matches: int = 0,
    n_priced: int = 0,
    n_skipped_gate: int = 0,
    mode: str = "live",
) -> QuantReport:
    """Compute dashboard metrics from settled paper bets + equity path."""
    bets = list(settled_bets)
    total_bets = len(bets)
    wins = sum(1 for b in bets if _is_win(b.get("status")))
    losses = sum(1 for b in bets if _is_loss(b.get("status")))
    decided = wins + losses

    stakes = [_f(b.get("stake") or b.get("final_stake_amount")) for b in bets]
    pnls = [_f(b.get("pnl")) for b in bets]
    total_staked = sum(s for s in stakes if math.isfinite(s))
    net_pnl = sum(p for p in pnls if math.isfinite(p))

    roi = (net_pnl / total_staked) if total_staked > 0 else float("nan")
    yields = [
        (p / s)
        for p, s in zip(pnls, stakes)
        if math.isfinite(p) and math.isfinite(s) and s > 0
    ]
    yield_ = float(sum(yields) / len(yields)) if yields else float("nan")

    clvs = [
        _f(b.get("clv_value"))
        for b in bets
        if b.get("clv_value") is not None and math.isfinite(_f(b.get("clv_value")))
    ]
    mean_clv = float(sum(clvs) / len(clvs)) if clvs else float("nan")
    pct_pos = (
        100.0 * sum(1 for c in clvs if c > 0.0) / len(clvs) if clvs else float("nan")
    )

    mdd = QuantMetrics.max_drawdown(list(equity_curve))
    mdd_pct = (
        float(mdd) * 100.0 if math.isfinite(mdd) else float("nan")
    )
    sharpe = _annualized_sharpe(equity_curve)
    max_consec = _max_consecutive_losses(
        [str(b.get("status") or "") for b in bets]
    )

    y_t = list(y_true)
    y_p = list(y_prob)
    brier = log_loss = ece = float("nan")
    n_prob = 0
    if y_t and len(y_t) == len(y_p):
        n_prob = len(y_t)
        brier = QuantMetrics.brier_score(y_t, y_p)
        log_loss = QuantMetrics.log_loss_score(y_t, y_p)
        ece = QuantMetrics.expected_calibration_error(y_t, y_p, n_bins=10)

    tiers = {
        "A": int(tier_counts.get("A", 0) or tier_counts.get("MODEL_TIER_A", 0)),
        "B": int(tier_counts.get("B", 0) or tier_counts.get("MODEL_TIER_B", 0)),
        "X": int(tier_counts.get("X", 0) or tier_counts.get("MODEL_TIER_X", 0)),
    }
    # Fold any leftover keys into letter buckets.
    for k, v in tier_counts.items():
        key = str(k).upper()
        if key in tiers:
            continue
        if "TIER_A" in key or key.endswith("_A"):
            tiers["A"] += int(v)
        elif "TIER_B" in key or key.endswith("_B"):
            tiers["B"] += int(v)
        elif "TIER_X" in key or key.endswith("_X"):
            tiers["X"] += int(v)

    return QuantReport(
        brier=brier,
        log_loss=log_loss,
        ece=ece,
        n_prob_samples=n_prob,
        total_bets=total_bets,
        win_rate=(wins / decided) if decided else float("nan"),
        net_pnl=net_pnl,
        total_staked=total_staked,
        roi=roi,
        yield_=yield_,
        final_bankroll=float(final_bankroll),
        initial_bankroll=float(initial_bankroll),
        mean_clv=mean_clv,
        pct_positive_clv=pct_pos,
        n_clv=len(clvs),
        max_drawdown_pct=mdd_pct,
        sharpe_annualized=sharpe,
        max_consecutive_losses=max_consec,
        tier_counts=tiers,
        n_matches=n_matches,
        n_priced=n_priced,
        n_skipped_gate=n_skipped_gate,
        mode=mode,
    )


def render_quant_report(report: QuantReport, *, title: str | None = None) -> str:
    """Render a Quant Report Dashboard (rich tables when available)."""
    hdr = title or "Quant Report Dashboard"
    if _HAS_RICH:
        return _render_rich(report, hdr)
    return _render_plain(report, hdr)


def _render_rich(report: QuantReport, title: str) -> str:
    import io

    buf = io.StringIO()
    # ASCII boxes stay printable on Windows cp1252 consoles.
    console = Console(file=buf, width=88, force_terminal=False, soft_wrap=True)
    ascii_box = box.ASCII if box is not None else None
    console.print(Panel.fit(f"{title}  |  mode={report.mode}", box=ascii_box))

    prob = Table(
        title="Probability",
        show_header=True,
        header_style="bold",
        box=ascii_box,
    )
    prob.add_column("Metric")
    prob.add_column("Value", justify="right")
    prob.add_row("Brier", _fmt(report.brier, 4))
    prob.add_row("Log Loss", _fmt(report.log_loss, 4))
    prob.add_row("ECE", _fmt(report.ece, 4))
    prob.add_row("N samples", str(report.n_prob_samples))
    console.print(prob)

    fin = Table(
        title="Financial",
        show_header=True,
        header_style="bold",
        box=ascii_box,
    )
    fin.add_column("Metric")
    fin.add_column("Value", justify="right")
    fin.add_row("Total Bets", str(report.total_bets))
    fin.add_row("Win Rate", _fmt_pct(report.win_rate))
    fin.add_row("Net PnL", _fmt_money(report.net_pnl))
    fin.add_row("Total Staked", _fmt_money(report.total_staked))
    fin.add_row("ROI", _fmt_pct(report.roi))
    fin.add_row("Yield", _fmt_pct(report.yield_))
    fin.add_row("Initial Bankroll", _fmt_money(report.initial_bankroll))
    fin.add_row("Final Bankroll", _fmt_money(report.final_bankroll))
    console.print(fin)

    mkt = Table(
        title="Market Edge",
        show_header=True,
        header_style="bold",
        box=ascii_box,
    )
    mkt.add_column("Metric")
    mkt.add_column("Value", justify="right")
    mkt.add_row("Mean CLV", _fmt(report.mean_clv, 4))
    mkt.add_row("% Positive CLV", _fmt(report.pct_positive_clv, 1) + "%")
    mkt.add_row("N CLV", str(report.n_clv))
    console.print(mkt)

    risk = Table(
        title="Risk",
        show_header=True,
        header_style="bold",
        box=ascii_box,
    )
    risk.add_column("Metric")
    risk.add_column("Value", justify="right")
    risk.add_row("Max Drawdown %", _fmt(report.max_drawdown_pct, 2) + "%")
    risk.add_row("Sharpe (ann.)", _fmt(report.sharpe_annualized, 3))
    risk.add_row("Max Consecutive Losses", str(report.max_consecutive_losses))
    console.print(risk)

    tiers = Table(
        title="Model Tier Distribution",
        show_header=True,
        header_style="bold",
        box=ascii_box,
    )
    tiers.add_column("Tier")
    tiers.add_column("Count", justify="right")
    for letter in ("A", "B", "X"):
        tiers.add_row(letter, str(report.tier_counts.get(letter, 0)))
    console.print(tiers)

    meta = Table(title="Replay Meta", show_header=False, box=ascii_box)
    meta.add_column("k")
    meta.add_column("v", justify="right")
    meta.add_row("Matches", str(report.n_matches))
    meta.add_row("Priced", str(report.n_priced))
    meta.add_row("Skipped (gate)", str(report.n_skipped_gate))
    console.print(meta)

    return buf.getvalue()


def _render_plain(report: QuantReport, title: str) -> str:
    lines = [
        "=" * 64,
        f" {title}  (mode={report.mode})",
        "=" * 64,
        "",
        "Probability",
        f"  Brier                 {_fmt(report.brier, 4)}",
        f"  Log Loss              {_fmt(report.log_loss, 4)}",
        f"  ECE                   {_fmt(report.ece, 4)}",
        f"  N samples             {report.n_prob_samples}",
        "",
        "Financial",
        f"  Total Bets            {report.total_bets}",
        f"  Win Rate              {_fmt_pct(report.win_rate)}",
        f"  Net PnL               {_fmt_money(report.net_pnl)}",
        f"  Total Staked          {_fmt_money(report.total_staked)}",
        f"  ROI                   {_fmt_pct(report.roi)}",
        f"  Yield                 {_fmt_pct(report.yield_)}",
        f"  Initial Bankroll      {_fmt_money(report.initial_bankroll)}",
        f"  Final Bankroll        {_fmt_money(report.final_bankroll)}",
        "",
        "Market Edge",
        f"  Mean CLV              {_fmt(report.mean_clv, 4)}",
        f"  % Positive CLV        {_fmt(report.pct_positive_clv, 1)}%",
        f"  N CLV                 {report.n_clv}",
        "",
        "Risk",
        f"  Max Drawdown %        {_fmt(report.max_drawdown_pct, 2)}%",
        f"  Sharpe (ann.)         {_fmt(report.sharpe_annualized, 3)}",
        f"  Max Consecutive Losses {report.max_consecutive_losses}",
        "",
        "Model Tier Distribution",
        f"  A                     {report.tier_counts.get('A', 0)}",
        f"  B                     {report.tier_counts.get('B', 0)}",
        f"  X                     {report.tier_counts.get('X', 0)}",
        "",
        "Replay Meta",
        f"  Matches               {report.n_matches}",
        f"  Priced                {report.n_priced}",
        f"  Skipped (gate)        {report.n_skipped_gate}",
        "=" * 64,
    ]
    return "\n".join(lines)


def _annualized_sharpe(equity_curve: Sequence[float]) -> float:
    """Daily-return Sharpe annualized with √365; NaN if insufficient data."""
    eq = [float(x) for x in equity_curve if math.isfinite(float(x))]
    if len(eq) < 3:
        return float("nan")
    rets: list[float] = []
    for i in range(1, len(eq)):
        prev = eq[i - 1]
        if prev <= 0:
            continue
        rets.append((eq[i] - prev) / prev)
    if len(rets) < 2:
        return float("nan")
    mean = sum(rets) / len(rets)
    var = sum((r - mean) ** 2 for r in rets) / (len(rets) - 1)
    std = math.sqrt(var) if var > 0 else 0.0
    if std <= 0:
        return float("nan")
    return float(mean / std * math.sqrt(365.0))


def _max_consecutive_losses(statuses: Sequence[str]) -> int:
    best = cur = 0
    for st in statuses:
        if _is_loss(st):
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def _is_win(status: Any) -> bool:
    s = str(status or "").upper()
    return s in {"WON", "WIN", "HALF_WIN"}


def _is_loss(status: Any) -> bool:
    s = str(status or "").upper()
    return s in {"LOST", "LOSS", "HALF_LOSS"}


def _f(v: Any) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def _fmt(v: float, digits: int = 3) -> str:
    if v is None or not math.isfinite(float(v)):
        return "n/a"
    return f"{float(v):.{digits}f}"


def _fmt_pct(v: float) -> str:
    if v is None or not math.isfinite(float(v)):
        return "n/a"
    # Accept fraction (0.12) or already-percent (>1.5 rare for win-rate).
    x = float(v)
    if abs(x) <= 1.5:
        x *= 100.0
    return f"{x:.2f}%"


def _fmt_money(v: float) -> str:
    if v is None or not math.isfinite(float(v)):
        return "n/a"
    return f"{float(v):,.2f}"


def merge_tier_counter(counter: Counter[str], model_tier: str | None) -> None:
    """Increment A/B/X buckets from a model_tier string."""
    raw = str(model_tier or "MODEL_TIER_X").upper()
    if "TIER_A" in raw or raw.endswith("_A") or raw == "A":
        counter["A"] += 1
    elif "TIER_B" in raw or raw.endswith("_B") or raw == "B":
        counter["B"] += 1
    else:
        counter["X"] += 1
