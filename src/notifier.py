r"""Telegram alerts for EPL Value Bet recommendations.

Uses the Bot API ``sendMessage`` endpoint via ``requests``.
Configure ``bot_token`` + ``chat_id`` from the Streamlit sidebar or ``.env``.
"""

from __future__ import annotations

import logging
from typing import Any, Mapping, Sequence

import pandas as pd
import requests

TELEGRAM_API = "https://api.telegram.org/bot{token}/sendMessage"
logger = logging.getLogger(__name__)


def format_value_bet_alert(
    *,
    home_team: str,
    away_team: str,
    selection: str,
    odds: float,
    ev: float,
    kelly_pct: float,
    stake_amount: float,
    match_date: str | None = None,
    market: str = "1X2",
    league: str = "EPL",
) -> str:
    """Build a rich HTML alert (Telegram parse_mode=HTML)."""
    when = match_date or "TBD"
    ev_pct = float(ev) * 100.0 if abs(float(ev)) <= 2 else float(ev)
    # If caller already passed ev as percent (> 1 looks like 12.4), keep as-is.
    if abs(float(ev)) > 2:
        ev_pct = float(ev)
    tag = str(league or "EPL").strip().upper() or "EPL"
    return (
        f"⚽ <b>[{tag} Value Bet Alert]</b>\n"
        f"📌 Trận: <b>{home_team}</b> vs <b>{away_team}</b>"
        f"{f' ({when})' if when and when != 'TBD' else ''}\n"
        f"🏟 Thị trường: <b>{market}</b>\n"
        f"🎯 Cửa đặt: <b>{selection}</b>\n"
        f"📊 Odds nhà cái: <b>{float(odds):.2f}</b> | "
        f"EV: <b>{ev_pct:+.1f}%</b>\n"
        f"💰 Vốn khuyến nghị: <b>{float(kelly_pct):.1f}% Kelly "
        f"(~${float(stake_amount):.1f})</b>"
    )


def send_telegram_message(
    bot_token: str,
    chat_id: str,
    text: str,
    *,
    parse_mode: str = "HTML",
    timeout: float = 15.0,
) -> dict[str, Any]:
    """Send one Telegram message. Returns API JSON (raises on HTTP/API error).

    Errors include HTTP status code and Telegram ``description`` when present
    (e.g. ``401 Unauthorized``, ``400 Bad Request: chat not found``).
    """
    token = (bot_token or "").strip()
    chat = str(chat_id or "").strip()
    if not token or not chat:
        raise ValueError("Thiếu bot_token hoặc chat_id Telegram.")

    url = TELEGRAM_API.format(token=token)
    payload = {
        "chat_id": chat,
        "text": text,
        "parse_mode": parse_mode,
        "disable_web_page_preview": True,
    }

    try:
        resp = requests.post(url, json=payload, timeout=timeout)
    except requests.Timeout as exc:
        logger.exception("Telegram POST timeout")
        raise RuntimeError(f"Telegram request timeout after {timeout}s: {exc}") from exc
    except requests.RequestException as exc:
        logger.exception("Telegram POST failed")
        raise RuntimeError(
            f"Telegram request error ({type(exc).__name__}): {exc}"
        ) from exc

    try:
        data = resp.json() if resp.content else {}
    except Exception:
        data = {"raw": (resp.text or "")[:500]}

    if resp.status_code != 200 or not (isinstance(data, dict) and data.get("ok")):
        desc = ""
        if isinstance(data, dict):
            desc = str(data.get("description") or "")
        desc = desc or resp.reason or resp.text[:200]
        err = f"HTTP {resp.status_code}: {desc}"
        logger.error("Telegram API error: %s | body=%s", err, data)
        raise RuntimeError(f"Telegram API lỗi — {err}")

    logger.info(
        "Telegram message sent (message_id=%s)",
        data.get("result", {}).get("message_id"),
    )
    return data


def _row_get(row: Mapping[str, Any] | pd.Series, *keys: str, default: Any = None) -> Any:
    for k in keys:
        if isinstance(row, pd.Series):
            if k in row.index and pd.notna(row[k]):
                return row[k]
        elif k in row and row[k] is not None:
            return row[k]
    return default


def send_telegram_value_bets(
    recommendations_df: pd.DataFrame | Sequence[Mapping[str, Any]],
    bot_token: str,
    chat_id: str,
    *,
    bankroll: float = 1000.0,
    min_ev: float = 0.05,
    markets: Sequence[str] = ("1X2",),
    league: str = "EPL",
) -> dict[str, Any]:
    """Send one Telegram alert per Value Bet row meeting ``min_ev``.

    Parameters
    ----------
    recommendations_df:
        DataFrame / list of dicts with at least
        ``home_team``, ``away_team``, ``selection``, ``bookmaker_odds`` (or ``odds``),
        ``ev`` or ``ev_pct``, and optionally ``kelly_fraction`` / ``kelly_pct``,
        ``market``, ``match_date`` / ``kickoff``.
    bot_token, chat_id:
        Telegram Bot credentials.
    bankroll:
        Used to compute stake when stake column is absent.
    min_ev:
        Minimum EV (fraction, e.g. 0.05) to include.
    markets:
        Only alert these markets (default 1X2).
    league:
        Competition tag in the alert header (``EPL`` / ``UWCL``).
    """
    if recommendations_df is None:
        return {"sent": 0, "skipped": 0, "errors": []}

    if isinstance(recommendations_df, pd.DataFrame):
        rows = recommendations_df.to_dict(orient="records")
    else:
        rows = list(recommendations_df)

    allowed = {str(m).upper() for m in markets}
    sent = 0
    skipped = 0
    errors: list[str] = []
    league_tag = str(league or "EPL").strip().upper() or "EPL"

    for row in rows:
        market = str(_row_get(row, "market", default="1X2") or "1X2").upper()
        if allowed and market not in allowed:
            skipped += 1
            continue

        ev = _row_get(row, "ev", default=None)
        ev_pct = _row_get(row, "ev_pct", default=None)
        if ev is None and ev_pct is not None:
            ev = float(ev_pct) / 100.0
        if ev is None:
            skipped += 1
            continue
        ev_f = float(ev)
        # Accept either fraction or already-percent
        ev_frac = ev_f / 100.0 if abs(ev_f) > 2 else ev_f
        if ev_frac < float(min_ev):
            skipped += 1
            continue

        odds = float(_row_get(row, "bookmaker_odds", "odds", default=0) or 0)
        if odds <= 1.0:
            skipped += 1
            continue

        kelly_frac = _row_get(row, "kelly_fraction", default=None)
        kelly_pct = _row_get(row, "kelly_pct", default=None)
        if kelly_frac is None and kelly_pct is not None:
            kelly_frac = float(kelly_pct) / 100.0
        if kelly_frac is None:
            kelly_frac = 0.0
        kelly_frac = float(kelly_frac)
        kelly_pct_v = kelly_frac * 100.0

        stake = _row_get(row, "stake", "stake_amount", default=None)
        if stake is None:
            stake = float(bankroll) * kelly_frac
        stake = float(stake)

        home = str(_row_get(row, "home_team", "HomeTeam", default="?"))
        away = str(_row_get(row, "away_team", "AwayTeam", default="?"))
        selection = str(_row_get(row, "selection", default="?"))
        when = _row_get(row, "match_date", "kickoff", "Kickoff", default=None)
        if when is not None and not isinstance(when, str):
            try:
                when = pd.Timestamp(when).strftime("%d/%m - %H:%M")
            except Exception:
                when = str(when)

        text = format_value_bet_alert(
            home_team=home,
            away_team=away,
            selection=selection,
            odds=odds,
            ev=ev_frac,
            kelly_pct=kelly_pct_v,
            stake_amount=stake,
            match_date=str(when) if when else None,
            market=market,
            league=league_tag,
        )
        try:
            send_telegram_message(bot_token, chat_id, text)
            sent += 1
        except Exception as exc:  # noqa: BLE001
            msg = f"{home} vs {away}: {exc}"
            logger.error("Failed to send VB alert: %s", msg)
            errors.append(msg)

    return {"sent": sent, "skipped": skipped, "errors": errors}


def format_recheck_summary(
    *,
    league: str = "EPL",
    settled: int = 0,
    wins: int = 0,
    losses: int = 0,
    pushes: int = 0,
    session_pnl: float = 0.0,
    hit_rate: float | None = None,
    realised_pnl: float | None = None,
    roi: float | None = None,
    avg_clv: float | None = None,
    brier: float | None = None,
    retrain_ok: bool | None = None,
    retrain_detail: str | None = None,
    skipped: int = 0,
) -> str:
    """Short HTML Telegram summary after post-match recheck / retrain."""
    tag = str(league or "EPL").strip().upper() or "EPL"
    decided = int(wins) + int(losses)
    if hit_rate is None:
        hit = (float(wins) / decided) if decided else float("nan")
    else:
        hit = float(hit_rate)

    def _pct(x: float | None, digits: int = 1) -> str:
        if x is None or x != x:  # NaN
            return "n/a"
        return f"{float(x) * 100.0:+.{digits}f}%"

    def _num(x: float | None, digits: int = 2) -> str:
        if x is None or x != x:
            return "n/a"
        return f"{float(x):+.{digits}f}"

    def _brier(x: float | None) -> str:
        if x is None or x != x:
            return "n/a"
        return f"{float(x):.3f}"

    hit_s = f"{hit * 100:.0f}%" if hit == hit else "n/a"
    pnl_show = realised_pnl if realised_pnl is not None else session_pnl
    lines = [
        f"📊 <b>[{tag} Recheck]</b>",
        (
            f"Settled: <b>{int(settled)}</b> "
            f"({int(wins)}W/{int(losses)}L/{int(pushes)}P) · Hit <b>{hit_s}</b>"
        ),
        f"PnL: <b>{_num(pnl_show)}</b> · ROI: <b>{_pct(roi)}</b>",
        f"Avg CLV: <b>{_pct(avg_clv)}</b> · Brier: <b>{_brier(brier)}</b>",
    ]
    if skipped:
        lines.append(f"Skipped (no result yet): {int(skipped)}")
    if retrain_ok is True:
        detail = retrain_detail or "DC + LGBM"
        lines.append(f"Retrain: <b>OK</b> ({detail})")
    elif retrain_ok is False:
        detail = retrain_detail or "failed"
        lines.append(f"Retrain: <b>FAIL</b> — {detail}")
    elif retrain_ok is None and retrain_detail:
        lines.append(f"Retrain: {retrain_detail}")
    return "\n".join(lines)
