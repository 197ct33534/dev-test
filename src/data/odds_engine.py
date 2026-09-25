"""In-memory odds history engine with sharp / soft bookmaker classification.

v1 stores snapshots in process memory. The public interface is intentionally
lake-ready (match_id + as-of timestamp) so a later parquet/DB backend can
drop in without changing callers.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping, MutableMapping, Sequence

# Names that count as sharp Betfair Exchange (normalized).
_BETFAIR_EXCHANGE_MARKERS: frozenset[str] = frozenset(
    {"betfair exchange", "bf exchange", "betfair ex"}
)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _ensure_aware(ts: datetime) -> datetime:
    if ts.tzinfo is None:
        return ts.replace(tzinfo=timezone.utc)
    return ts


def normalize_bookmaker_name(name: str) -> str:
    """Lowercase + collapse whitespace for bookmaker matching."""
    return " ".join(str(name or "").strip().lower().split())


def classify_bookmaker(name: str) -> dict[str, bool]:
    """Return ``is_sharp`` / ``is_soft`` flags for a bookmaker name.

    Sharp (v1): Pinnacle and Betfair Exchange. Everything else is soft.
    """
    key = normalize_bookmaker_name(name)
    if not key:
        return {"is_sharp": False, "is_soft": True}

    if key in {"pinnacle", "pinnacle sports", "pinny"} or key.startswith("pinnacle"):
        return {"is_sharp": True, "is_soft": False}

    if key in _BETFAIR_EXCHANGE_MARKERS or (
        "betfair" in key and "exchange" in key
    ):
        return {"is_sharp": True, "is_soft": False}

    # Plain "betfair" / Betfair Sportsbook → soft (not exchange).
    return {"is_sharp": False, "is_soft": True}


def _parse_timestamp(raw: Any) -> datetime:
    if isinstance(raw, datetime):
        return _ensure_aware(raw)
    if raw is None:
        return _utc_now()
    text = str(raw).strip()
    if not text:
        return _utc_now()
    # Support trailing Z.
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    return _ensure_aware(datetime.fromisoformat(text))


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if v <= 1.0:
        return None
    return v


class OddsHistoryEngine:
    """Parse, classify, and query historical odds snapshots.

    Storage layout (in-memory)::

        {match_id: [ {timestamp, bookmaker, is_sharp, is_soft, market, ...}, ... ]}

    Entries are kept sorted by timestamp ascending per match.
    """

    def __init__(self) -> None:
        self._store: dict[str, list[dict[str, Any]]] = {}
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Sync core
    # ------------------------------------------------------------------

    def process_odds_snapshot(self, odds_payload: dict) -> dict:
        """Parse an odds payload, classify sharp/soft, and store rows.

        Accepted shapes (flexible)::

            {
              "match_id": "...",
              "timestamp": "2026-09-25T12:00:00+00:00",  # optional
              "bookmaker": "Pinnacle",                   # optional top-level
              "markets": [                               # optional list
                {"market": "1X2", "odds": {"H": 2.1, "D": 3.4, "A": 3.5}},
                ...
              ],
              # OR flat bookmaker list:
              "odds": [
                {"bookmaker": "Pinnacle", "market": "1X2",
                 "home": 2.1, "draw": 3.4, "away": 3.5, "timestamp": "..."},
              ],
              # OR nested by bookmaker:
              "bookmakers": [
                {"name": "Betfair Exchange", "markets": {...}, "timestamp": "..."}
              ]
            }

        Returns
        -------
        dict
            Summary with ``match_id``, ``n_rows``, ``n_sharp``, ``n_soft``,
            and the stored ``rows``.
        """
        if not isinstance(odds_payload, Mapping):
            raise TypeError("odds_payload must be a dict/mapping")

        match_id = str(
            odds_payload.get("match_id")
            or odds_payload.get("canonical_match_id")
            or odds_payload.get("fixture_id")
            or ""
        ).strip()
        if not match_id:
            raise ValueError("odds_payload missing match_id")

        default_ts = _parse_timestamp(
            odds_payload.get("timestamp")
            or odds_payload.get("observed_at")
            or odds_payload.get("as_of")
        )
        default_book = odds_payload.get("bookmaker") or odds_payload.get("source")

        rows: list[dict[str, Any]] = []
        rows.extend(
            self._rows_from_odds_list(
                match_id,
                odds_payload.get("odds"),
                default_ts=default_ts,
                default_book=default_book,
            )
        )
        rows.extend(
            self._rows_from_bookmakers(
                match_id,
                odds_payload.get("bookmakers"),
                default_ts=default_ts,
            )
        )
        rows.extend(
            self._rows_from_markets(
                match_id,
                odds_payload.get("markets"),
                default_ts=default_ts,
                default_book=default_book,
            )
        )

        # Single flat market dict at top level (home/draw/away or odds_home…).
        if not rows and any(
            k in odds_payload
            for k in ("home", "draw", "away", "odds_home", "H", "over", "under")
        ):
            rows.extend(
                self._rows_from_flat_market(
                    match_id,
                    odds_payload,
                    default_ts=default_ts,
                    default_book=default_book,
                )
            )

        if not rows:
            raise ValueError("odds_payload contained no parseable odds rows")

        bucket = self._store.setdefault(match_id, [])
        bucket.extend(rows)
        bucket.sort(key=lambda r: r["timestamp"])

        n_sharp = sum(1 for r in rows if r["is_sharp"])
        n_soft = sum(1 for r in rows if r["is_soft"])
        return {
            "match_id": match_id,
            "n_rows": len(rows),
            "n_sharp": n_sharp,
            "n_soft": n_soft,
            "rows": list(rows),
        }

    def get_sharp_consensus_odds(
        self,
        match_id: str,
        as_of_time: datetime,
    ) -> dict:
        """Closing / current sharp line as of ``as_of_time``.

        Uses the latest sharp snapshot at or before ``as_of_time``. When
        multiple sharp books share that timestamp, consensus is the
        per-selection arithmetic mean of decimal odds.

        Returns
        -------
        dict
            ``match_id``, ``as_of_time``, ``sources``, ``markets`` mapping
            market → consensus odds dict, plus ``n_snapshots``. Empty
            ``markets`` when no sharp data is available.
        """
        mid = str(match_id).strip()
        as_of = _ensure_aware(as_of_time)
        history = self._store.get(mid, [])
        eligible = [
            r
            for r in history
            if r["is_sharp"] and r["timestamp"] <= as_of
        ]
        if not eligible:
            return {
                "match_id": mid,
                "as_of_time": as_of.isoformat(),
                "sources": [],
                "markets": {},
                "n_snapshots": 0,
                "consensus_timestamp": None,
            }

        # Prefer the latest timestamp; average all sharp books at that tick.
        latest_ts = max(r["timestamp"] for r in eligible)
        at_latest = [r for r in eligible if r["timestamp"] == latest_ts]

        markets: dict[str, dict[str, float]] = {}
        # Group by market then selection.
        by_market: dict[str, dict[str, list[float]]] = {}
        sources: list[str] = []
        for r in at_latest:
            mkt = str(r.get("market") or "unknown")
            book = str(r.get("bookmaker") or "")
            if book and book not in sources:
                sources.append(book)
            sel_map = by_market.setdefault(mkt, {})
            odds_map = r.get("odds") or {}
            if isinstance(odds_map, Mapping):
                for sel, price in odds_map.items():
                    v = _as_float(price)
                    if v is None:
                        continue
                    sel_map.setdefault(str(sel), []).append(v)

        for mkt, sels in by_market.items():
            markets[mkt] = {
                sel: sum(vals) / len(vals) for sel, vals in sels.items() if vals
            }

        return {
            "match_id": mid,
            "as_of_time": as_of.isoformat(),
            "sources": sources,
            "markets": markets,
            "n_snapshots": len(at_latest),
            "consensus_timestamp": latest_ts.isoformat(),
        }

    def clear(self, match_id: str | None = None) -> None:
        """Drop stored snapshots (one match or all)."""
        if match_id is None:
            self._store.clear()
        else:
            self._store.pop(str(match_id).strip(), None)

    # ------------------------------------------------------------------
    # Async wrappers (natural for lake / concurrent ingest)
    # ------------------------------------------------------------------

    async def aprocess_odds_snapshot(self, odds_payload: dict) -> dict:
        """Async wrapper around :meth:`process_odds_snapshot`."""
        async with self._lock:
            return self.process_odds_snapshot(odds_payload)

    async def aget_sharp_consensus_odds(
        self,
        match_id: str,
        as_of_time: datetime,
    ) -> dict:
        """Async wrapper around :meth:`get_sharp_consensus_odds`."""
        async with self._lock:
            return self.get_sharp_consensus_odds(match_id, as_of_time)

    # ------------------------------------------------------------------
    # Parsers
    # ------------------------------------------------------------------

    def _annotate_book(self, bookmaker: str | None) -> dict[str, Any]:
        name = str(bookmaker or "unknown")
        flags = classify_bookmaker(name)
        return {"bookmaker": name, **flags}

    def _rows_from_odds_list(
        self,
        match_id: str,
        odds_list: Any,
        *,
        default_ts: datetime,
        default_book: Any,
    ) -> list[dict[str, Any]]:
        if not isinstance(odds_list, Sequence) or isinstance(odds_list, (str, bytes)):
            return []
        rows: list[dict[str, Any]] = []
        for item in odds_list:
            if not isinstance(item, Mapping):
                continue
            ts = _parse_timestamp(item.get("timestamp") or item.get("observed_at") or default_ts)
            book = item.get("bookmaker") or item.get("name") or default_book
            meta = self._annotate_book(str(book) if book is not None else None)
            market = str(item.get("market") or item.get("market_type") or "1X2")
            odds_map = self._extract_odds_map(item)
            if not odds_map:
                continue
            rows.append(
                {
                    "match_id": match_id,
                    "timestamp": ts,
                    "market": market,
                    "odds": odds_map,
                    **meta,
                }
            )
        return rows

    def _rows_from_bookmakers(
        self,
        match_id: str,
        bookmakers: Any,
        *,
        default_ts: datetime,
    ) -> list[dict[str, Any]]:
        if not isinstance(bookmakers, Sequence) or isinstance(bookmakers, (str, bytes)):
            return []
        rows: list[dict[str, Any]] = []
        for bk in bookmakers:
            if not isinstance(bk, Mapping):
                continue
            book = bk.get("name") or bk.get("bookmaker") or "unknown"
            ts = _parse_timestamp(bk.get("timestamp") or bk.get("observed_at") or default_ts)
            meta = self._annotate_book(str(book))
            markets = bk.get("markets")
            if isinstance(markets, Mapping):
                for mkt_name, mkt_odds in markets.items():
                    odds_map = self._extract_odds_map(
                        mkt_odds if isinstance(mkt_odds, Mapping) else {"odds": mkt_odds}
                    )
                    if not odds_map:
                        continue
                    rows.append(
                        {
                            "match_id": match_id,
                            "timestamp": ts,
                            "market": str(mkt_name),
                            "odds": odds_map,
                            **meta,
                        }
                    )
            elif isinstance(markets, Sequence) and not isinstance(markets, (str, bytes)):
                for m in markets:
                    if not isinstance(m, Mapping):
                        continue
                    market = str(m.get("market") or m.get("market_type") or "1X2")
                    odds_map = self._extract_odds_map(m)
                    if not odds_map:
                        continue
                    rows.append(
                        {
                            "match_id": match_id,
                            "timestamp": ts,
                            "market": market,
                            "odds": odds_map,
                            **meta,
                        }
                    )
            else:
                odds_map = self._extract_odds_map(bk)
                if odds_map:
                    rows.append(
                        {
                            "match_id": match_id,
                            "timestamp": ts,
                            "market": str(bk.get("market") or "1X2"),
                            "odds": odds_map,
                            **meta,
                        }
                    )
        return rows

    def _rows_from_markets(
        self,
        match_id: str,
        markets: Any,
        *,
        default_ts: datetime,
        default_book: Any,
    ) -> list[dict[str, Any]]:
        if markets is None:
            return []
        meta = self._annotate_book(str(default_book) if default_book is not None else None)
        rows: list[dict[str, Any]] = []
        if isinstance(markets, Mapping):
            iterable: Iterable[tuple[str, Any]] = markets.items()
            for mkt_name, mkt_odds in iterable:
                odds_map = self._extract_odds_map(
                    mkt_odds if isinstance(mkt_odds, Mapping) else {"odds": mkt_odds}
                )
                if not odds_map:
                    continue
                rows.append(
                    {
                        "match_id": match_id,
                        "timestamp": default_ts,
                        "market": str(mkt_name),
                        "odds": odds_map,
                        **meta,
                    }
                )
            return rows
        if isinstance(markets, Sequence) and not isinstance(markets, (str, bytes)):
            for m in markets:
                if not isinstance(m, Mapping):
                    continue
                ts = _parse_timestamp(m.get("timestamp") or default_ts)
                book = m.get("bookmaker") or default_book
                row_meta = self._annotate_book(str(book) if book is not None else None)
                market = str(m.get("market") or m.get("market_type") or "1X2")
                odds_map = self._extract_odds_map(m)
                if not odds_map:
                    continue
                rows.append(
                    {
                        "match_id": match_id,
                        "timestamp": ts,
                        "market": market,
                        "odds": odds_map,
                        **row_meta,
                    }
                )
        return rows

    def _rows_from_flat_market(
        self,
        match_id: str,
        payload: Mapping[str, Any],
        *,
        default_ts: datetime,
        default_book: Any,
    ) -> list[dict[str, Any]]:
        odds_map = self._extract_odds_map(payload)
        if not odds_map:
            return []
        meta = self._annotate_book(str(default_book) if default_book is not None else None)
        market = str(payload.get("market") or payload.get("market_type") or "1X2")
        return [
            {
                "match_id": match_id,
                "timestamp": default_ts,
                "market": market,
                "odds": odds_map,
                **meta,
            }
        ]

    @staticmethod
    def _extract_odds_map(item: Mapping[str, Any] | Any) -> dict[str, float]:
        if not isinstance(item, Mapping):
            return {}
        # Nested "odds" dict preferred.
        nested = item.get("odds")
        raw: MutableMapping[str, Any]
        if isinstance(nested, Mapping):
            raw = dict(nested)
        else:
            raw = dict(item)

        aliases = {
            "H": ("H", "home", "odds_home", "1", "home_odds"),
            "D": ("D", "draw", "odds_draw", "X", "draw_odds"),
            "A": ("A", "away", "odds_away", "2", "away_odds"),
            "over": ("over", "odds_over", "O", "Over"),
            "under": ("under", "odds_under", "U", "Under"),
            "ah_home": ("ah_home", "odds_ah_home", "odds_home_handicap", "AHH"),
            "ah_away": ("ah_away", "odds_ah_away", "odds_away_handicap", "AHA"),
        }
        out: dict[str, float] = {}
        used_keys: set[str] = set()
        for canon, keys in aliases.items():
            for k in keys:
                if k in raw and k not in used_keys:
                    v = _as_float(raw[k])
                    if v is not None:
                        out[canon] = v
                        used_keys.add(k)
                        break
        # Pass through any other numeric odds-like keys (selection → price).
        skip = {
            "match_id",
            "canonical_match_id",
            "fixture_id",
            "timestamp",
            "observed_at",
            "as_of",
            "bookmaker",
            "name",
            "source",
            "market",
            "market_type",
            "line",
            "handicap",
            "markets",
            "bookmakers",
            "odds",
            "is_sharp",
            "is_soft",
        }
        for k, v in raw.items():
            if k in skip or k in used_keys or k in out:
                continue
            price = _as_float(v)
            if price is not None:
                out[str(k)] = price
        return out
