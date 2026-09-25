"""Feature lineage mapping from engineered features to raw observations."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional


def _parse_dt(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        text = value.replace("Z", "+00:00")
        try:
            return datetime.fromisoformat(text)
        except ValueError:
            return None
    return None


def _to_iso(value: Optional[datetime]) -> Optional[str]:
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc).isoformat()
    return value.isoformat()


class FeatureLineageTracker:
    """Build a lineage map: feature_key → raw source ids + max observation time.

    ``raw_sources_map`` shape
    -------------------------
    Keys are feature names. Values are either:

    - ``list[dict]`` with ``raw_id`` / ``id`` and ``observed_at``, or
    - ``list[str]`` of raw ids (then ``observed_at_max`` is None unless
      a parallel ``observed_at`` list is provided via dict form).

    Output per feature
    ------------------
    ::

        {
          "raw_ids": ["...", "..."],
          "observed_at_max": "2025-09-19T12:00:00+00:00" | None,
        }

    Only keys present in ``feature_dict`` are emitted (stable sort by key).
    Unknown feature keys in ``raw_sources_map`` are ignored.
    """

    def build_lineage_map(
        self,
        feature_dict: dict[str, Any],
        raw_sources_map: dict[str, Any],
    ) -> dict[str, dict[str, Any]]:
        """Map each feature to its contributing raw ids and max ``observed_at``.

        Parameters
        ----------
        feature_dict:
            Engineered features ``{feature_key: value}``.
        raw_sources_map:
            ``{feature_key: list[raw_id | {raw_id, observed_at}]}``.

        Returns
        -------
        dict
            ``{feature_key: {"raw_ids": [...], "observed_at_max": iso|None}}``.
        """
        lineage: dict[str, dict[str, Any]] = {}
        for feature_key in sorted(feature_dict.keys()):
            sources = raw_sources_map.get(feature_key, [])
            raw_ids, observed_at_max = self._collect_sources(sources)
            lineage[feature_key] = {
                "raw_ids": raw_ids,
                "observed_at_max": _to_iso(observed_at_max),
            }
        return lineage

    @staticmethod
    def _collect_sources(
        sources: Any,
    ) -> tuple[list[str], Optional[datetime]]:
        raw_ids: list[str] = []
        max_obs: Optional[datetime] = None

        if sources is None:
            return raw_ids, max_obs

        if not isinstance(sources, (list, tuple)):
            sources = [sources]

        for item in sources:
            if isinstance(item, dict):
                rid = item.get("raw_id", item.get("id"))
                if rid is not None:
                    raw_ids.append(str(rid))
                obs = _parse_dt(item.get("observed_at"))
                if obs is not None and (max_obs is None or obs > max_obs):
                    max_obs = obs
            else:
                raw_ids.append(str(item))

        return raw_ids, max_obs
