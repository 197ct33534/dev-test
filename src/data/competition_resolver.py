"""Competition auto-resolver: alias lookup → fuzzy match → staging queue.

Additive module that complements :class:`~src.data.entity_resolver.EntityResolver`.
Uses ``difflib.SequenceMatcher`` (stdlib) — no rapidfuzz dependency.
"""

from __future__ import annotations

import logging
import uuid
from difflib import SequenceMatcher
from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.data.entity_resolver import MAPPED, EntityResolver
from src.db.schema_v2 import (
    CanonicalCompetition,
    CompetitionAlias,
    StagingUnmappedCompetition,
)

logger = logging.getLogger(__name__)

# Auto-map when SequenceMatcher.ratio() is at or above this threshold.
SIMILARITY_THRESHOLD = 0.88
PENDING_REVIEW = "PENDING_REVIEW"


def similarity_ratio(a: str, b: str) -> float:
    """Return ``SequenceMatcher`` ratio in ``[0, 1]`` (case-insensitive)."""
    left = (a or "").strip().lower()
    right = (b or "").strip().lower()
    if not left or not right:
        return 0.0
    return SequenceMatcher(None, left, right).ratio()


class CompetitionResolver:
    """Resolve raw competition names to ``canonical_competitions.id``.

    Lookup order
    ------------
    1. ``competition_aliases`` via :class:`EntityResolver` helpers.
    2. Fuzzy match against ``canonical_competitions.name`` / ``code``.
    3. If best score ``>=`` :data:`SIMILARITY_THRESHOLD`, auto-create a
       ``MAPPED`` alias and return the canonical id.
    4. Otherwise insert ``staging_unmapped_competitions`` with
       ``PENDING_REVIEW`` and return ``None``.
    """

    def __init__(
        self,
        session: Optional[AsyncSession] = None,
        *,
        entity_resolver: Optional[EntityResolver] = None,
        similarity_threshold: float = SIMILARITY_THRESHOLD,
    ) -> None:
        self.session = session
        self.entity_resolver = entity_resolver or EntityResolver(session)
        self.similarity_threshold = float(similarity_threshold)

    async def resolve_competition(
        self,
        session: AsyncSession,
        source_name: str,
        raw_comp_name: str,
    ) -> Optional[uuid.UUID]:
        """Resolve ``raw_comp_name`` from source ``source_name`` to a UUID.

        Parameters
        ----------
        session:
            Async SQLAlchemy session.
        source_name:
            Provenance key (e.g. ``football-data``) — stored as
            ``CompetitionAlias.source_type``.
        raw_comp_name:
            Competition label as seen at the source (e.g. ``Premier League``
            or division code ``E0``).

        Returns
        -------
        Optional[uuid.UUID]
            Canonical competition id, or ``None`` when staged for review.
        """
        if not source_name or not raw_comp_name or not str(raw_comp_name).strip():
            raise ValueError("source_name and raw_comp_name are required")

        source_type = source_name.strip()
        raw = str(raw_comp_name).strip()

        existing = await self.entity_resolver._find_competition_alias(
            session,
            source_type=source_type,
            source_name=raw,
            source_competition_id=None,
        )
        if existing is not None:
            return existing.canonical_competition_id

        # Also try alias by source_competition_id == raw (FD codes like E0).
        existing_by_id = await self.entity_resolver._find_competition_alias(
            session,
            source_type=source_type,
            source_name=raw,
            source_competition_id=raw,
        )
        if existing_by_id is not None:
            return existing_by_id.canonical_competition_id

        best = await self._best_canonical_match(session, raw)
        if best is not None:
            canonical_id, match_name, score = best
            if score >= self.similarity_threshold:
                await self._create_mapped_alias(
                    session,
                    canonical_id=canonical_id,
                    source_type=source_type,
                    raw_name=raw,
                    score=score,
                    match_name=match_name,
                )
                return canonical_id

            await self._stage_unmapped(
                session,
                source_type=source_type,
                raw_name=raw,
                best_match_name=match_name,
                best_match_score=score,
            )
            return None

        await self._stage_unmapped(
            session,
            source_type=source_type,
            raw_name=raw,
            best_match_name=None,
            best_match_score=None,
        )
        return None

    async def _best_canonical_match(
        self,
        session: AsyncSession,
        raw: str,
    ) -> Optional[tuple[uuid.UUID, str, float]]:
        """Return ``(canonical_id, display_name, score)`` for the best fuzzy hit."""
        rows = (
            await session.execute(select(CanonicalCompetition))
        ).scalars().all()
        if not rows:
            return None

        best_id: Optional[uuid.UUID] = None
        best_name = ""
        best_score = -1.0
        for row in rows:
            candidates = [row.name]
            if row.code:
                candidates.append(row.code)
            for cand in candidates:
                score = similarity_ratio(raw, cand)
                if score > best_score:
                    best_score = score
                    best_id = row.id
                    best_name = row.name

        if best_id is None:
            return None
        return best_id, best_name, best_score

    async def _create_mapped_alias(
        self,
        session: AsyncSession,
        *,
        canonical_id: uuid.UUID,
        source_type: str,
        raw_name: str,
        score: float,
        match_name: str,
    ) -> CompetitionAlias:
        alias = CompetitionAlias(
            canonical_competition_id=canonical_id,
            source_type=source_type,
            source_competition_id=raw_name if len(raw_name) <= 32 else None,
            source_name=raw_name,
            mapping_status=MAPPED,
        )
        session.add(alias)
        await session.flush()
        logger.info(
            "Auto-mapped competition %r → %s (%s) score=%.3f",
            raw_name,
            match_name,
            canonical_id,
            score,
        )
        return alias

    async def _stage_unmapped(
        self,
        session: AsyncSession,
        *,
        source_type: str,
        raw_name: str,
        best_match_name: Optional[str],
        best_match_score: Optional[float],
    ) -> StagingUnmappedCompetition:
        existing = (
            await session.execute(
                select(StagingUnmappedCompetition).where(
                    StagingUnmappedCompetition.source_type == source_type,
                    StagingUnmappedCompetition.raw_name == raw_name,
                )
            )
        ).scalar_one_or_none()
        if existing is not None:
            existing.best_match_name = best_match_name
            existing.best_match_score = best_match_score
            if existing.status != PENDING_REVIEW:
                existing.status = PENDING_REVIEW
            await session.flush()
            logger.warning(
                "Unmapped competition %r already staged (id=%s, best=%.3f)",
                raw_name,
                existing.id,
                best_match_score if best_match_score is not None else -1.0,
            )
            return existing

        row = StagingUnmappedCompetition(
            source_type=source_type,
            raw_name=raw_name,
            best_match_name=best_match_name,
            best_match_score=best_match_score,
            status=PENDING_REVIEW,
            meta_json={"created_via": "competition_resolver"},
        )
        session.add(row)
        await session.flush()
        logger.warning(
            "PENDING_REVIEW competition %r (source=%s, best=%r score=%s) staged as %s",
            raw_name,
            source_type,
            best_match_name,
            f"{best_match_score:.3f}" if best_match_score is not None else "n/a",
            row.id,
        )
        return row


async def resolve_competition(
    session: AsyncSession,
    source_name: str,
    raw_comp_name: str,
    *,
    similarity_threshold: float = SIMILARITY_THRESHOLD,
) -> Optional[uuid.UUID]:
    """Module-level helper wrapping :meth:`CompetitionResolver.resolve_competition`."""
    resolver = CompetitionResolver(similarity_threshold=similarity_threshold)
    return await resolver.resolve_competition(session, source_name, raw_comp_name)
