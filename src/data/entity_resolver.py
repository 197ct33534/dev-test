"""Entity resolution: source aliases → canonical teams / competitions."""

from __future__ import annotations

import logging
import re
import unicodedata
import uuid
from typing import Optional, Union

from sqlalchemy import Select, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.schema_v2 import (
    CanonicalCompetition,
    CanonicalTeam,
    CompetitionAlias,
    TeamAlias,
)

logger = logging.getLogger(__name__)

UNMAPPED = "UNMAPPED"
MAPPED = "MAPPED"

# Generic club-form tokens dropped when comparing team names.
_TEAM_NOISE_TOKENS: frozenset[str] = frozenset({"fc", "cf", "sc"})
_NON_ALNUM_RE = re.compile(r"[^0-9a-z]+")


def normalize_team_key(name: Optional[str]) -> str:
    """Comparison key for team names.

    Casefolds, strips accents, replaces punctuation with spaces and drops the
    generic ``FC`` / ``CF`` / ``SC`` tokens, e.g. ``"Atlético Madrid CF"`` and
    ``"atletico-madrid"`` both map to ``"atletico madrid"``. If dropping the
    tokens would leave nothing (a club literally named ``"FC"``), they are kept.
    """
    if not name:
        return ""
    text = unicodedata.normalize("NFKD", str(name))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.casefold().replace("&", " and ")
    tokens = [t for t in _NON_ALNUM_RE.split(text) if t]
    kept = [t for t in tokens if t not in _TEAM_NOISE_TOKENS]
    return " ".join(kept or tokens)

try:
    from src.monitoring.data_monitor import DataMonitor, default_monitor
except ImportError:  # pragma: no cover - monitor is optional for Worker-1 scope
    DataMonitor = None  # type: ignore[misc, assignment]
    default_monitor = None  # type: ignore[assignment]


class EntityResolver:
    """Resolve source team/competition identifiers to canonical UUIDs.

    Lookup order: ``*_aliases`` by ``(source_type, source_*_id)`` then
    ``(source_type, source_name)``. On miss, create a canonical row labeled
    ``UNMAPPED``, attach an alias, and emit a warning log.

    Preferred::

        await resolver.get_or_create_canonical_team(
            session, source_name, source_type, source_team_id
        )

    Back-compat (session bound in ``__init__``)::

        EntityResolver(session).get_or_create_canonical_team(
            source_name, source_type, source_team_id=...
        )
    """

    def __init__(
        self,
        session: Optional[AsyncSession] = None,
        monitor: Optional[object] = None,
    ) -> None:
        self.session = session
        if monitor is not None:
            self.monitor = monitor
        elif default_monitor is not None:
            self.monitor = default_monitor
        else:
            self.monitor = None

    async def get_or_create_canonical_team(
        self,
        session: Union[AsyncSession, str],
        source_name: Optional[str] = None,
        source_type: Optional[str] = None,
        source_team_id: Optional[str] = None,
    ) -> uuid.UUID:
        """Return ``canonical_teams.id`` for a source team name/id.

        Parameters
        ----------
        session:
            Async SQLAlchemy session, **or** (legacy) the ``source_name`` string
            when the session was bound in ``__init__``.
        source_name:
            Display name as seen at the source (or legacy ``source_type``).
        source_type:
            Provenance key (e.g. ``football-data``, ``flashscore``).
        source_team_id:
            Optional stable id from the source feed.
        """
        sess, name, stype, sid = self._unpack_call(
            session, source_name, source_type, source_team_id
        )

        existing = await self._find_team_alias(
            sess,
            source_type=stype,
            source_name=name,
            source_team_id=sid,
        )
        if existing is not None:
            return existing.canonical_team_id

        matched = await self._find_team_by_normalized_name(sess, name)
        if matched is not None:
            canonical_id, status = matched
            sess.add(
                TeamAlias(
                    canonical_team_id=canonical_id,
                    source_type=stype,
                    source_team_id=sid,
                    source_name=name.strip(),
                    mapping_status=status or UNMAPPED,
                )
            )
            await sess.flush()
            logger.info(
                "Team %s/%r matched existing canonical %s by normalized name",
                stype,
                name,
                canonical_id,
            )
            return canonical_id

        team = CanonicalTeam(
            name=name.strip(),
            mapping_status=UNMAPPED,
            meta_json={
                "created_via": "entity_resolver",
                "source_type": stype,
                "source_team_id": sid,
            },
        )
        sess.add(team)
        await sess.flush()

        alias = TeamAlias(
            canonical_team_id=team.id,
            source_type=stype,
            source_team_id=sid,
            source_name=name.strip(),
            mapping_status=UNMAPPED,
        )
        sess.add(alias)
        await sess.flush()

        self._warn_unmapped_team(
            source_name=name,
            source_type=stype,
            source_team_id=sid,
            canonical_team_id=team.id,
        )
        return team.id

    async def get_or_create_canonical_competition(
        self,
        session: Union[AsyncSession, str],
        source_name: Optional[str] = None,
        source_type: Optional[str] = None,
        source_comp_id: Optional[str] = None,
        *,
        source_competition_id: Optional[str] = None,
    ) -> uuid.UUID:
        """Return ``canonical_competitions.id`` for a source competition name/id.

        ``source_comp_id`` is preferred; ``source_competition_id`` is a
        keyword alias for back-compat.
        """
        explicit_id = (
            source_comp_id if source_comp_id is not None else source_competition_id
        )
        sess, name, stype, cid = self._unpack_call(
            session, source_name, source_type, explicit_id
        )

        existing = await self._find_competition_alias(
            sess,
            source_type=stype,
            source_name=name,
            source_competition_id=cid,
        )
        if existing is not None:
            return existing.canonical_competition_id

        competition = CanonicalCompetition(
            name=name.strip(),
            code=cid,
            mapping_status=UNMAPPED,
            meta_json={
                "created_via": "entity_resolver",
                "source_type": stype,
                "source_competition_id": cid,
            },
        )
        sess.add(competition)
        await sess.flush()

        alias = CompetitionAlias(
            canonical_competition_id=competition.id,
            source_type=stype,
            source_competition_id=cid,
            source_name=name.strip(),
            mapping_status=UNMAPPED,
        )
        sess.add(alias)
        await sess.flush()

        self._warn_unmapped_competition(
            source_name=name,
            source_type=stype,
            source_comp_id=cid,
            canonical_competition_id=competition.id,
        )
        return competition.id

    def _unpack_call(
        self,
        session: Union[AsyncSession, str],
        source_name: Optional[str],
        source_type: Optional[str],
        source_id: Optional[str],
    ) -> tuple[AsyncSession, str, str, Optional[str]]:
        """Normalize preferred vs legacy (session-bound) call shapes."""
        if isinstance(session, AsyncSession):
            if not source_name or not source_type:
                raise TypeError("source_name and source_type are required")
            return session, source_name, source_type, source_id

        # Legacy: first positional arg is source_name; session from __init__.
        if self.session is None:
            raise TypeError(
                "session is required: pass AsyncSession as the first argument "
                "or construct EntityResolver(session)"
            )
        name = session
        stype = source_name
        if not stype:
            raise TypeError("legacy call requires source_name and source_type")
        # Prefer explicit id kw; else 3rd positional was the id.
        sid = source_id if source_id is not None else source_type
        return self.session, name, stype, sid

    def _warn_unmapped_team(
        self,
        *,
        source_name: str,
        source_type: str,
        source_team_id: Optional[str],
        canonical_team_id: uuid.UUID,
    ) -> None:
        msg = f"UNMAPPED team alias created for {source_name!r}"
        if self.monitor is not None and hasattr(self.monitor, "warn"):
            self.monitor.warn(
                msg,
                code="ENTITY_UNMAPPED_TEAM",
                source_type=source_type,
                source_team_id=source_team_id,
                source_name=source_name,
                canonical_team_id=str(canonical_team_id),
            )
        logger.warning(
            "Created UNMAPPED canonical team %s for %s/%s",
            canonical_team_id,
            source_type,
            source_name,
        )

    def _warn_unmapped_competition(
        self,
        *,
        source_name: str,
        source_type: str,
        source_comp_id: Optional[str],
        canonical_competition_id: uuid.UUID,
    ) -> None:
        msg = f"UNMAPPED competition alias created for {source_name!r}"
        if self.monitor is not None and hasattr(self.monitor, "warn"):
            self.monitor.warn(
                msg,
                code="ENTITY_UNMAPPED_COMPETITION",
                source_type=source_type,
                source_competition_id=source_comp_id,
                source_name=source_name,
                canonical_competition_id=str(canonical_competition_id),
            )
        logger.warning(
            "Created UNMAPPED canonical competition %s for %s/%s",
            canonical_competition_id,
            source_type,
            source_name,
        )

    async def _find_team_alias(
        self,
        session: AsyncSession,
        *,
        source_type: str,
        source_name: str,
        source_team_id: Optional[str],
    ) -> Optional[TeamAlias]:
        stmts: list[Select[tuple[TeamAlias]]] = []
        if source_team_id:
            stmts.append(
                select(TeamAlias).where(
                    TeamAlias.source_type == source_type,
                    TeamAlias.source_team_id == source_team_id,
                )
            )
        stmts.append(
            select(TeamAlias).where(
                TeamAlias.source_type == source_type,
                TeamAlias.source_name == source_name.strip(),
            )
        )
        for stmt in stmts:
            result = await session.execute(stmt.limit(1))
            row = result.scalar_one_or_none()
            if row is not None:
                return row
        return None

    async def _find_team_by_normalized_name(
        self,
        session: AsyncSession,
        name: str,
    ) -> Optional[tuple[uuid.UUID, str]]:
        """Unique canonical team whose name/alias normalizes like ``name``.

        Returns ``(canonical_team_id, mapping_status)`` or ``None`` when there is
        no match or the normalized key is ambiguous (several canonical teams).
        """
        key = normalize_team_key(name)
        if not key:
            return None
        hits: dict[uuid.UUID, str] = {}
        teams = (
            await session.execute(
                select(CanonicalTeam.id, CanonicalTeam.name, CanonicalTeam.mapping_status)
            )
        ).all()
        status_by_id: dict[uuid.UUID, str] = {}
        for tid, tname, tstatus in teams:
            status_by_id[tid] = tstatus
            if normalize_team_key(tname) == key:
                hits[tid] = tstatus
        aliases = (
            await session.execute(
                select(TeamAlias.canonical_team_id, TeamAlias.source_name)
            )
        ).all()
        for tid, alias_name in aliases:
            if normalize_team_key(alias_name) == key:
                hits.setdefault(tid, status_by_id.get(tid, UNMAPPED))
        if len(hits) == 1:
            tid, status = next(iter(hits.items()))
            return tid, status
        if len(hits) > 1:
            logger.warning(
                "Ambiguous normalized team name %r (%d canonical teams); "
                "creating a new UNMAPPED team",
                name,
                len(hits),
            )
        return None

    async def _find_competition_alias(
        self,
        session: AsyncSession,
        *,
        source_type: str,
        source_name: str,
        source_competition_id: Optional[str],
    ) -> Optional[CompetitionAlias]:
        stmts: list[Select[tuple[CompetitionAlias]]] = []
        if source_competition_id:
            stmts.append(
                select(CompetitionAlias).where(
                    CompetitionAlias.source_type == source_type,
                    CompetitionAlias.source_competition_id == source_competition_id,
                )
            )
        stmts.append(
            select(CompetitionAlias).where(
                CompetitionAlias.source_type == source_type,
                CompetitionAlias.source_name == source_name.strip(),
            )
        )
        for stmt in stmts:
            result = await session.execute(stmt.limit(1))
            row = result.scalar_one_or_none()
            if row is not None:
                return row
        return None
