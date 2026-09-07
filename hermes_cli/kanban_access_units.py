"""Access-unit registry: stable outcome keys for access-workflow cards.

Implements the accepted non-live Kanban control-plane design (Option A):

* a **stable access outcome key** — ``profile|provider|target|access_class``
  — identifies one intended access result across retries, correction cycles
  and re-reviews, so the board can enforce *one non-terminal unit per
  outcome* instead of deduplicating by prose;
* review/continuation key builders on the same registry, giving **atomic
  creation** with one active card per key under concurrency;
* a **semantic recurrence fingerprint** over
  outcome/unit/actor/action/provider/reason, so login -> consent -> masked
  entry reads as forward progress (distinct fingerprints) while a true
  repeat (same fingerprint) is recognised as a loop;
* **narrow PR collision resources**: repository-scoped file/service/schema/
  secret/profile declarations that only collide on genuine overlap;
* **idempotent successor reconciliation** that releases only the verified
  task's explicitly owned keys, never archives independent unfinished work,
  and NEVER deletes registry rows or events (append-only history).

Everything here is inert unless the owning surface passes an explicitly
enabled flag set (see :func:`flag_enabled`); default-off is a hard contract.
The module depends only on stdlib + the kanban write transaction so the
kernel, tools and the dashboard plugin can adopt it without new coupling.
"""
from __future__ import annotations

import sqlite3
import time
from typing import Any, Mapping, Optional

# Classes of registered access units. ``outcome`` is the primary lane;
# review and continuation cards use the same registry machinery with their
# own key shapes (build_review_key / build_continuation_key).
ACCESS_UNIT_CLASSES = ("outcome", "review", "continuation")

# Registry DDL. Executed by kanban_db_connect's migration pass on every
# board — additive, IF NOT EXISTS, a no-op on fresh DBs which already carry
# it from SCHEMA_SQL. The partial unique index is what makes concurrent
# duplicate creation impossible: at most one non-terminal row per key.
ACCESS_UNITS_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS access_units (
    key            TEXT NOT NULL,
    unit_class     TEXT NOT NULL,
    task_id        TEXT NOT NULL,
    created_by     TEXT,
    created_at     INTEGER NOT NULL,
    released_at    INTEGER,
    release_reason TEXT
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_access_units_one_active
    ON access_units(key) WHERE released_at IS NULL;
CREATE INDEX IF NOT EXISTS idx_access_units_task ON access_units(task_id);
"""

# Task statuses that make a registered unit's key releasable. Kept in one
# place so wrappers and reconciliation agree.
TERMINAL_TASK_STATUSES = frozenset({"done", "archived"})

# Collision-resource prefixes accepted by parse_collision_resources.
_RESOURCE_PREFIXES = ("repo", "file", "service", "schema", "secret", "profile")
_MAX_COLLISION_DECLARATIONS = 64
_MAX_COLLISION_DECLARATION_CHARS = 512


class AccessUnitConflict(ValueError):
    """Another non-terminal unit already owns the key.

    ``existing_task_id`` lets the caller converge (observe/extend the
    winner) instead of creating a duplicate lane.
    """

    def __init__(self, key: str, unit_class: str, existing_task_id: str):
        self.key = key
        self.unit_class = unit_class
        self.existing_task_id = existing_task_id
        super().__init__(
            f"active {unit_class} unit already exists for access outcome "
            f"{key!r}: {existing_task_id}"
        )


# ---------------------------------------------------------------------------
# Key builders
# ---------------------------------------------------------------------------

def _norm_field(field: str) -> str:
    text = str(field).strip().casefold()
    if not text or "|" in text:
        raise ValueError("access key fields must be non-empty and pipe-free")
    return text


def canonical_outcome_key(key: str) -> str:
    """Canonicalise an already-built outcome key supplied by a caller.

    The model-facing surfaces pass outcome keys as opaque strings; this is
    the domain boundary that makes `` Ruth | GA4 | p/1 | read_only `` and
    ``ruth|ga4|p/1|read_only`` the SAME key before they reach the registry,
    so casing/spacing variants cannot fork (or bypass) uniqueness. A string
    that is not four non-empty pipe-free fields is rejected loudly — a
    malformed key is a caller bug, not a key to store.
    """
    text = str(key).strip()
    parts = text.split("|")
    if len(parts) != 4:
        raise ValueError(
            "access_outcome_key must be 'profile|provider|target|access_class' "
            "(four non-empty pipe-free fields)"
        )
    return "|".join(_norm_field(part) for part in parts)


def build_outcome_key(
    profile: str, provider: str, target: str, access_class: str,
) -> str:
    """Canonical stable outcome key: ``profile|provider|target|access_class``.

    Fields are normalised (strip + case-fold) so display casing and padding
    never fork the key. Pipes are rejected outright: escaping was judged
    cleverer than the data deserves — a provider id containing ``|`` is a
    bug in the caller, not a key to smuggle through.
    """
    return "|".join((
        _norm_field(profile), _norm_field(provider),
        _norm_field(target), _norm_field(access_class),
    ))


def build_review_key(artifact_digest: str, rubric_digest: str, review_class: str) -> str:
    """Atomic review key: tuple of (artifact digest, rubric_digest, class).

    Digests are digests, not names: they keep their case (they are already
    canonical identifiers) but must be non-empty. A different artifact
    digest yields a different key — a re-review of a new head is a new
    review, while the same tuple twice is a duplicate watchdog the registry
    rejects.
    """
    def _digest_field(value: str) -> str:
        text = str(value).strip()
        if not text or "|" in text:
            raise ValueError("review key digests must be non-empty and pipe-free")
        return text

    return "review|" + "|".join((
        _digest_field(artifact_digest), _digest_field(rubric_digest),
        _norm_field(review_class),
    ))


def build_continuation_key(outcome_key: str, unit_digest: str, predecessor_id: str) -> str:
    """Atomic continuation key: outcome key + unit digest + terminal predecessor.

    The outcome key carries embedded field pipes, so it is validated and
    canonicalised as a full outcome key before composition.
    """
    return "cont|" + "|".join((
        canonical_outcome_key(outcome_key),
        _norm_field(unit_digest), _norm_field(predecessor_id),
    ))


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def _now() -> int:
    return int(time.time())


def _active_row(conn: sqlite3.Connection, key: str) -> Optional[sqlite3.Row]:
    """The unreleased registry row for ``key`` whose owner is NON-terminal.

    The uniqueness invariant is "one NON-TERMINAL unit per key", not "one
    row ever": a row whose owning task is ``done``/``archived`` is not an
    active unit even before it has been explicitly released (a flag may be
    disabled, a process may have died mid-completion). Checking the owner's
    live status in the same query — instead of trusting ``released_at`` —
    makes the invariant hold independently of which combination of the six
    default-off flags is enabled. Registry rows themselves are never
    deleted here; read-only stale detection.
    """
    return conn.execute(
        "SELECT u.task_id FROM access_units u "
        "JOIN tasks t ON t.id = u.task_id "
        "WHERE u.key = ? AND u.released_at IS NULL "
        "AND t.status NOT IN ('done', 'archived')",
        (key,),
    ).fetchone()


def register_access_unit(
    conn: sqlite3.Connection, *, key: str, unit_class: str, task_id: str,
    created_by: Optional[str] = None,
) -> None:
    """Register ``task_id`` as the one active unit for ``key``.

    Raises :class:`AccessUnitConflict` when another non-terminal unit
    already owns the key — the caller should converge on
    ``existing_task_id`` rather than create a duplicate lane. Registering
    the task that already owns the key is an idempotent no-op.

    The write happens in its own (nestable) ``write_txn`` so it composes
    with task creation: a card that was never registered cannot silently
    own an outcome key, and a rolled-back creation leaves no registry row.
    """
    if unit_class not in ACCESS_UNIT_CLASSES:
        raise ValueError(f"unit_class must be one of {ACCESS_UNIT_CLASSES}")
    from hermes_cli.kanban_db_connect import write_txn

    with write_txn(conn, allow_nested=True):
        active = _active_row(conn, key)
        if active is not None and active["task_id"] != task_id:
            raise AccessUnitConflict(key, unit_class, active["task_id"])
        if active is not None:
            return  # idempotent re-register of the current owner
        # The key may still be held by a TERMINAL owner (completed while a
        # release flag was off, or a crash before the release landed). The
        # invariant is "one non-terminal unit": release the stale row —
        # preserving it, never deleting — and let the new owner register.
        conn.execute(
            "UPDATE access_units SET released_at = ?, release_reason = ? "
            "WHERE key = ? AND released_at IS NULL",
            (_now(), "owner_terminal", key),
        )
        conn.execute(
            "INSERT INTO access_units (key, unit_class, task_id, created_by, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (key, unit_class, task_id, created_by, _now()),
        )


def active_access_unit(conn: sqlite3.Connection, key: str) -> Optional[str]:
    """The task id currently owning ``key``, or ``None``."""
    row = _active_row(conn, key)
    return row["task_id"] if row is not None else None


def release_access_unit(
    conn: sqlite3.Connection, key: str, *, reason: str = "completed",
) -> bool:
    """Release the key so a successor unit can register.

    Idempotent — releasing an already-released key returns ``False``. The
    registry row is never deleted: which task owned which outcome stays
    queryable for audit for the life of the board.
    """
    from hermes_cli.kanban_db_connect import write_txn

    with write_txn(conn, allow_nested=True):
        cur = conn.execute(
            "UPDATE access_units SET released_at = ?, release_reason = ? "
            "WHERE key = ? AND released_at IS NULL",
            (_now(), reason, key),
        )
        return cur.rowcount == 1


# ---------------------------------------------------------------------------
# Semantic recurrence
# ---------------------------------------------------------------------------

_RECURRENCE_FIELDS = ("outcome", "unit", "actor", "action", "provider", "reason")


def recurrence_fingerprint(
    *, outcome: str, unit: str, actor: str, action: str, provider: str, reason: str,
) -> str:
    """Stable fingerprint of a block/wait episode for loop detection.

    Two episodes with the same fingerprint are a TRUE repeat (increment
    recurrence); any field change — most importantly ``action``, so
    provider login -> consent -> masked entry is three distinct
    fingerprints — is forward progress, not a loop. ``outcome`` is usually
    a full outcome key (pipes allowed): fields are compared case-blind,
    not pipe-sanitised.
    """
    return "\x1f".join(
        str(value).strip().casefold()
        for value in (outcome, unit, actor, action, provider, reason)
        if str(value).strip()
    )


# ---------------------------------------------------------------------------
# Narrow PR collision resources
# ---------------------------------------------------------------------------

class CollisionResources:
    """Parsed ``prefix:value`` collision declarations for one task.

    ``files`` is a set of ``(repo, path)`` with the repo normalised; the
    other classes keep their canonical declared token. File declarations
    do NOT imply a repository-wide declaration: disjoint files in one repo
    remain independent. See :func:`resources_overlap`.
    """

    __slots__ = ("repositories", "files", "services", "schemas", "secrets", "profiles")

    def __init__(self) -> None:
        self.repositories: set[str] = set()
        self.files: set[tuple[str, str]] = set()
        self.services: set[str] = set()
        self.schemas: set[str] = set()
        self.secrets: set[str] = set()
        self.profiles: set[str] = set()


def parse_collision_resources(decls: Optional[IterableStr]) -> "CollisionResources":
    """Parse ``prefix:value`` declarations into a :class:`CollisionResources`.

    Unknown prefixes raise: a typo'd declaration must fail loudly rather
    than silently narrowing the guard to nothing. ``file:`` values carry an
    extra ``repo:path`` segment.
    """
    res = CollisionResources()
    if isinstance(decls, (str, bytes)):
        raise ValueError("collision resources must be an iterable of declarations, not a string")
    items = list(decls or ())
    if len(items) > _MAX_COLLISION_DECLARATIONS:
        raise ValueError(
            f"at most {_MAX_COLLISION_DECLARATIONS} collision resources may be declared"
        )
    for decl in items:
        text = str(decl).strip()
        if len(text) > _MAX_COLLISION_DECLARATION_CHARS:
            raise ValueError(
                "collision resource declarations must be at most "
                f"{_MAX_COLLISION_DECLARATION_CHARS} characters"
            )
        if not text:
            continue
        prefix, sep, value = text.partition(":")
        if not sep or prefix not in _RESOURCE_PREFIXES:
            raise ValueError(
                f"collision resource must be one of {list(_RESOURCE_PREFIXES)}: {decl!r}"
            )
        value = value.strip()
        if not value:
            raise ValueError(f"collision resource value must be non-empty: {decl!r}")
        if prefix == "repo":
            res.repositories.add(value.casefold())
        elif prefix == "file":
            repo, fsep, path = value.partition(":")
            if not fsep or not repo.strip() or not path.strip():
                raise ValueError(f"file collision resource must be 'file:owner/repo:path': {decl!r}")
            res.files.add((repo.strip().casefold(), path.strip()))
        elif prefix == "service":
            res.services.add(value)
        elif prefix == "schema":
            res.schemas.add(value)
        elif prefix == "secret":
            res.secrets.add(value)
        elif prefix == "profile":
            res.profiles.add(value)
    return res


def collision_resources_payload(resources: CollisionResources) -> dict[str, list[str]]:
    """Canonical durable payload for an ``access_collision_resources`` event."""
    return {
        "repositories": sorted(resources.repositories),
        "files": [f"{repo}:{path}" for repo, path in sorted(resources.files)],
        "services": sorted(resources.services),
        "schemas": sorted(resources.schemas),
        "secrets": sorted(resources.secrets),
        "profiles": sorted(resources.profiles),
    }


def resources_overlap(a: CollisionResources, b: CollisionResources) -> bool:
    """True iff the two declared resource sets genuinely collide.

    Granularity is deliberate: identical files collide, while disjoint files
    in one repository do not. An EXPLICIT ``repo:`` declaration is broad and
    overlaps every explicit repo or file declaration in that repository.
    Service/schema/secret/profile declarations overlap only on their exact
    canonical token. Unrelated resources never guard each other.
    """
    a_file_repos = {repo for repo, _path in a.files}
    b_file_repos = {repo for repo, _path in b.files}
    if (
        a.repositories & b.repositories
        or a.repositories & b_file_repos
        or b.repositories & a_file_repos
        or a.files & b.files
    ):
        return True
    return bool(
        a.services & b.services
        or a.schemas & b.schemas
        or a.secrets & b.secrets
        or a.profiles & b.profiles
    )


# ---------------------------------------------------------------------------
# Successor reconciliation
# ---------------------------------------------------------------------------

def reconcile_successor(
    conn: sqlite3.Connection, successor_task_id: str,
    *, reason: str = "successor_verified",
) -> dict[str, Any]:
    """Release the verified unit's OWN keys after its task completes.

    Scoped to explicit provenance — exactly the registry rows owned by
    ``successor_task_id`` — and release-only by design:

    * keys whose task is terminal (done/archived) are released, preserving
      the row; registry rows and events are NEVER deleted;
    * INDEPENDENT active outcomes (different keys, different tasks) are
      never archived or released — completing a GA4 unit must not close an
      unrelated unfinished QuickFile card. An unfinished duplicate lane for
      the SAME key cannot exist while the partial unique index plus the
      non-terminal-JOIN hold ("one non-terminal unit per key"), so there is
      nothing to archive.

    Safe to re-run. Returns ``{"released": [keys...], "archived": []}``
    (``archived`` is kept for caller compatibility and is always empty).
    """
    from hermes_cli.kanban_db_connect import write_txn

    released: list[str] = []
    # allow_nested: complete_task runs this INSIDE its completion txn so the
    # release lands atomically with the completion itself (all-or-nothing);
    # standalone calls get the same semantics.
    with write_txn(conn, allow_nested=True):
        rows = conn.execute(
            "SELECT key FROM access_units WHERE task_id = ? AND released_at IS NULL",
            (successor_task_id,),
        ).fetchall()
        for row in rows:
            _release(conn, row["key"], reason)
            released.append(row["key"])
    return {"released": released, "archived": []}


def _release(conn: sqlite3.Connection, key: str, reason: str) -> None:
    conn.execute(
        "UPDATE access_units SET released_at = ?, release_reason = ? "
        "WHERE key = ? AND released_at IS NULL",
        (_now(), reason, key),
    )


# ---------------------------------------------------------------------------
# Flag gating
# ---------------------------------------------------------------------------

def flag_enabled(flags: Optional[Mapping[str, Any]], name: str) -> bool:
    """True iff ``name`` is explicitly truthy in the flag mapping.

    Flags default OFF: ``None``, a missing key, ``False``, ``0`` and every
    other falsy value are disabled, so an unconfigured install keeps
    byte-identical behaviour.
    """
    return bool(flags.get(name)) if flags else False


# Typing alias kept at the bottom to avoid widening the import surface.
IterableStr = Any
