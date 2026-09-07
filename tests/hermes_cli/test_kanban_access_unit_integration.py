"""Kernel integration of the access-unit registry behind default-off flags.

Flag surface (all six default False in DEFAULT_CONFIG["kanban"]):

* ``outcome_keys``       — create_task registers the outcome key atomically
                           and duplicate creators converge on the active unit;
* ``review_keys``        — request_review derives the atomic review key from
                           (artifact digest, rubric digest, review class) and
                           a duplicate watchdog observes the existing review
                           card instead of creating another;
* ``continuation_keys``  — create_task with continuation provenance registers
                           the continuation key; one active continuation;
* ``semantic_recurrence``— block_task recurrence is keyed on the semantic
                           fingerprint (outcome/unit/actor/action/provider/
                           reason): login -> consent -> masked entry is
                           progress, not a loop;
* ``narrow_pr_guards``   — the dispatcher's active-PR guard fires only on
                           genuine resource overlap and writes ONE durable
                           wait event;
* ``stale_reconciliation``— complete_task of a registered successor runs the
                           idempotent reconciliation (keys released, stale
                           duplicates archived, history preserved).

With a flag off the corresponding seam keeps its exact pre-change
behaviour — asserted by the disabled-flag tests at the bottom.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_access_units as kau
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd
from hermes_cli.config_defaults import DEFAULT_CONFIG


def _access_flags(**overrides: bool) -> dict[str, bool]:
    flags = {name: False for name in (
        "outcome_keys", "review_keys", "continuation_keys",
        "semantic_recurrence", "narrow_pr_guards", "stale_reconciliation",
    )}
    flags.update(overrides)
    return flags


@pytest.fixture
def kanban_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(home))
    # Board resolution prefers HERMES_KANBAN_DB over HERMES_HOME; scrub it so
    # a dispatcher-inherited pin can never resolve tests to the live board.
    monkeypatch.delenv("HERMES_KANBAN_DB", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_BOARD", raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


@pytest.fixture
def conn(kanban_home: Path):
    with kbc.connect_closing() as c:
        yield c


@pytest.fixture
def flags(monkeypatch: pytest.MonkeyPatch):
    """Inject the flag dict into the kernel seam; tests override entries."""
    state = _access_flags()
    monkeypatch.setattr(kb, "_access_unit_flags", lambda: state)
    return state


# ---------------------------------------------------------------------------
# outcome_keys
# ---------------------------------------------------------------------------

def test_flags_default_off_in_config():
    access = DEFAULT_CONFIG["kanban"]["access_units"]
    assert access == {
        "outcome_keys": False,
        "review_keys": False,
        "continuation_keys": False,
        "semantic_recurrence": False,
        "narrow_pr_guards": False,
        "stale_reconciliation": False,
    }


def test_create_task_registers_outcome_key_when_enabled(conn, flags):
    flags["outcome_keys"] = True
    key = kau.build_outcome_key("ruth", "ga4", "properties/7", "read_only")
    tid = kb.create_task(
        conn, title="GA4 access unit", assignee="vladamir", access_outcome_key=key,
    )
    assert kau.active_access_unit(conn, key) == tid
    # Duplicate creation converges on the SAME task instead of a new lane.
    tid2 = kb.create_task(
        conn, title="GA4 duplicate", assignee="vladamir", access_outcome_key=key,
    )
    assert tid2 == tid


def test_create_task_outcome_conflict_message_names_existing(conn, flags):
    flags["outcome_keys"] = True
    key = kau.build_outcome_key("ruth", "ga4", "properties/7", "read_only")
    first = kb.create_task(conn, title="first", assignee="vladamir", access_outcome_key=key)
    dup = kb.create_task(conn, title="dup", assignee="vladamir")
    with pytest.raises(kau.AccessUnitConflict) as exc:
        kau.register_access_unit(conn, key=key, unit_class="outcome", task_id=dup)
    assert exc.value.existing_task_id == first


def test_outcome_key_flag_off_keeps_legacy_behaviour(conn, flags):
    """Flag off: access_outcome_key is ignored — two tasks, no registry rows."""
    key = kau.build_outcome_key("ruth", "ga4", "properties/7", "read_only")
    a = kb.create_task(conn, title="a", assignee="v", access_outcome_key=key)
    b = kb.create_task(conn, title="b", assignee="v", access_outcome_key=key)
    assert a != b
    assert kau.active_access_unit(conn, key) is None


# ---------------------------------------------------------------------------
# review_keys
# ---------------------------------------------------------------------------

def _request_review_ready(conn, tid, summary="ready for review"):
    """Drive a fresh task into review via the kernel path."""
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (tid,))
    ok = kb.request_review(conn, tid, summary=summary)
    assert ok is True


def test_request_review_records_atomic_review_key(conn, flags):
    flags["review_keys"] = True
    tid = kb.create_task(conn, title="impl", assignee="nadia")
    _request_review_ready(
        conn, tid,
        summary="review of artifact sha256:aaa under rubric sha256:bbb (independent)",
    )
    key = kau.build_review_key("sha256:aaa", "sha256:bbb", "independent")
    assert kau.active_access_unit(conn, key) == tid


def test_duplicate_review_watchdog_observes_existing_card(conn, flags):
    """A duplicate watchdog card for the same review tuple must not create a
    second review lane — it converges on the existing review task."""
    flags["review_keys"] = True
    key = kau.build_review_key("sha256:aaa", "sha256:bbb", "independent")
    existing = kb.create_task(conn, title="independent review", assignee="vladamir")
    kau.register_access_unit(conn, key=key, unit_class="review", task_id=existing)

    dup = kb.create_access_review_task(
        conn, artifact_digest="sha256:aaa", rubric_digest="sha256:bbb",
        review_class="independent", assignee="vladamir",
        title="duplicate watchdog", flags=kb._access_unit_flags(),
    )
    assert dup == existing


def test_duplicate_request_review_converges_and_archives_duplicate(conn, flags):
    """Normal lifecycle path: a second task requesting the same review tuple
    converges on the active review and cannot remain an orphan active card."""
    flags["review_keys"] = True
    descriptor = ("sha256:aaa", "sha256:bbb", "independent")
    first = kb.create_task(conn, title="review first", assignee="nadia")
    duplicate = kb.create_task(conn, title="review duplicate", assignee="nadia")
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET status='ready' WHERE id IN (?, ?)", (first, duplicate))

    ok1, landed1 = kb.request_review(
        conn, first, access_review_of=descriptor, with_reason=True,
    )
    ok2, landed2 = kb.request_review(
        conn, duplicate, access_review_of=descriptor, with_reason=True,
    )
    assert (ok1, landed1) == (True, first)
    assert (ok2, landed2) == (True, first)
    first_task = kb.get_task(conn, first)
    duplicate_task = kb.get_task(conn, duplicate)
    assert first_task is not None and first_task.status == "review"
    assert duplicate_task is not None and duplicate_task.status == "archived"
    active = conn.execute(
        "SELECT COUNT(*) FROM access_units WHERE unit_class='review' AND released_at IS NULL"
    ).fetchone()[0]
    assert active == 1


def test_concurrent_review_creation_converges_without_orphan_active_cards(
    kanban_home, flags,
):
    """Separate connections race the real create_task review descriptor.
    Exactly one card commits; all callers observe it and no orphan survives."""
    import threading

    flags["review_keys"] = True
    descriptor = ("sha256:head", "sha256:rubric", "independent")
    outcomes: list[str] = []
    errors: list[BaseException] = []
    barrier = threading.Barrier(6)

    def create(index: int) -> None:
        try:
            barrier.wait()
            with kbc.connect_closing() as thread_conn:
                outcomes.append(kb.create_task(
                    thread_conn, title=f"review {index}", assignee="vladamir",
                    access_review_of=descriptor,
                ))
        except BaseException as exc:  # test thread must report, not disappear
            errors.append(exc)

    threads = [threading.Thread(target=create, args=(i,)) for i in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)
    assert not any(thread.is_alive() for thread in threads)
    assert errors == []
    assert len(set(outcomes)) == 1
    with kbc.connect_closing() as check:
        active_cards = check.execute(
            "SELECT COUNT(*) FROM tasks WHERE status NOT IN ('done', 'archived')"
        ).fetchone()[0]
        active_reviews = check.execute(
            "SELECT COUNT(*) FROM access_units WHERE unit_class='review' AND released_at IS NULL"
        ).fetchone()[0]
    assert active_cards == 1
    assert active_reviews == 1


def test_new_artifact_digest_permits_re_review(conn, flags):
    flags["review_keys"] = True
    key = kau.build_review_key("sha256:aaa", "sha256:bbb", "independent")
    first = kb.create_task(conn, title="review head-1", assignee="vladamir")
    kau.register_access_unit(conn, key=key, unit_class="review", task_id=first)
    kb.complete_task(conn, first, summary="reviewed")  # releases the key

    second = kb.create_access_review_task(
        conn, artifact_digest="sha256:ccc", rubric_digest="sha256:bbb",
        review_class="independent", assignee="vladamir",
        title="re-review head-2", flags=kb._access_unit_flags(),
    )
    assert second != first
    key2 = kau.build_review_key("sha256:ccc", "sha256:bbb", "independent")
    assert kau.active_access_unit(conn, key2) == second


def test_review_keys_flag_off_no_registration(conn, flags):
    tid = kb.create_task(conn, title="impl", assignee="nadia")
    _request_review_ready(conn, tid)
    rows = conn.execute(
        "SELECT COUNT(*) FROM access_units WHERE unit_class = 'review'").fetchone()[0]
    assert rows == 0


# ---------------------------------------------------------------------------
# continuation_keys
# ---------------------------------------------------------------------------

def test_continuation_registration_and_uniqueness(conn, flags):
    flags["continuation_keys"] = True
    outcome = kau.build_outcome_key("sophie", "klaviyo", "list/4", "read_only")
    pred = kb.create_task(conn, title="terminal predecessor", assignee="sophie")
    kb.complete_task(conn, pred, summary="done")

    cont = kb.create_task(
        conn, title="continuation", assignee="sophie",
        access_continuation_of=(outcome, "sha256:unit1", pred),
    )
    key = kau.build_continuation_key(outcome, "sha256:unit1", pred)
    assert kau.active_access_unit(conn, key) == cont

    dup = kb.create_task(
        conn, title="dup continuation", assignee="sophie",
        access_continuation_of=(outcome, "sha256:unit1", pred),
    )
    assert dup == cont  # converged on the active continuation


def test_continuation_flag_off_allows_duplicates(conn, flags):
    outcome = kau.build_outcome_key("sophie", "klaviyo", "list/4", "read_only")
    pred = kb.create_task(conn, title="pred", assignee="sophie")
    a = kb.create_task(
        conn, title="a", assignee="sophie",
        access_continuation_of=(outcome, "sha256:u", pred),
    )
    b = kb.create_task(
        conn, title="b", assignee="sophie",
        access_continuation_of=(outcome, "sha256:u", pred),
    )
    assert a != b


# ---------------------------------------------------------------------------
# semantic_recurrence
# ---------------------------------------------------------------------------

def _block_from_running(conn, tid, reason, kind=None, **fp_kwargs):
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (tid,))
    assert kb.claim_task(conn, tid, claimer="worker") is not None
    return kb.block_task(conn, tid, reason=reason, kind=kind, **fp_kwargs)


def test_semantic_progress_is_not_a_loop(conn, flags):
    """login -> consent -> masked entry: three DIFFERENT action types with the
    same outcome/unit/actor/provider. Each re-block must land in ``blocked``
    (progress), never ``triage`` (loop), because the semantic fingerprint
    changes between episodes."""
    flags["semantic_recurrence"] = True
    tid = kb.create_task(conn, title="guided session", assignee="vladamir")
    fp = dict(
        outcome="ruth|ga4|properties/7|read_only", unit="sha256:u1", actor="chris",
        provider="ga4", reason="guided_session",
    )
    assert _block_from_running(conn, tid, "need login", kind="needs_input",
                               semantic_fingerprint=kau.recurrence_fingerprint(action="provider_login", **fp))
    assert kb.get_task(conn, tid).status == "blocked"
    kb.unblock_task(conn, tid)

    assert _block_from_running(conn, tid, "need consent", kind="needs_input",
                               semantic_fingerprint=kau.recurrence_fingerprint(action="provider_consent", **fp))
    assert kb.get_task(conn, tid).status == "blocked"
    kb.unblock_task(conn, tid)

    assert _block_from_running(conn, tid, "need masked entry", kind="needs_input",
                               semantic_fingerprint=kau.recurrence_fingerprint(action="masked_entry", **fp))
    assert kb.get_task(conn, tid).status == "blocked"  # progress, not triage


def test_true_semantic_repeat_is_a_loop(conn, flags):
    flags["semantic_recurrence"] = True
    tid = kb.create_task(conn, title="stuck", assignee="vladamir")
    fp = dict(
        outcome="ruth|ga4|properties/7|read_only", unit="sha256:u1", actor="chris",
        action="provider_login", provider="ga4", reason="guided_session",
    )
    for _ in range(kb.BLOCK_RECURRENCE_LIMIT):
        assert _block_from_running(
            conn, tid, "same unresolved login", kind="needs_input",
            semantic_fingerprint=kau.recurrence_fingerprint(**fp))
        kb.unblock_task(conn, tid)
    # The LIMIT-th same-fingerprint re-block routes to triage (a true loop).
    assert _block_from_running(
        conn, tid, "same unresolved login", kind="needs_input",
        semantic_fingerprint=kau.recurrence_fingerprint(**fp))
    assert kb.get_task(conn, tid).status == "triage"


def test_recurrence_flag_off_uses_kind_only(conn, flags):
    """Flag off: identical kind re-blocks trip the existing counter (legacy)."""
    tid = kb.create_task(conn, title="legacy", assignee="vladamir")
    for _ in range(kb.BLOCK_RECURRENCE_LIMIT):
        assert _block_from_running(conn, tid, "r", kind="needs_input")
        kb.unblock_task(conn, tid)
    assert _block_from_running(conn, tid, "r", kind="needs_input")
    assert kb.get_task(conn, tid).status == "triage"


# ---------------------------------------------------------------------------
# narrow_pr_guards
# ---------------------------------------------------------------------------

PR_COMMENT_SAME_REPO = "Opened https://github.com/acme/erp/pull/42 for the connector"


def _ready_task(conn, title):
    tid = kb.create_task(conn, title=title, assignee="nadia")
    return tid


def _declare_resources(conn, tid, payload_json):
    with kb.write_txn(conn):
        conn.execute(
            "INSERT INTO task_events (task_id, run_id, kind, payload, created_at) "
            "VALUES (?, NULL, 'access_collision_resources', ?, ?)",
            (tid, payload_json, 0),
        )


def test_narrow_guard_skips_unrelated_repo(conn, flags):
    """A task whose declared collision surface touches an UNRELATED repo is
    not guarded by a PR URL comment on some other card — only genuine
    resource overlap guards. The task has its own PR comment (legacy guard
    input) but declares unrelated resources, so the narrow guard lets it run."""
    flags["narrow_pr_guards"] = True
    tid = _ready_task(conn, "connector work on other/repo")
    kb.add_comment(conn, tid, author="nadia", body=PR_COMMENT_SAME_REPO)
    _declare_resources(conn, tid, '{"repositories":["other/repo"]}')

    reason = kbd.check_respawn_guard(conn, tid, lane="ready", flags=kb._access_unit_flags())
    assert reason is None  # unrelated work proceeds despite the PR comment


def test_narrow_guard_blocks_same_resource(conn, flags):
    flags["narrow_pr_guards"] = True
    tid = _ready_task(conn, "same repo work")
    kb.add_comment(conn, tid, author="nadia", body=PR_COMMENT_SAME_REPO)
    _declare_resources(
        conn, tid, '{"repositories":["acme/erp"],"files":["acme/erp:src/app.py"]}')

    reason = kbd.check_respawn_guard(conn, tid, lane="ready", flags=kb._access_unit_flags())
    assert reason == "active_pr"
    # Exactly ONE durable wait event, never one per tick.
    waits = conn.execute(
        "SELECT COUNT(*) FROM task_events WHERE task_id = ? AND kind = 'pr_collision_wait'",
        (tid,),
    ).fetchone()[0]
    assert waits == 1
    again = kbd.check_respawn_guard(conn, tid, lane="ready", flags=kb._access_unit_flags())
    assert again == "active_pr"  # still guarded...
    waits = conn.execute(
        "SELECT COUNT(*) FROM task_events WHERE task_id = ? AND kind = 'pr_collision_wait'",
        (tid,),
    ).fetchone()[0]
    assert waits == 1  # ...but no event storm


def test_narrow_guard_no_declaration_keeps_legacy_behaviour(conn, flags):
    """A task with a PR comment but NO collision declaration keeps the legacy
    broad guard even when the flag is on (declaration is the opt-in)."""
    flags["narrow_pr_guards"] = True
    tid = _ready_task(conn, "undeclared")
    kb.add_comment(conn, tid, author="nadia", body=PR_COMMENT_SAME_REPO)
    reason = kbd.check_respawn_guard(conn, tid, lane="ready", flags=kb._access_unit_flags())
    assert reason == "active_pr"


def test_pr_guard_flag_off_keeps_broad_guard(conn, flags):
    """Flag off: the legacy broad behaviour — any PR URL comment guards —
    stays exactly as it was."""
    guarded = _ready_task(conn, "PR owner")
    kb.add_comment(conn, guarded, author="nadia", body=PR_COMMENT_SAME_REPO)
    # A different task with NO collision declaration is NOT guarded by the
    # legacy guard either (the legacy guard is per-task comments only).
    other = _ready_task(conn, "other")
    assert kbd.check_respawn_guard(conn, other, lane="ready") is None
    assert kbd.check_respawn_guard(conn, guarded, lane="ready") == "active_pr"


def test_narrow_guard_preserves_explicit_requeue_bypass(conn, flags):
    """Upstream #104277: a same-task requeue after its PR comment is an
    explicit instruction to continue that card, even with narrow guards on."""
    flags["narrow_pr_guards"] = True
    tid = kb.create_task(
        conn, title="continue PR", assignee="nadia",
        access_collision_resources=["file:acme/erp:src/app.py"],
    )
    kb.add_comment(conn, tid, author="nadia", body=PR_COMMENT_SAME_REPO)
    with kb.write_txn(conn):
        kb._append_event(conn, tid, "promoted_manual", {"reason": "explicit correction"})
    assert kbd.check_respawn_guard(
        conn, tid, lane="ready", flags=kb._access_unit_flags(),
    ) is None


def test_dispatcher_narrow_guard_waits_once_and_spawns_disjoint_file(
    conn, flags, monkeypatch,
):
    """Real dispatcher path: configured flags reach the guard; five ticks
    yield one durable wait (and no legacy event storm) for an overlapping
    active PR, while same-repo disjoint-file work proceeds."""
    import os

    flags["narrow_pr_guards"] = True
    monkeypatch.setattr(kbd, "_profile_exists_fn", lambda: (lambda _name: True))

    pr_owner = kb.create_task(
        conn, title="existing PR", assignee="nadia",
        access_collision_resources=["file:acme/erp:src/orders.py"],
    )
    kb.add_comment(
        conn, pr_owner, author="nadia",
        body="Opened https://github.com/acme/erp/pull/42",
    )
    # It is evidence-only for this fixture, not dispatchable work.
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET status='done' WHERE id=?", (pr_owner,))

    overlapping = kb.create_task(
        conn, title="same file", assignee="nadia",
        access_collision_resources=["file:acme/erp:src/orders.py"],
    )
    disjoint = kb.create_task(
        conn, title="other file", assignee="nadia",
        access_collision_resources=["file:acme/erp:src/customers.py"],
    )

    spawned: list[str] = []

    def spawn(task, _workspace):
        spawned.append(task.id)
        return os.getpid()

    for _ in range(5):
        kbd.dispatch_once(
            conn, spawn_fn=spawn, reconcile_orphans=False,
            max_in_progress=8,
        )

    assert spawned == [disjoint]
    overlapping_task = kb.get_task(conn, overlapping)
    disjoint_task = kb.get_task(conn, disjoint)
    assert overlapping_task is not None and overlapping_task.status == "ready"
    assert disjoint_task is not None and disjoint_task.status == "running"
    events = conn.execute(
        "SELECT kind, COUNT(*) AS n FROM task_events WHERE task_id = ? "
        "AND kind IN ('pr_collision_wait', 'respawn_guarded') GROUP BY kind",
        (overlapping,),
    ).fetchall()
    assert {row["kind"]: row["n"] for row in events} == {"pr_collision_wait": 1}


# ---------------------------------------------------------------------------
# stale_reconciliation
# ---------------------------------------------------------------------------

def test_completion_releases_key_and_reconciles(conn, flags):
    flags["outcome_keys"] = True
    flags["stale_reconciliation"] = True
    key = kau.build_outcome_key("ruth", "ga4", "properties/3", "read_only")
    tid = kb.create_task(conn, title="unit", assignee="vladamir", access_outcome_key=key)
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (tid,))
    assert kb.complete_task(conn, tid, summary="verified") is True
    assert kau.active_access_unit(conn, key) is None  # released on completion


def test_outcome_key_completion_releases_without_reconciliation(conn, flags):
    """Outcome-key uniqueness is independently one NON-terminal unit: even
    with stale_reconciliation off, completion releases its registry owner."""
    flags["outcome_keys"] = True
    key = kau.build_outcome_key("ruth", "ga4", "properties/3", "read_only")
    tid = kb.create_task(conn, title="unit", assignee="vladamir", access_outcome_key=key)
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (tid,))
    assert kb.complete_task(conn, tid, summary="verified") is True
    assert kau.active_access_unit(conn, key) is None
    row = conn.execute(
        "SELECT released_at, release_reason FROM access_units WHERE key = ? AND task_id = ?",
        (key, tid),
    ).fetchone()
    assert row["released_at"] is not None
    assert row["release_reason"] == "completed"


# ---------------------------------------------------------------------------
# Independent connectors stay unchained
# ---------------------------------------------------------------------------

def test_independent_units_have_no_parent_edge(conn, flags):
    """Two units for different providers never get a dependency edge between
    them, even when created back-to-back by the same orchestrator."""
    flags["outcome_keys"] = True
    ga4 = kau.build_outcome_key("ruth", "ga4", "properties/1", "read_only")
    quickfile = kau.build_outcome_key("ruth", "quickfile", "reports", "read_only")
    a = kb.create_task(conn, title="GA4 unit", assignee="vladamir", access_outcome_key=ga4)
    b = kb.create_task(conn, title="QuickFile unit", assignee="vladamir", access_outcome_key=quickfile)
    assert kb.parent_ids(conn, a) == []
    assert kb.parent_ids(conn, b) == []
    assert kb.child_ids(conn, a) == []
    assert kb.child_ids(conn, b) == []


# ---------------------------------------------------------------------------
# Review-found regressions (t_e1bf786b): unsafe global reconciliation and
# terminal-key deadlock. Probes first reproduced independently against
# head d734364c91 in a fixture board; both failed before the fixes.
# ---------------------------------------------------------------------------

def test_completed_successor_must_not_archive_unrelated_active_unit(conn, flags):
    """Probe A: completing one outcome's unit must never archive or release
    an unrelated, independently registered, non-terminal unit — regardless of
    which flags are on. Reconciliation is scoped to the verified outcome."""
    flags["outcome_keys"] = True
    flags["stale_reconciliation"] = True
    unrelated_key = kau.build_outcome_key("ruth", "quickfile", "reports", "read_only")
    unrelated = kb.create_task(
        conn, title="unfinished QuickFile", assignee="vladamir",
        access_outcome_key=unrelated_key,
    )
    successor_key = kau.build_outcome_key("ruth", "ga4", "property/1", "read_only")
    successor = kb.create_task(
        conn, title="verified GA4 successor", assignee="vladamir",
        access_outcome_key=successor_key,
    )
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (successor,))
    assert kb.complete_task(conn, successor, summary="verified") is True
    unrelated_task = kb.get_task(conn, unrelated)
    assert unrelated_task is not None and unrelated_task.status != "archived"
    # The unrelated unit keeps its key: independent active outcomes are never
    # archived or released by another outcome's reconciliation.
    assert kau.active_access_unit(conn, unrelated_key) == unrelated


def test_terminal_owner_must_not_block_successor_without_reconcile_flag(conn, flags):
    """Probe B: with outcome_keys on and stale_reconciliation OFF, a completed
    owner stops blocking: creating the same outcome again yields a NEW task.
    The uniqueness invariant is "one NON-TERMINAL unit", not "one row ever"."""
    flags["outcome_keys"] = True
    flags["stale_reconciliation"] = False
    key = kau.build_outcome_key("ruth", "meta", "account/1", "read_only")
    first = kb.create_task(conn, title="first", assignee="vladamir", access_outcome_key=key)
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (first,))
    assert kb.complete_task(conn, first, summary="done") is True
    successor = kb.create_task(conn, title="successor", assignee="vladamir", access_outcome_key=key)
    assert successor != first
    assert kau.active_access_unit(conn, key) == successor
    # History preserved: the terminal owner's registry row still exists.
    rows = conn.execute(
        "SELECT COUNT(*) FROM access_units WHERE key = ?", (key,)).fetchone()[0]
    assert rows == 2
