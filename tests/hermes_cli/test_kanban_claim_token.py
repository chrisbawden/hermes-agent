"""Delegated children must not mutate a live-claimed task by env editing.

Incident (t_ec6048fe): a ``delegate_task`` child inherited the parent's
``HERMES_KANBAN_TASK`` / claim lock, the CLI refused the mutation because
``HERMES_DELEGATED_CHILD_CONTEXT=1``, and the child then ran
``env -u HERMES_DELEGATED_CHILD_CONTEXT hermes kanban complete ...`` and
successfully completed the parent task.

The marker (env var or ContextVar) is a *negative* signal: any process that
removes it looks like the operator's shell. The fix is a per-claim capability
token: minted at claim time, stored only as a SHA-256 digest on the run row,
exported ONLY into the dispatched worker's environment, and scrubbed from
every other spawn surface. ``write_txn`` refuses mutations from any process
that presents worker identity env without the matching token.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _make_claimed_task(monkeypatch, tmp_path):
    """A claimed (``running``) task + its raw claim token, in a temp home."""
    home = tmp_path / "hermes-home"
    home.mkdir()
    db_path = tmp_path / "kanban.db"
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    for var in ("HERMES_KANBAN_TASK", "HERMES_KANBAN_RUN_ID",
                "HERMES_KANBAN_CLAIM_LOCK", "HERMES_KANBAN_CLAIM_TOKEN"):
        monkeypatch.delenv(var, raising=False)

    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc

    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    conn = kbc.connect()
    try:
        tid = kb.create_task(conn, title="victim", assignee="worker")
        claimed = kb.claim_task(conn, tid)
        assert claimed is not None
        assert claimed.claim_token, "claim_task must mint a raw claim token"
        run_id = claimed.current_run_id
        token = claimed.claim_token
    finally:
        conn.close()
    return kb, tid, run_id, token, db_path, home


def _stripped_worker_env(kb, tid, run_id, db_path, home, *, token=None) -> dict:
    """The incident grandchild env: full worker identity, marker removed."""
    env = os.environ.copy()
    env.update({
        "HERMES_HOME": str(home),
        "HERMES_KANBAN_DB": str(db_path),
        "HERMES_KANBAN_TASK": tid,
        "HERMES_KANBAN_RUN_ID": str(run_id),
        "HERMES_KANBAN_CLAIM_LOCK": "79a7777088c9:109303",  # host:pid — public
        "HERMES_KANBAN_BOARD": "default",
    })
    # The bypass: drop the delegated-child marker and any token.
    env.pop("HERMES_DELEGATED_CHILD_CONTEXT", None)
    env.pop("HERMES_KANBAN_CLAIM_TOKEN", None)
    if token:
        env["HERMES_KANBAN_CLAIM_TOKEN"] = token
    env["PYTHONPATH"] = str(_REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    return env


_CHILD_COMPLETE = """
import sys
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc

tid = sys.argv[1]
run_id = int(sys.argv[2])
try:
    conn = kbc.connect()
except PermissionError:
    print("MUTATION_REFUSED")
    sys.exit(1)
try:
    try:
        ok = kb.complete_task(conn, tid, summary="child completion", expected_run_id=run_id)
    except PermissionError:
        ok = False
finally:
    conn.close()
print("MUTATION_OK" if ok else "MUTATION_REFUSED")
sys.exit(0 if ok else 1)
"""


def test_stripped_marker_grandchild_cannot_complete_live_claim(
    monkeypatch, tmp_path,
):
    """E2E regression: the exact incident path must fail.

    A grandchild process with the parent's full worker identity env (task id,
    run id, claim lock, DB pin) but with ``HERMES_DELEGATED_CHILD_CONTEXT``
    removed — and without the claim token — must not complete the parent's
    live-claimed task. Removing/adding env markers must never equal ownership.
    """
    kb, tid, run_id, _token, db_path, home = _make_claimed_task(monkeypatch, tmp_path)

    env = _stripped_worker_env(kb, tid, run_id, db_path, home)
    proc = subprocess.run(
        [sys.executable, "-c", _CHILD_COMPLETE, tid, str(run_id)],
        cwd=str(tmp_path), env=env, capture_output=True, text=True, timeout=60,
    )

    assert "MUTATION_REFUSED" in proc.stdout, proc.stderr
    assert proc.returncode != 0

    from hermes_cli import kanban_db_connect as kbc

    conn = kbc.connect()
    try:
        task = kb.get_task(conn, tid)
        run = kb.latest_run(conn, tid)
    finally:
        conn.close()
    assert task.status == "running", "victim task must remain running"
    assert run.status == "running", "victim run must remain running"
    assert task.result is None
    assert task.claim_lock, "the live worker's claim must be untouched"


def test_worker_with_claim_token_completes_own_task(
    monkeypatch, tmp_path,
):
    """The legitimate path keeps working: the worker holding the claim token
    completes its own task through the same DB API and subprocess shape."""
    kb, tid, run_id, token, db_path, home = _make_claimed_task(monkeypatch, tmp_path)

    env = _stripped_worker_env(kb, tid, run_id, db_path, home, token=token)
    proc = subprocess.run(
        [sys.executable, "-c", _CHILD_COMPLETE, tid, str(run_id)],
        cwd=str(tmp_path), env=env, capture_output=True, text=True, timeout=60,
    )

    assert "MUTATION_OK" in proc.stdout, proc.stderr
    assert proc.returncode == 0

    from hermes_cli import kanban_db_connect as kbc

    conn = kbc.connect()
    try:
        task = kb.get_task(conn, tid)
        run = kb.latest_run(conn, tid)
    finally:
        conn.close()
    assert task.status == "done"
    assert run.outcome == "completed"


def test_clean_env_operator_still_completes_ready_task(monkeypatch, tmp_path):
    """No worker identity env at all -> operator semantics preserved."""
    kb, tid, run_id, _token, db_path, home = _make_claimed_task(monkeypatch, tmp_path)

    env = os.environ.copy()
    for var in ("HERMES_KANBAN_TASK", "HERMES_KANBAN_RUN_ID",
                "HERMES_KANBAN_CLAIM_LOCK", "HERMES_KANBAN_CLAIM_TOKEN",
                "HERMES_DELEGATED_CHILD_CONTEXT"):
        env.pop(var, None)
    env["HERMES_HOME"] = str(home)
    env["HERMES_KANBAN_DB"] = str(db_path)
    env["PYTHONPATH"] = str(_REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run(
        [sys.executable, "-c", _CHILD_COMPLETE, tid, str(run_id)],
        cwd=str(tmp_path), env=env, capture_output=True, text=True, timeout=60,
    )
    assert "MUTATION_OK" in proc.stdout, proc.stderr


def test_claim_token_scrubbed_from_delegated_child_subprocesses(monkeypatch, tmp_path):
    """The token must never cross into a delegate_task child's subprocess env."""
    kb, tid, run_id, token, _db_path, _home = _make_claimed_task(monkeypatch, tmp_path)
    monkeypatch.setenv("HERMES_KANBAN_CLAIM_TOKEN", token)
    monkeypatch.setenv("HERMES_KANBAN_TASK", tid)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(run_id))

    from agent.delegation_context import KANBAN_ENV_KEYS, delegated_child_context, scrub_kanban_env

    assert "HERMES_KANBAN_CLAIM_TOKEN" in KANBAN_ENV_KEYS
    with delegated_child_context():
        scrubbed = scrub_kanban_env(dict(os.environ))
    assert "HERMES_KANBAN_CLAIM_TOKEN" not in scrubbed
    assert scrubbed["HERMES_DELEGATED_CHILD_CONTEXT"] == "1"


def test_claim_token_stripped_from_all_spawned_subprocesses():
    """Tier-1 policy: the token never crosses ANY spawn surface."""
    from tools.environments.local_env_policy import _ALWAYS_STRIP_KEYS

    assert "HERMES_KANBAN_CLAIM_TOKEN" in _ALWAYS_STRIP_KEYS


def test_token_hash_not_plaintext_in_db(monkeypatch, tmp_path):
    """Only the digest of the token is stored; the raw token must never appear
    in task_runs, task_events, or the tasks row."""
    kb, tid, run_id, token, _db_path, _home = _make_claimed_task(monkeypatch, tmp_path)

    from hermes_cli import kanban_db_connect as kbc

    conn = kbc.connect()
    try:
        dump = "\n".join(
            f"{row[0]}|{row[1]}"
            for row in conn.execute(
                "SELECT 'tasks', (SELECT group_concat(id || ' ' || coalesce(claim_lock,''), ' ') FROM tasks)"
                " UNION ALL SELECT 'runs', (SELECT group_concat(id || ' ' || coalesce(claim_lock,''), ' ') FROM task_runs)"
                " UNION ALL SELECT 'runs_tok', (SELECT group_concat(coalesce(claim_token_hash,''), ' ') FROM task_runs)"
                " UNION ALL SELECT 'events', (SELECT group_concat(coalesce(payload,''), ' ') FROM task_events)"
            ).fetchall()
        )
    finally:
        conn.close()
    assert token not in dump, "raw claim token leaked into the DB"
    assert len(token) >= 32


def test_dispatcher_exports_claim_token_to_worker_env():
    """The dispatcher must inject the token into the worker's environment."""
    import inspect

    from hermes_cli import kanban_db_dispatch as dispatch

    src = inspect.getsource(dispatch._default_spawn)
    assert "CLAIM_TOKEN_ENV" in src and "claim_token" in src


def test_wrong_token_refused_even_with_full_worker_env(monkeypatch, tmp_path):
    """A presented-but-wrong token (spoof) must refuse, not just its absence."""
    kb, tid, run_id, token, _db_path, _home = _make_claimed_task(monkeypatch, tmp_path)

    from hermes_cli import kanban_db_connect as kbc

    monkeypatch.setenv("HERMES_KANBAN_TASK", tid)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(run_id))
    monkeypatch.setenv("HERMES_KANBAN_CLAIM_LOCK", "spoof:1")
    monkeypatch.setenv("HERMES_KANBAN_CLAIM_TOKEN", token[:-4] + "AAAA")
    conn = kbc.connect()
    try:
        import pytest

        with pytest.raises(PermissionError):
            kb.block_task(conn, tid, reason="spoofed token")
    finally:
        conn.close()
    conn = kbc.connect()
    try:
        assert kb.get_task(conn, tid).status == "running"
    finally:
        conn.close()


def test_block_requires_token_for_live_claim_in_worker_env(monkeypatch, tmp_path):
    """block from a worker-env process without the token must be refused."""
    kb, tid, run_id, _token, _db_path, _home = _make_claimed_task(monkeypatch, tmp_path)

    from hermes_cli import kanban_db_connect as kbc

    monkeypatch.setenv("HERMES_KANBAN_TASK", tid)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(run_id))
    monkeypatch.delenv("HERMES_KANBAN_CLAIM_TOKEN", raising=False)
    conn = kbc.connect()
    try:
        import pytest

        with pytest.raises(PermissionError):
            kb.block_task(conn, tid, reason="child block")
    finally:
        conn.close()
    conn = kbc.connect()
    try:
        assert kb.get_task(conn, tid).status == "running"
    finally:
        conn.close()
