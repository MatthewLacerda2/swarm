"""Landing pushes: fast-forward, clean merge, conflict, oversized, new branch."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from swarm import gitops

ENV = os.environ | gitops.GIT_ENV


def sh(*args, cwd):
    subprocess.run(args, cwd=cwd, env=ENV, check=True, capture_output=True)


@pytest.fixture
def repos(tmp_path):
    central = tmp_path / "central.git"
    gitops.init_central(central, "# readme\n")
    clones = {}
    for agent in ("a1", "a2"):
        outbox = tmp_path / agent / ".outbox" / "push.git"
        gitops.init_bare(outbox)
        work = tmp_path / f"{agent}-work"
        sh("git", "clone", "-q", str(central), str(work), cwd=tmp_path)
        sh("git", "remote", "set-url", "--push", "origin", str(outbox), cwd=work)
        clones[agent] = (tmp_path / agent, outbox, work)
    return tmp_path, central, clones


def commit(work: Path, name: str, text: str, msg: str = "c"):
    (work / name).write_text(text)
    sh("git", "add", "-A", cwd=work)
    sh("git", "commit", "-qm", msg, cwd=work)
    sh("git", "push", "-q", "origin", "HEAD:main", cwd=work)


def land(tmp, central, clones, agent, limit=50_000_000, seen=None):
    home, outbox, _ = clones[agent]
    staging = tmp / "staging" / f"{agent}.git"
    ok, err = gitops.sandboxed_fetch(outbox, staging, home)
    assert ok, err
    return {r.branch: r for r in gitops.land(central, staging, agent, 1, limit, seen)}


def test_ff_then_clean_merge_then_conflict(repos):
    tmp, central, clones = repos
    commit(clones["a1"][2], "a.txt", "one\n")
    commit(clones["a2"][2], "b.txt", "two\n")
    assert land(tmp, central, clones, "a1")["main"].result == "ff"
    r = land(tmp, central, clones, "a2")["main"]
    assert r.result == "merge"
    files = gitops.git("ls-tree", "--name-only", "main", git_dir=central).stdout.split()
    assert {"a.txt", "b.txt", "README.md"} <= set(files)

    # both edit the same line -> second one rejected
    for agent, text in (("a1", "from a1\n"), ("a2", "from a2\n")):
        work = clones[agent][2]
        sh("git", "pull", "-q", "--no-rebase", "origin", "main", cwd=work)
        commit(work, "README.md", text)
    assert land(tmp, central, clones, "a1")["main"].result in ("ff", "merge")
    r = land(tmp, central, clones, "a2")["main"]
    assert r.result == "rejected" and "README.md" in r.detail
    assert gitops.git("show", "main:README.md", git_dir=central).stdout == "from a1\n"


def test_unchanged_and_new_branch(repos):
    tmp, central, clones = repos
    work = clones["a1"][2]
    sh("git", "checkout", "-q", "-b", "data", cwd=work)
    (work / "d.txt").write_text("d\n")
    sh("git", "add", "-A", cwd=work)
    sh("git", "commit", "-qm", "d", cwd=work)
    sh("git", "push", "-q", "origin", "data", cwd=work)
    res = land(tmp, central, clones, "a1")
    assert res["data"].result == "new"
    assert land(tmp, central, clones, "a1")["data"].result == "unchanged"


def test_oversized_push_rejected(repos):
    tmp, central, clones = repos
    work = clones["a1"][2]
    (work / "big.bin").write_bytes(os.urandom(2_000_000))
    sh("git", "add", "-A", cwd=work)
    sh("git", "commit", "-qm", "big", cwd=work)
    sh("git", "push", "-q", "origin", "HEAD:main", cwd=work)
    r = land(tmp, central, clones, "a1", limit=1_000_000)["main"]
    assert r.result == "rejected" and "too large" in r.detail
    assert "big.bin" not in gitops.git("ls-tree", "--name-only", "main", git_dir=central).stdout


def test_hostile_outbox_config_does_not_run(repos, tmp_path):
    """A hook or fsmonitor in the agent's outbox config must not execute on the host."""
    tmp, central, clones = repos
    home, outbox, work = clones["a1"]
    canary = tmp_path / "canary"
    gitops.git("config", "core.fsmonitor", f"touch {canary}", git_dir=outbox)
    gitops.git("config", "uploadpack.packObjectsHook", f"touch {canary}; git", git_dir=outbox)
    commit(work, "x.txt", "x\n")
    assert land(tmp, central, clones, "a1")["main"].result == "ff"
    assert not canary.exists()


def test_rejected_push_not_reported_again(repos):
    tmp, central, clones = repos
    for agent, text in (("a1", "from a1\n"), ("a2", "from a2\n")):
        commit(clones[agent][2], "README.md", text)
    land(tmp, central, clones, "a1")
    first = land(tmp, central, clones, "a2")["main"]
    assert first.result == "rejected"
    again = land(tmp, central, clones, "a2", seen={"main": first.pushed})["main"]
    assert again.result == "unchanged"
