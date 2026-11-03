"""The shared repository: agent outboxes -> central repo -> read-only snapshot.

Agents push into a bare repo in their own folder ($HOME/.outbox/push.git). At the
end of the tick:

  1. a sandboxed git (no network, read-only everything except a staging repo)
     fetches the agent's branches into that agent's staging repo. Running git on a
     repository an agent controls is never done unsandboxed;
  2. the host fetches from the staging repo (created and configured by us) into
     the central repo, with object checks on;
  3. each branch lands if new, fast-forward or cleanly mergeable; conflicts and
     oversized pushes are rejected.

All git commands ignore the user's global/system config.
"""

from __future__ import annotations

import os
import stat
import subprocess
from dataclasses import dataclass
from pathlib import Path

from . import sandbox

GIT_ENV = {
    "GIT_CONFIG_GLOBAL": "/dev/null",
    "GIT_CONFIG_NOSYSTEM": "1",
    "GIT_TERMINAL_PROMPT": "0",
    "GIT_AUTHOR_NAME": "swarm",
    "GIT_AUTHOR_EMAIL": "swarm@swarm.local",
    "GIT_COMMITTER_NAME": "swarm",
    "GIT_COMMITTER_EMAIL": "swarm@swarm.local",
}


class GitError(RuntimeError):
    pass


def git(*args: str, git_dir: Path | None = None, cwd: Path | None = None, check: bool = True,
        input: str | None = None, env: dict | None = None) -> subprocess.CompletedProcess:
    cmd = ["git"]
    if git_dir is not None:
        cmd.append(f"--git-dir={git_dir}")
    cmd += list(args)
    res = subprocess.run(
        cmd, cwd=cwd, env=os.environ | GIT_ENV | (env or {}), capture_output=True, text=True, input=input,
    )
    if check and res.returncode != 0:
        raise GitError(f"{' '.join(cmd)}: {res.stderr.strip()}")
    return res


def init_bare(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    git("init", "-q", "--bare", "-b", "main", str(path))
    git("config", "receive.denyNonFastForwards", "false", git_dir=path)


def init_central(repo: Path, readme: str) -> None:
    """Central repo with one commit on main holding the README."""
    init_bare(repo)
    blob = git("hash-object", "-w", "--stdin", git_dir=repo, input=readme).stdout.strip()
    tree = git("mktree", git_dir=repo, input=f"100644 blob {blob}\tREADME.md\n").stdout.strip()
    commit = git("commit-tree", tree, "-m", "Start", git_dir=repo).stdout.strip()
    git("update-ref", "refs/heads/main", commit, git_dir=repo)


def rev(repo: Path, ref: str) -> str | None:
    res = git("rev-parse", "--verify", "-q", f"{ref}^{{commit}}", git_dir=repo, check=False)
    return res.stdout.strip() or None


def is_ancestor(repo: Path, a: str, b: str) -> bool:
    return git("merge-base", "--is-ancestor", a, b, git_dir=repo, check=False).returncode == 0


# --- fetching from an agent's outbox ------------------------------------------------------


def _real_dir(path: Path, stop: Path) -> bool:
    """path is a real directory and no component between stop and path is a symlink."""
    p = path
    while True:
        try:
            st = os.lstat(p)
        except FileNotFoundError:
            return False
        if not stat.S_ISDIR(st.st_mode):
            return False
        if p == stop:
            return True
        p = p.parent
        if p == p.parent:
            return False


def sandboxed_fetch(src: Path, staging: Path, home_root: Path, timeout: int = 180) -> tuple[bool, str]:
    """git fetch src -> staging inside bwrap: no network, only staging writable."""
    if not _real_dir(src, home_root):
        return False, "outbox missing or not a directory"
    if not staging.exists():
        init_bare(staging)
    args = [
        "timeout", "-k", "5", str(timeout),
        "bwrap", "--unshare-all", "--die-with-parent", "--new-session", "--cap-drop", "ALL",
        *sandbox._root_binds(),
        "--dev", "/dev", "--proc", "/proc", "--tmpfs", "/tmp", "--tmpfs", "/run",
        "--dir", "/home", "--dir", "/mnt",
        "--ro-bind", str(src), "/src",
        "--bind", str(staging), "/dst",
        "--clearenv",
        "--setenv", "PATH", "/usr/bin",
        "--setenv", "HOME", "/tmp",
        "--setenv", "GIT_CONFIG_NOSYSTEM", "1",
        "--setenv", "GIT_CONFIG_GLOBAL", "/dev/null",
        "--", "git", "--git-dir=/dst",
        "-c", "fetch.fsckObjects=true", "-c", "protocol.file.allow=always",
        "fetch", "-q", "--no-tags", "--no-write-fetch-head", "--prune",
        "/src", "+refs/heads/*:refs/heads/*",
    ]
    res = subprocess.run(args, capture_output=True, text=True)
    if res.returncode != 0:
        return False, res.stderr.strip()[-500:]
    return True, ""


@dataclass
class PushResult:
    branch: str
    result: str          # new | ff | merge | rejected | unchanged
    detail: str = ""
    old: str | None = None
    new: str | None = None
    commits: list[str] | None = None


def land(central: Path, staging: Path, agent: str, tick: int, limit_bytes: int) -> list[PushResult]:
    """Bring every branch of an agent's staging repo into the central repo."""
    ns = f"refs/incoming/{agent}"
    git("-c", "fetch.fsckObjects=true", "fetch", "-q", "--no-tags", "--no-write-fetch-head",
        str(staging), f"+refs/heads/*:{ns}/*", git_dir=central)
    refs = git("for-each-ref", "--format=%(refname)", f"{ns}/", git_dir=central).stdout.split()
    results: list[PushResult] = []
    try:
        for ref in refs:
            branch = ref[len(ns) + 1:]
            results.append(_land_branch(central, ref, branch, agent, tick, limit_bytes))
    finally:
        for ref in refs:
            git("update-ref", "-d", ref, git_dir=central, check=False)
    return results


def _land_branch(central: Path, ref: str, branch: str, agent: str, tick: int, limit: int) -> PushResult:
    new = rev(central, ref)
    target = f"refs/heads/{branch}"
    old = rev(central, target)
    if new is None:
        return PushResult(branch, "rejected", "not a commit")
    if old and (new == old or is_ancestor(central, new, old)):
        return PushResult(branch, "unchanged", old=old, new=new)

    usage = git("rev-list", "--objects", "--disk-usage", new, "--not", "--branches", git_dir=central)
    size = int(usage.stdout.strip() or 0)
    if size > limit:
        return PushResult(branch, "rejected", f"push too large ({size / 1e6:.1f} MB > {limit / 1e6:.0f} MB)", old, new)
    commits = git("rev-list", new, "--not", "--branches", git_dir=central).stdout.split()

    if old is None:
        git("update-ref", target, new, "", git_dir=central)
        return PushResult(branch, "new", old=old, new=new, commits=commits)
    if is_ancestor(central, old, new):
        git("update-ref", target, new, old, git_dir=central)
        return PushResult(branch, "ff", old=old, new=new, commits=commits)

    merge = git("merge-tree", "--write-tree", "--name-only", "--no-messages", old, new,
                git_dir=central, check=False)
    if merge.returncode != 0:
        lines = merge.stdout.strip().splitlines()
        conflicted = [l for l in lines[1:] if l.strip()][:10]
        return PushResult(branch, "rejected", "conflict in " + ", ".join(conflicted or ["?"]), old, new)
    tree = merge.stdout.split()[0]
    msg = f"Merge {agent}'s push to {branch} (round {tick})"
    commit = git("commit-tree", tree, "-p", old, "-p", new, "-m", msg, git_dir=central).stdout.strip()
    git("update-ref", target, commit, old, git_dir=central)
    return PushResult(branch, "merge", old=old, new=commit, commits=commits)


def mirror(central: Path, dest: Path) -> None:
    """Read-only copy for the snapshot: branches only, objects hardlinked."""
    git("clone", "-q", "--mirror", "--local", str(central), str(dest))
    git("remote", "remove", "origin", git_dir=dest, check=False)


# --- agents' $HOME history ---------------------------------------------------------------


def commit_home(history_git: Path, home: Path, agent: str, tick: int) -> None:
    """Commit an agent's $HOME with a git dir outside it (the agent can't touch its config)."""
    if not history_git.exists():
        init_bare(history_git)
        git("config", "core.bare", "false", git_dir=history_git)
        (history_git / "info").mkdir(exist_ok=True)
        (history_git / "info" / "exclude").write_text(".outbox/\n.cache/\n*.tmp\n")
    env = {"GIT_AUTHOR_NAME": agent, "GIT_AUTHOR_EMAIL": f"{agent}@swarm.local"}
    git("--work-tree", str(home), "-c", "core.fsmonitor=false", "-c", "core.hooksPath=/dev/null",
        "add", "-A", "--ignore-errors", git_dir=history_git, check=False)
    git("--work-tree", str(home), "-c", "core.hooksPath=/dev/null", "commit", "-q", "--allow-empty",
        "-m", f"round {tick}", git_dir=history_git, check=False, env=env)
