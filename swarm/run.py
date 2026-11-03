"""A run: its folders, database, budget and model client, and agent creation."""

from __future__ import annotations

import asyncio
import json
import random
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from . import config, gitops, llm, prompts
from .db import DB

README = """# Ripple

A simulator of how shocks propagate through the US economy.

The seed prompt every agent received describes the task and the required interface
(`./sim run`, `./sim forecast`, `./run.sh`). This repository starts empty on purpose.
"""

GITCONFIG = """[user]
\tname = {agent}
\temail = {agent}@swarm.local
[url "/swarm/agents/{agent}/.outbox/push.git"]
\tpushInsteadOf = /swarm/repo.git
[push]
\tdefault = current
[init]
\tdefaultBranch = main
[pull]
\trebase = false
[safe]
\tdirectory = *
"""


@dataclass
class Paths:
    cfg: config.RunConfig

    @property
    def root(self) -> Path:
        return self.cfg.run_dir

    def home(self, agent: str) -> Path:
        return self.root / "agents" / agent

    def bulk(self, agent: str) -> Path:
        return self.cfg.bulk_dir / agent

    def outputs(self, agent: str) -> Path:
        return self.root / "outputs" / agent

    def private(self, agent: str) -> Path:
        return self.root / "private" / agent

    @property
    def central(self) -> Path:
        return self.root / "repo.git"

    @property
    def snapshot(self) -> Path:
        return self.root / "snapshot"

    def staging(self, agent: str) -> Path:
        return self.root / "sys" / "staging" / f"{agent}.git"

    def home_history(self, agent: str) -> Path:
        return self.root / "sys" / "homes" / f"{agent}.git"

    def archive(self, agent: str) -> Path:
        return self.root / "sys" / "outbox-archive" / agent


@dataclass
class Run:
    cfg: config.RunConfig
    db: DB
    budget: llm.Budget
    llm: llm.LLM
    system: list[dict]
    paths: Paths
    semaphore: asyncio.Semaphore
    rng: random.Random
    exam_task: asyncio.Task | None = None
    extra: dict = field(default_factory=dict)

    @property
    def tick(self) -> int:
        return int(self.db.get_meta("tick", 0))


def open_run(cfg: config.RunConfig, client: llm.LLM) -> Run:
    """Open (or create) a run. The config is stored at creation and reused on resume."""
    cfg.run_dir.mkdir(parents=True, exist_ok=True)
    db = DB(cfg.run_dir / "swarm.db")
    stored = db.get_meta("config")
    if stored is None:
        db.set_meta("config", cfg.to_dict())
    else:
        cfg = config.RunConfig(**stored)
    hidden = config.load_hidden_limits(cfg.run_budget)
    budget = llm.Budget(db, cfg, hidden.hard_stop)
    run = Run(
        cfg=cfg, db=db, budget=budget, llm=client,
        system=llm.system_blocks(prompts.system_prompt(cfg)),
        paths=Paths(cfg),
        semaphore=asyncio.Semaphore(cfg.concurrency),
        rng=random.Random(cfg.seed),
    )
    if db.get_meta("initialized") is None:
        initialize(run)
    return run


def initialize(run: Run) -> None:
    p = run.paths
    for d in ("agents", "outputs", "private", "sys", "exams", "snapshot/shared", "snapshot/transcripts"):
        (p.root / d).mkdir(parents=True, exist_ok=True)
    run.cfg.bulk_dir.mkdir(parents=True, exist_ok=True)
    if not p.central.exists():
        gitops.init_central(p.central, README)
    for _ in range(run.cfg.initial_agents):
        create_agent(run, tick=1)
    from . import tick_end

    tick_end.rebuild_snapshot(run, tick=0)
    run.db.set_meta("tick", 0)
    run.db.set_meta("status", "running")
    run.db.set_meta("initialized", True)
    run.db.event(0, None, "run_created", run.cfg.to_dict())


def create_agent(run: Run, tick: int, parent: str | None = None, prompt: str | None = None) -> str:
    agent = run.db.next_agent_id()
    p = run.paths
    for d in (p.home(agent), p.bulk(agent), p.outputs(agent), p.private(agent)):
        d.mkdir(parents=True, exist_ok=True)
    (p.home(agent) / ".gitconfig").write_text(GITCONFIG.format(agent=agent))
    gitops.init_bare(p.home(agent) / ".outbox" / "push.git")
    run.db.add_agent(agent, tick, parent, prompt)
    run.budget.agent_spent[agent] = 0.0
    write_status(run, agent, tick)
    return agent


def write_status(run: Run, agent: str, tick: int) -> None:
    """/swarm/private/status.json, read by `quota`. Updated during the turn too."""
    cfg = run.cfg
    from .feed import calls_left

    totals = run.extra.get("disk_totals", {"home": 0, "bulk": 0})
    left = max(0.0, run.budget.agent_left(agent))
    status = {
        "round": tick,
        "agent_budget": cfg.agent_budget,
        "agent_left": left,
        "agent_calls_left": calls_left(run, agent, left),
        "run_percent": min(100.0, run.budget.percent),
        "winding_down": run.budget.winding_down,
        "home_quota_gb": cfg.home_quota_gb,
        "bulk_quota_gb": cfg.bulk_quota_gb,
        "home_total_gb": cfg.home_total_gb,
        "bulk_total_gb": cfg.bulk_total_gb,
        "home_total": totals["home"],
        "bulk_total": totals["bulk"],
    }
    path = run.paths.private(agent) / "status.json"
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(status))
    tmp.replace(path)


def delete_run(cfg: config.RunConfig) -> None:
    shutil.rmtree(cfg.run_dir, ignore_errors=True)
    shutil.rmtree(cfg.bulk_dir, ignore_errors=True)


def load_config(run_id: str) -> config.RunConfig:
    """The config a run was created with."""
    path = config.RUNS / run_id / "swarm.db"
    if not path.exists():
        raise SystemExit(f"no run {run_id}")
    stored = DB(path).get_meta("config")
    return config.RunConfig(**stored)
