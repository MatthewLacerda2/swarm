"""Paths and run values.

Everything an agent may learn about its limits lives in RunConfig and is rendered
into the seed prompt. The hard stop is the one exception: it is read from
~/.config/swarm/limits.toml, which the sandbox hides.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field, asdict
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
RUNS = PROJECT / "runs"
AGENT_BIN = PROJECT / "agent_bin"
EXAM_DIR = PROJECT / "exam"
PROMPTS = Path(__file__).resolve().parent / "prompts"

# Machine-specific settings (gitignored): see config.local.example.toml.
_LOCAL_FILE = Path(__file__).resolve().parent / "config.local.toml"
LOCAL: dict = tomllib.loads(_LOCAL_FILE.read_text()) if _LOCAL_FILE.exists() else {}

BULK_ROOT = Path(os.environ.get("SWARM_BULK_ROOT", LOCAL.get("bulk_root", str(Path.home() / "swarm-bulk"))))

SECRETS_DIR = Path.home() / ".config" / "swarm"
API_KEY_FILE = SECRETS_DIR / "anthropic.key"
LIMITS_FILE = SECRETS_DIR / "limits.toml"

UV_BINARY = Path.home() / ".local" / "bin" / "uv"

# Host ports reachable from inside the sandbox (pasta -T).
OLLAMA_PORT = 11434
SEARXNG_PORT = 8888

MODEL = "claude-haiku-5-5"

# Per million tokens, prompts up to 100k. Above that the whole request costs 5x.
PRICE_INPUT = 0.10
PRICE_CACHE_WRITE_5M = 0.125
PRICE_CACHE_WRITE_1H = 0.20
PRICE_CACHE_READ = 0.01
PRICE_OUTPUT = 0.50
LONG_PROMPT_THRESHOLD = 100_000
LONG_PROMPT_MULTIPLIER = 5

BUDGET_THRESHOLDS = (20, 40, 60, 80, 85, 90, 95, 96, 97, 98, 99, 100)


@dataclass
class RunConfig:
    run_id: str
    seed: int = 0
    # Budgets (US$). run_budget is what agents are told.
    run_budget: float = 18.0
    agent_budget: float = 1.50
    # Population
    initial_agents: int = 6
    max_alive: int = 24
    max_total: int = 30
    # Turn
    commands_per_turn: int = 8
    max_ticks: int = 200
    concurrency: int = 16
    max_tokens: int = 12_000
    output_chars: int = 4_000
    # Context
    compaction_trigger: int = 85_000
    refuse_above: int = 99_000
    cache_ttl: str = "1h"
    # Sandbox
    command_timeout: int = 300
    sandbox_memory_gb: int = 3
    sandbox_cpus: int = 2
    sandbox_tasks: int = 512
    slice_memory_gb: int = 9
    slice_cpus: int = 6
    home_quota_gb: float = 1.0
    bulk_quota_gb: float = 10.0
    home_total_gb: float = 10.0
    bulk_total_gb: float = 150.0
    push_limit_mb: int = 50
    block_lan: bool = False      # the user is fine with agents reaching the local network
    # Exam
    exam_every: int = 5
    sim_timeout: int = 600
    sim_memory_gb: int = 4
    # Task scope (pilot restricts counties)
    task: str = "full"           # full | pilot
    over_quota_commands: int = 3
    min_free_gb: float = 5.0
    extra: dict = field(default_factory=dict)

    @property
    def run_dir(self) -> Path:
        return RUNS / self.run_id

    @property
    def bulk_dir(self) -> Path:
        return BULK_ROOT / self.run_id

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class HiddenLimits:
    """Never rendered anywhere an agent can see."""

    hard_stop: float = 22.0


def load_hidden_limits(run_budget: float) -> HiddenLimits:
    """Hard stop = run budget x ratio (22/18 unless limits.toml says otherwise)."""
    ratio = 22 / 18
    if LIMITS_FILE.exists():
        data = tomllib.loads(LIMITS_FILE.read_text())
        ratio = float(data.get("hard_stop_ratio", ratio))
    return HiddenLimits(hard_stop=round(run_budget * ratio, 2))


def read_api_key() -> str:
    if not API_KEY_FILE.exists():
        raise SystemExit(
            f"No API key: create {API_KEY_FILE} containing only the key, then chmod 600 it."
        )
    key = API_KEY_FILE.read_text().strip()
    if not key:
        raise RuntimeError(f"{API_KEY_FILE} is empty")
    return key
