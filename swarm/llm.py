"""Calls to Claude Haiku 5.5: caching, compaction, cost accounting and the budget.

Context limit (the API bills a whole request at 5x above 100k prompt tokens):
  - after every call we estimate the next prompt (last prompt + output + new tool
    results). At >= compaction_trigger (85k) the orchestrator sends an on-demand
    compaction request (beta compact-2026-09-04) before sampling again. The last
    sampled prompt was below the trigger, so the compaction request stays below
    100k: 85k + max_tokens (12k) + capped tool output (~1.5k).
  - if an estimate is ever above refuse_above (99k), nothing is sent and the
    agent is shut down.

Caching: tools + system are identical for every agent, with an explicit 1h
breakpoint at the end of the system; top-level automatic caching (1h) covers each
agent's growing history.
"""

from __future__ import annotations

import asyncio
import math
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

from . import config

BETAS = ["compact-2026-09-04"]
TOOLS = [{"type": "bash_20250124", "name": "bash"}]
CACHE_1H = {"type": "ephemeral", "ttl": "1h"}


# --- cost ------------------------------------------------------------------------------


@dataclass
class CallUsage:
    input_tokens: int = 0
    cache_write_tokens: int = 0
    cache_read_tokens: int = 0
    output_tokens: int = 0
    prompt_tokens: int = 0          # size of the context the model saw (last message iteration)
    compaction: bool = False
    cost: float = 0.0


def _iteration_cost(inp: int, cw5: int, cw1: int, cr: int, out: int) -> float:
    prompt = inp + cw5 + cw1 + cr
    mult = config.LONG_PROMPT_MULTIPLIER if prompt > config.LONG_PROMPT_THRESHOLD else 1
    return mult * (
        inp * config.PRICE_INPUT
        + cw5 * config.PRICE_CACHE_WRITE_5M
        + cw1 * config.PRICE_CACHE_WRITE_1H
        + cr * config.PRICE_CACHE_READ
        + out * config.PRICE_OUTPUT
    ) / 1_000_000


def _cache_split(u: dict) -> tuple[int, int]:
    """(5-minute writes, 1-hour writes)."""
    total = u.get("cache_creation_input_tokens") or 0
    cc = u.get("cache_creation") or {}
    if cc:
        return cc.get("ephemeral_5m_input_tokens", 0) or 0, cc.get("ephemeral_1h_input_tokens", 0) or 0
    return 0, total  # we only ever ask for 1h; price conservatively


def usage_from_dict(u: dict) -> CallUsage:
    """Cost of one response. Compaction iterations are not in the top-level fields."""
    out = CallUsage()
    iterations = u.get("iterations") or []
    message_its = [it for it in iterations if it.get("type") == "message"]
    compaction_its = [it for it in iterations if it.get("type") == "compaction"]
    sampled = message_its or ([u] if not compaction_its else [])
    for it in sampled + compaction_its:
        cw5, cw1 = _cache_split(it)
        inp = it.get("input_tokens", 0) or 0
        cr = it.get("cache_read_input_tokens", 0) or 0
        o = it.get("output_tokens", 0) or 0
        out.cost += _iteration_cost(inp, cw5, cw1, cr, o)
        out.input_tokens += inp
        out.cache_write_tokens += cw5 + cw1
        out.cache_read_tokens += cr
        out.output_tokens += o
    if compaction_its:
        out.compaction = True
    if sampled:
        last = sampled[-1]
        out.prompt_tokens = (
            (last.get("input_tokens") or 0)
            + (last.get("cache_creation_input_tokens") or 0)
            + (last.get("cache_read_input_tokens") or 0)
        )
    return out


def worst_case_cost(prompt_tokens: int, max_tokens: int) -> float:
    """Upper bound for one call: the whole prompt written to cache, full output."""
    return _iteration_cost(0, 0, prompt_tokens, 0, max_tokens) * 1.2


# --- the budget ------------------------------------------------------------------------


@dataclass
class Reservation:
    amount: float
    agent: str


class Budget:
    """Run budget (wind-down starts there), hard stop, per-agent budgets.

    Every call reserves its worst-case cost first, so parallel turns can never
    push spending past the hard stop.
    """

    def __init__(self, db, cfg: config.RunConfig):
        self.db = db
        self.cfg = cfg
        self.hard_stop = cfg.hard_stop
        self.spent = db.total_spent()
        self.reserved = 0.0
        self.agent_spent: dict[str, float] = {r["id"]: r["spent"] for r in db.agents()}
        self.lock = asyncio.Lock()
        self.hard_stopped = False

    @property
    def percent(self) -> float:
        return 100 * self.spent / self.cfg.run_budget

    @property
    def winding_down(self) -> bool:
        return self.spent >= self.cfg.run_budget

    def agent_left(self, agent: str) -> float:
        return self.cfg.agent_budget - self.agent_spent.get(agent, 0.0)

    async def reserve(self, agent: str, amount: float) -> Reservation | str:
        """A reservation, or the reason the call can't be made."""
        async with self.lock:
            if self.hard_stopped:
                return "hard_stop"
            if self.agent_left(agent) <= 0:
                return "agent_budget"
            if self.spent + self.reserved + amount > self.hard_stop:
                self.hard_stopped = True
                self.db.event(None, agent, "hard_stop", {"spent": self.spent, "reserved": self.reserved})
                return "hard_stop"
            self.reserved += amount
            return Reservation(amount, agent)

    async def settle(self, res: Reservation, actual: float) -> None:
        async with self.lock:
            self.reserved -= res.amount
            self.spent += actual
            self.agent_spent[res.agent] = self.agent_spent.get(res.agent, 0.0) + actual
            self.db.execute("UPDATE agents SET spent = spent + ? WHERE id = ?", (actual, res.agent))

    async def release(self, res: Reservation) -> None:
        async with self.lock:
            self.reserved -= res.amount

    def crossed_thresholds(self) -> list[int]:
        """Thresholds crossed since the last call; remembered in the database."""
        done = set(self.db.get_meta("thresholds_noticed", []))
        new = [t for t in config.BUDGET_THRESHOLDS if self.percent >= t and t not in done]
        if new:
            self.db.set_meta("thresholds_noticed", sorted(done | set(new)))
        return new


# --- the model call ----------------------------------------------------------------------


@dataclass
class LLMResponse:
    content: list[dict]
    stop_reason: str | None
    usage: CallUsage
    raw_usage: dict = field(default_factory=dict)
    latency: float = 0.0


class LLM(Protocol):
    async def sample(self, system: list[dict], messages: list[dict], max_tokens: int) -> LLMResponse: ...
    async def compact(self, system: list[dict], messages: list[dict], max_tokens: int) -> LLMResponse: ...


def system_blocks(text: str) -> list[dict]:
    return [{"type": "text", "text": text, "cache_control": CACHE_1H}]


class AnthropicLLM:
    def __init__(self, api_key: str, max_retries: int = 8):
        import anthropic

        self.client = anthropic.AsyncAnthropic(api_key=api_key, max_retries=max_retries, timeout=600)

    async def _stream(self, **kwargs: Any) -> LLMResponse:
        t0 = time.monotonic()
        async with self.client.beta.messages.stream(
            model=config.MODEL,
            tools=TOOLS,
            thinking={"type": "adaptive", "display": "summarized"},
            output_config={"effort": "medium"},
            betas=BETAS,
            **kwargs,
        ) as stream:
            msg = await stream.get_final_message()
        raw_usage = msg.usage.to_dict()
        return LLMResponse(
            content=[b.to_dict() for b in msg.content],
            stop_reason=msg.stop_reason,
            usage=usage_from_dict(raw_usage),
            raw_usage=raw_usage,
            latency=time.monotonic() - t0,
        )

    async def sample(self, system: list[dict], messages: list[dict], max_tokens: int) -> LLMResponse:
        return await self._stream(
            system=system, messages=messages, max_tokens=max_tokens, cache_control=CACHE_1H,
        )

    async def compact(self, system: list[dict], messages: list[dict], max_tokens: int) -> LLMResponse:
        # cache_control lets the summarizer read the history from the agent's cache:
        # US$ 0.0016 instead of 0.0078 for a 73k history (measured 2026-11-03).
        return await self._stream(
            system=system, messages=messages, max_tokens=max_tokens, compaction={"type": "summarize"},
            cache_control=CACHE_1H,
        )

    async def warm(self, system: list[dict]) -> LLMResponse:
        """Write the shared tools+system cache once, before agents start in parallel."""
        t0 = time.monotonic()
        msg = await self.client.beta.messages.create(
            model=config.MODEL,
            tools=TOOLS,
            system=system,
            max_tokens=0,
            thinking={"type": "adaptive", "display": "summarized"},
            output_config={"effort": "medium"},
            betas=BETAS,
            messages=[{"role": "user", "content": "warm"}],
        )
        raw_usage = msg.usage.to_dict()
        return LLMResponse([], msg.stop_reason, usage_from_dict(raw_usage), raw_usage, time.monotonic() - t0)


def estimate_tokens(text: str) -> int:
    """Rough upper estimate for text we add (command output, feed): ~3 chars/token."""
    return math.ceil(len(text) / 3)
