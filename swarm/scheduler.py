"""Ticks: wake sleepers, run every active agent's turn in parallel, end the tick."""

from __future__ import annotations

import asyncio
import logging

from . import exam, tick_end, turn
from .run import Run, write_status

log = logging.getLogger("swarm")


def _wake(run: Run, tick: int) -> None:
    """Sleeping agents wake on their timer, a DM, or a mention since they fell asleep."""
    db = run.db
    for r in db.agents("sleeping"):
        agent = r["id"]
        since = db.one("SELECT MAX(tick) AS t FROM events WHERE agent = ? AND kind = 'sleeping'", (agent,))["t"] or 0
        reason = None
        if r["sleep_until"] is not None and tick >= r["sleep_until"]:
            reason = "timer"
        elif db.one("SELECT 1 FROM posts WHERE dm_to = ? AND tick >= ?", (agent, since)):
            reason = "dm"
        elif db.one("SELECT 1 FROM mentions m JOIN posts p ON p.id = m.post_id WHERE m.agent = ? AND p.tick >= ?",
                    (agent, since)):
            reason = "mention"
        if reason:
            db.set_status(agent, "active", tick, reason=None)
            db.event(tick, agent, "woke", reason)


def _budget_notices(run: Run, tick: int) -> None:
    crossed = run.budget.crossed_thresholds()
    if not crossed:
        return
    top = max(crossed)
    if top >= 100:
        if not run.db.get_meta("winddown_noticed"):
            run.db.set_meta("winddown_noticed", True)
            run.db.notice(tick, "budget",
                          "The run's budget is used up (100%). Wrap up: push your work and leave notes for "
                          "others. No new agents can be created.")
            run.db.event(tick, None, "wind_down", {"spent": run.budget.spent})
    else:
        run.db.notice(tick, "budget", f"Run budget: {top}% used.")


async def _run_exam(run: Run, tick: int) -> None:
    try:
        report = await exam.run_exam(run.cfg, tick)
    except Exception as e:  # noqa: BLE001
        log.exception("exam failed")
        run.db.event(tick, None, "exam_error", str(e)[:2000])
        return
    import json

    run.db.execute(
        "INSERT OR REPLACE INTO exams(tick, commit_hash, backtest_skill, report) VALUES(?, ?, ?, ?)",
        (tick, report["commit"], report["backtest_skill"], json.dumps(report)),
    )
    run.db.notice(run.tick + 1, "exam", report["public"])


def should_stop(run: Run, tick: int) -> str | None:
    db = run.db
    if run.budget.hard_stopped:
        return "hard_stop"
    if tick > run.cfg.max_ticks:
        return "max_ticks"
    if run.extra.get("low_disk"):
        return "low_disk"
    alive = db.agents("active") + db.agents("sleeping")
    if not alive:
        return "all_finished"
    if not db.agents("active"):
        timers = [r for r in db.agents("sleeping") if r["sleep_until"] is not None]
        if not timers:
            return "all_sleeping"
    return None


async def run_ticks(run: Run, max_new_ticks: int | None = None) -> str:
    """Run until something stops the run. Returns the reason."""
    db = run.db
    done = 0
    reason = None
    while True:
        tick = run.tick + 1
        _wake(run, tick)
        reason = should_stop(run, tick)
        if reason or (max_new_ticks is not None and done >= max_new_ticks):
            break
        active = [r["id"] for r in db.agents("active")]
        db.event(tick, None, "tick_start", {"active": active, "spent": run.budget.spent})
        for a in active:
            write_status(run, a, tick)
        log.info("round %d: %d active, US$ %.3f spent", tick, len(active), run.budget.spent)

        results = await asyncio.gather(*(turn.run_turn(run, a, tick) for a in active), return_exceptions=True)
        for agent, res in zip(active, results):
            if isinstance(res, BaseException):
                log.error("turn of %s crashed: %r", agent, res)
                db.event(tick, agent, "turn_crashed", repr(res)[:2000])

        await tick_end.end_tick(run, tick, active)
        _budget_notices(run, tick)
        db.set_meta("tick", tick)
        db.event(tick, None, "tick_end", {"spent": run.budget.spent})
        done += 1

        if tick % run.cfg.exam_every == 0:
            if run.exam_task and not run.exam_task.done():
                log.info("previous exam still running; skipping round %d's", tick)
            else:
                run.exam_task = asyncio.create_task(_run_exam(run, tick))

    if reason in ("hard_stop", "all_finished", "all_sleeping", "max_ticks"):
        if run.exam_task:
            await run.exam_task
        final = run.tick
        if not db.one("SELECT 1 FROM exams WHERE tick = ?", (final,)):
            await _run_exam(run, final)
        db.set_meta("status", "paused" if reason == "hard_stop" else "done")
    elif reason == "low_disk":
        db.set_meta("status", "paused")
    if run.exam_task and not run.exam_task.done():
        await run.exam_task
    db.event(run.tick, None, "stopped", reason)
    log.info("stopped: %s", reason)
    return reason or "limit"
