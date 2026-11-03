"""One agent's turn: feed -> (call -> commands)* within K commands.

The history is append-only. The only time the API history is shortened is an
on-demand compaction, after which it restarts at the returned compaction block.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass

from . import feed, llm, prompts, sandbox
from .run import Run, write_status


@dataclass
class TurnResult:
    agent: str
    calls: int = 0
    commands: int = 0
    ended: str = "done"     # done | limit | budget | hard_stop | refusal | api_error | context | sandbox


async def run_turn(run: Run, agent: str, tick: int) -> TurnResult:
    result = TurnResult(agent)
    db = run.db
    row = db.agent(agent)
    first_turn = row["turns"] == 0
    k = run.cfg.commands_per_turn
    if row["over_quota"]:
        k = min(k, run.cfg.over_quota_commands)

    box = sandbox.Sandbox(sandbox.agent_spec(run.cfg, agent))
    try:
        await box.start()
    except sandbox.SandboxError as e:
        db.event(tick, agent, "sandbox_error", str(e))
        result.ended = "sandbox"
        return result

    try:
        messages = db.history(agent)
        est = row["context_est"]
        text = feed.build(run, agent, tick)
        if first_turn:
            text = prompts.first_message(agent, row["parent"], row["spawn_prompt"]) + "\n\n" + text
        if row["over_quota"]:
            text += (f"\nYou are over your disk quota: you get {k} commands this round. "
                     "Delete files to get your full turn back.")
        if est + llm.estimate_tokens(text) >= run.cfg.compaction_trigger and messages:
            messages, est = await _compact(run, agent, tick, messages, est)
            if messages is None:
                result.ended = "context"
                return result
            text = prompts.COMPACTED_NOTE + "\n\n" + text
        content: list[dict] = []
        if messages and messages[-1]["role"] == "assistant":
            # A tool call left without a result (e.g. a crash mid-turn) must be answered first.
            for b in messages[-1]["content"]:
                if b.get("type") == "tool_use":
                    content.append(_tool_result(b["id"], "Not run: the round ended.", True))
        content.append({"type": "text", "text": text})
        user = {"role": "user", "content": content}
        messages.append(user)
        db.add_message(agent, tick, "user", user["content"])
        est += llm.estimate_tokens(text)

        commands_left = k
        seq = 0
        cut_off = 0
        while True:
            if est >= run.cfg.compaction_trigger:
                messages, est = await _compact(run, agent, tick, messages, est)
                if messages is None:
                    result.ended = "context"
                    return result
                note = {"role": "user", "content": [{"type": "text", "text": prompts.COMPACTED_NOTE}]}
                messages.append(note)
                db.add_message(agent, tick, "user", note["content"])
            if est > run.cfg.refuse_above:
                _shutdown(run, agent, tick, "context", {"estimate": est})
                result.ended = "context"
                return result

            resp = await _call(run, agent, tick, messages, est, result)
            if resp is None:
                return result
            result.calls += 1
            assistant = {"role": "assistant", "content": resp.content}
            messages.append(assistant)
            db.add_message(agent, tick, "assistant", resp.content)
            est = resp.usage.prompt_tokens + resp.usage.output_tokens
            _save_est(run, agent, est)

            if resp.stop_reason == "refusal":
                db.event(tick, agent, "refusal", {"content": resp.content})
                result.ended = "refusal"
                return result

            tool_uses = [b for b in resp.content if b.get("type") == "tool_use"]
            if not tool_uses and resp.stop_reason == "max_tokens" and cut_off < 2:
                # All thinking, no action: say so and let it continue, instead of ending the turn silently.
                cut_off += 1
                note = (f"Your reply hit the {run.cfg.max_tokens:,}-token limit before you ran anything. "
                        "Think in smaller steps and act sooner.")
                user = {"role": "user", "content": [{"type": "text", "text": note}]}
                messages.append(user)
                db.add_message(agent, tick, "user", user["content"])
                est += llm.estimate_tokens(note)
                continue
            if not tool_uses:
                return result

            per_output = max(1000, run.cfg.output_chars // len(tool_uses))
            results = []
            for tu in tool_uses:
                if resp.stop_reason == "max_tokens":
                    content = (f"Not run: your reply hit the {run.cfg.max_tokens:,}-token limit and was cut off. "
                               "Write long files in several smaller commands.")
                    results.append(_tool_result(tu["id"], content, True))
                    continue
                if commands_left <= 0:
                    results.append(_tool_result(tu["id"], "Not run: no commands left this round.", True))
                    continue
                seq += 1
                commands_left -= 1
                content, is_error, box = await _execute(run, agent, tick, seq, box, tu, per_output)
                results.append(_tool_result(tu["id"], f"{content}\n[{commands_left} commands left this round]", is_error))
                result.commands += 1
            user = {"role": "user", "content": results}
            messages.append(user)
            db.add_message(agent, tick, "user", results)
            est += sum(llm.estimate_tokens(r["content"]) for r in results)
            _save_est(run, agent, est)
            write_status(run, agent, tick)
            if commands_left <= 0:
                result.ended = "limit"
                return result
    finally:
        await box.close()
        db.execute("UPDATE agents SET turns = turns + 1 WHERE id = ?", (agent,))


def _tool_result(tool_use_id: str, content: str, is_error: bool) -> dict:
    block = {"type": "tool_result", "tool_use_id": tool_use_id, "content": content}
    if is_error:
        block["is_error"] = True
    return block


def _save_est(run: Run, agent: str, est: int) -> None:
    run.db.execute("UPDATE agents SET context_est = ? WHERE id = ?", (est, agent))


def _shutdown(run: Run, agent: str, tick: int, reason: str, detail: dict | None = None) -> None:
    run.db.set_status(agent, "finished", tick, reason)
    run.db.event(tick, agent, "shutdown", {"reason": reason, **(detail or {})})


async def _call(run: Run, agent: str, tick: int, messages: list[dict], est: int,
                result: TurnResult) -> llm.LLMResponse | None:
    """One sampling call under the budget and the concurrency limit."""
    res = await run.budget.reserve(agent, llm.worst_case_cost(est + 2000, run.cfg.max_tokens))
    if isinstance(res, str):
        if res == "agent_budget":
            _shutdown(run, agent, tick, "agent_budget")
            result.ended = "budget"
        else:
            result.ended = "hard_stop"
        return None
    try:
        async with run.semaphore:
            resp = await run.llm.sample(run.system, messages, run.cfg.max_tokens)
    except Exception as e:  # noqa: BLE001 - every API failure ends the turn, never the run
        await run.budget.release(res)
        _api_failure(run, agent, tick, e)
        result.ended = "api_error"
        return None
    await run.budget.settle(res, resp.usage.cost)
    _record_usage(run, agent, tick, resp)
    run.db.execute("UPDATE agents SET api_failures = 0 WHERE id = ?", (agent,))
    return resp


def _api_failure(run: Run, agent: str, tick: int, e: Exception) -> None:
    run.db.event(tick, agent, "api_error", {"type": type(e).__name__, "message": str(e)[:2000]})
    run.db.execute("UPDATE agents SET api_failures = api_failures + 1 WHERE id = ?", (agent,))
    if run.db.agent(agent)["api_failures"] >= 3:
        _shutdown(run, agent, tick, "api_error")


def _record_usage(run: Run, agent: str, tick: int, resp: llm.LLMResponse) -> None:
    u = resp.usage
    run.db.execute(
        "INSERT INTO usage(agent, tick, input_tokens, cache_write_tokens, cache_read_tokens, output_tokens,"
        " prompt_tokens, compaction, cost, latency, stop_reason, raw, created)"
        " VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (agent, tick, u.input_tokens, u.cache_write_tokens, u.cache_read_tokens, u.output_tokens,
         u.prompt_tokens, int(u.compaction), u.cost, resp.latency, resp.stop_reason,
         json.dumps(resp.raw_usage), time.time()),
    )


async def _compact(run: Run, agent: str, tick: int, messages: list[dict], est: int):
    """On-demand compaction. Returns (new history, estimate) or (None, est) after shutdown."""
    if est > run.cfg.refuse_above:
        _shutdown(run, agent, tick, "context", {"estimate": est})
        return None, est
    for attempt in range(2):
        res = await run.budget.reserve(agent, llm.worst_case_cost(est, run.cfg.max_tokens))
        if isinstance(res, str):
            _shutdown(run, agent, tick, "agent_budget" if res == "agent_budget" else "hard_stop_compaction")
            return None, est
        try:
            async with run.semaphore:
                resp = await run.llm.compact(run.system, messages, run.cfg.max_tokens)
        except Exception as e:  # noqa: BLE001
            await run.budget.release(res)
            run.db.event(tick, agent, "compaction_error", {"attempt": attempt, "error": str(e)[:2000]})
            continue
        await run.budget.settle(res, resp.usage.cost)
        _record_usage(run, agent, tick, resp)
        blocks = resp.content
        if resp.stop_reason == "compaction" and len(blocks) == 1 and blocks[0].get("type") == "compaction":
            run.db.add_message(agent, tick, "assistant", blocks, kind="compaction")
            run.db.event(tick, agent, "compacted", {"before": est})
            new_est = llm.estimate_tokens(blocks[0].get("content") or "") + 2500
            _save_est(run, agent, new_est)
            return [{"role": "assistant", "content": blocks}], new_est
        run.db.event(tick, agent, "compaction_error", {"attempt": attempt, "stop_reason": resp.stop_reason})
    _shutdown(run, agent, tick, "compaction_failed", {"estimate": est})
    return None, est


async def _execute(run: Run, agent: str, tick: int, seq: int, box: sandbox.Sandbox, tool_use: dict,
                   max_chars: int) -> tuple[str, bool, sandbox.Sandbox]:
    """Run one bash tool call. Returns (text for the model, is_error, possibly restarted sandbox)."""
    inp = tool_use.get("input") or {}
    if inp.get("restart"):
        await box.close()
        box = sandbox.Sandbox(box.spec)
        await box.start()
        return "The terminal was restarted.", False, box
    cmd = inp.get("command")
    if not isinstance(cmd, str) or not cmd.strip():
        return "Error: the bash tool needs a non-empty `command`.", True, box

    t0 = time.monotonic()
    res = await box.run(cmd, run.cfg.command_timeout)
    seconds = time.monotonic() - t0

    out_dir = run.paths.outputs(agent)
    out_path = out_dir / f"r{tick:04d}-{seq:02d}.txt"
    out_path.write_text(res.output)
    inside_path = f"/swarm/outputs/{agent}/{out_path.name}"

    run.db.execute(
        "INSERT INTO commands(agent, tick, seq, cmd, exit_code, seconds, output_bytes, output_path, timed_out, created)"
        " VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (agent, tick, seq, cmd, res.exit_code, seconds, res.total_bytes, str(out_path), int(res.timed_out), time.time()),
    )

    text = res.output
    parts = []
    if len(text) > max_chars:
        parts.append(f"[output cut: last {max_chars:,} of {len(text):,} characters; full output in {inside_path}]")
        text = text[-max_chars:]
    if res.dropped_bytes:
        parts.append(f"[{res.dropped_bytes:,} bytes from the middle of the output were not kept]")
    parts.append(text)
    is_error = False
    if res.timed_out:
        parts.append(f"[timed out after {run.cfg.command_timeout} s: the terminal was restarted, so "
                     "running processes, the current directory and variables were lost]")
        is_error = True
        run.db.event(tick, agent, "command_timeout", {"cmd": cmd[:500]})
    elif res.sandbox_died:
        parts.append("[the shell exited: the terminal was restarted]")
        is_error = True
    elif res.exit_code:
        parts.append(f"[exit code {res.exit_code}]")
    if res.timed_out or res.sandbox_died:
        await box.close()
        box = sandbox.Sandbox(box.spec)
        await box.start()
    return "\n".join(p for p in parts if p), is_error, box
