"""swarm run | status | exam | truth | calibrate"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys

from . import config


PILOT = {
    "task": "pilot", "run_budget": 1.0, "agent_budget": 0.30, "initial_agents": 3, "max_alive": 5,
    "max_total": 5, "max_ticks": 10, "bulk_quota_gb": 2.0, "bulk_total_gb": 10.0, "home_total_gb": 5.0,
}


def _build_config(run_id: str, pilot: bool, overrides: list[str] | None) -> config.RunConfig:
    cfg = config.RunConfig(run_id=run_id)
    if pilot:
        for k, v in PILOT.items():
            setattr(cfg, k, v)
    for kv in overrides or []:
        key, value = kv.split("=", 1)
        current = getattr(cfg, key)
        setattr(cfg, key, type(current)(value))
    return cfg


def cmd_init(args) -> None:
    from . import run as runmod

    if (config.RUNS / args.run_id / "swarm.db").exists():
        raise SystemExit(f"run {args.run_id} already exists")
    cfg = _build_config(args.run_id, args.pilot, args.set)
    run = runmod.open_run(cfg, client=None)  # type: ignore[arg-type]
    print(f"created {args.run_id}: {cfg.task}, {cfg.initial_agents} agents, US$ {cfg.run_budget}")
    print(f"start it with: swarm run {args.run_id}  (or systemctl --user start swarm@{args.run_id})")
    run.db.conn.close()


def _warm(run, client) -> None:
    import time

    if run.db.get_meta("warmed"):
        return
    res = asyncio.run(client.warm(run.system))
    run.db.execute(
        "INSERT INTO usage(agent, tick, input_tokens, cache_write_tokens, cache_read_tokens, output_tokens,"
        " prompt_tokens, cost, stop_reason, raw, created) VALUES('orchestrator', 0, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (res.usage.input_tokens, res.usage.cache_write_tokens, res.usage.cache_read_tokens,
         res.usage.output_tokens, res.usage.prompt_tokens, res.usage.cost, res.stop_reason,
         json.dumps(res.raw_usage), time.time()),
    )
    run.budget.spent += res.usage.cost
    run.db.set_meta("warmed", True)


def cmd_run(args) -> None:
    from . import llm, run as runmod, scheduler

    if not (config.RUNS / args.run_id / "swarm.db").exists():
        raise SystemExit(f"no run {args.run_id}; create it with `swarm init {args.run_id}`")
    cfg = runmod.load_config(args.run_id)
    client = llm.AnthropicLLM(config.read_api_key())
    run = runmod.open_run(cfg, client)
    run.db.set_meta("status", "running")
    _warm(run, client)
    reason = asyncio.run(scheduler.run_ticks(run, args.ticks))
    print(f"stopped: {reason}")


def cmd_calibrate(args) -> None:
    """One real agent on the pilot task for a few rounds: checks caching, cost and latency."""
    import time

    from . import llm, run as runmod, scheduler, status

    run_id = time.strftime("calibrate-%Y%m%d-%H%M%S")
    cfg = _build_config(run_id, True, [
        "initial_agents=1", "max_alive=1", "max_total=1", f"run_budget={args.budget}",
        f"agent_budget={args.budget}", f"max_ticks={args.rounds}", "exam_every=1000",
    ])
    client = llm.AnthropicLLM(config.read_api_key())
    run = runmod.open_run(cfg, client)
    _warm(run, client)
    asyncio.run(scheduler.run_ticks(run, args.rounds))
    print(status.render(run_id))
    print()
    print(f"{'round':>5} {'input':>7} {'write':>7} {'read':>7} {'out':>6} {'prompt':>7} {'US$':>8} {'s':>5}  stop")
    for r in run.db.all("SELECT * FROM usage ORDER BY id"):
        print(f"{r['tick']:>5} {r['input_tokens'] or 0:>7} {r['cache_write_tokens'] or 0:>7} "
              f"{r['cache_read_tokens'] or 0:>7} {r['output_tokens'] or 0:>6} {r['prompt_tokens'] or 0:>7} "
              f"{r['cost']:>8.5f} {r['latency'] or 0:>5.1f}  {r['stop_reason']}")


def cmd_status(args) -> None:
    from . import status

    print(status.render(args.run_id))


def cmd_exam(args) -> None:
    from . import exam

    exam.main([args.run_id, "--tick", str(args.tick)] + (["--events", *args.events] if args.events else []))


def cmd_truth(args) -> None:
    from . import truth

    truth.main(args.events)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(prog="swarm")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("init", help="create a run (no API calls)")
    p.add_argument("run_id")
    p.add_argument("--pilot", action="store_true", help="pilot settings (3 counties, US$ 1, 3 agents)")
    p.add_argument("--set", nargs="*", help="override config values, e.g. --set concurrency=8")
    p.set_defaults(func=cmd_init)

    p = sub.add_parser("run", help="start or resume a run")
    p.add_argument("run_id")
    p.add_argument("--ticks", type=int, help="stop after this many rounds")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("calibrate", help="one real agent, a few rounds: cache, cost, latency")
    p.add_argument("--rounds", type=int, default=8)
    p.add_argument("--budget", type=float, default=0.30)
    p.set_defaults(func=cmd_calibrate)

    p = sub.add_parser("status", help="summary of a run")
    p.add_argument("run_id")
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("exam", help="run the exam on a run's main branch now")
    p.add_argument("run_id")
    p.add_argument("--tick", type=int, default=0)
    p.add_argument("--events", nargs="*")
    p.set_defaults(func=cmd_exam)

    p = sub.add_parser("truth", help="build the exam's ground truth")
    p.add_argument("events", nargs="*")
    p.set_defaults(func=cmd_truth)

    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    sys.exit(main())
