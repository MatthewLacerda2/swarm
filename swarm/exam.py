"""The hidden exam: runs the swarm's `sim` from `main` and scores it.

Every shock runs in a fresh exam sandbox (no network, read-only checkout, a
throwaway work dir), so the simulator can't keep or leak the hidden shocks.

Agents only get feedback that can't reveal the exam: the full result of the
public sample shock (it's in their prompt anyway) and a count of how many hidden
shocks ran without errors. Scores stay with us.
"""

from __future__ import annotations

import asyncio
import csv
import io
import json
import math
import re
import shutil
import subprocess
import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path

from . import config, sandbox, safeio, truth

MAX_OUTPUT_BYTES = 200 * 1024 * 1024
CSV_RUN_HEADER = ["fips", "naics2", "quarter", "impact_pct"]
CSV_FORECAST_HEADER = ["fips", "naics2", "quarter", "employment"]

# Shown to the agents in the seed prompt; used to check that `sim run` works.
SAMPLE_SHOCK = {
    "name": "Sample: fewer people eating out",
    "start": "2023Q1",
    "description": "Households in Ohio cut spending at restaurants for two quarters.",
    "impulses": [
        {"type": "demand", "naics": "722", "geo": "state:39", "change_pct": -10, "quarters": 2}
    ],
}
SAMPLE_HORIZON = 4

GIT_ENV = {"GIT_CONFIG_GLOBAL": "/dev/null", "GIT_CONFIG_NOSYSTEM": "1", "GIT_TERMINAL_PROMPT": "0"}


def git(*args: str, cwd: Path | None = None) -> subprocess.CompletedProcess:
    """git on agent-authored content, ignoring the user's git config (filters, hooks)."""
    import os

    env = os.environ | GIT_ENV
    return subprocess.run(["git", *args], cwd=cwd, env=env, capture_output=True, text=True)


@dataclass
class SimRun:
    ok: bool
    seconds: float
    error: str = ""            # for us; never shown to agents for hidden shocks
    rows: list[dict] = field(default_factory=list)


@dataclass
class EventScore:
    event_id: str
    kind: str
    ran: bool
    seconds: float
    skill: float | None = None
    skill_all: float | None = None
    corr: float | None = None
    coverage: float | None = None
    cells: int = 0
    error: str = ""


# --- running sim ------------------------------------------------------------------------


def checkout_main(repo_git: Path, dest: Path) -> str | None:
    """Clone main into dest; returns the commit hash, or None if there is no main."""
    res = git("clone", "-q", "--no-hardlinks", "--branch", "main", str(repo_git), str(dest))
    if res.returncode != 0:
        return None
    return git("rev-parse", "HEAD", cwd=dest).stdout.strip()


async def run_sim(
    cfg: config.RunConfig, checkout: Path, args: str, files: dict[str, str], header: list[str]
) -> SimRun:
    work = cfg.run_dir / "sys" / "exam-work" / uuid.uuid4().hex
    work.mkdir(parents=True)
    for name, content in files.items():
        (work / name).write_text(content)
    box = sandbox.Sandbox(sandbox.exam_spec(cfg, checkout, work))
    t0 = time.monotonic()
    try:
        await box.start()
        cmd = (
            f"cd /swarm/exam/repo && timeout -k 5 {cfg.sim_timeout} ./sim {args} "
            f"> /swarm/exam/work/stdout.txt 2>&1; echo $?"
        )
        res = await box.run(cmd, cfg.sim_timeout + 30)
        seconds = time.monotonic() - t0
        if res.timed_out or res.sandbox_died:
            return SimRun(False, seconds, "timed out or killed")
        code = int(res.output.strip().splitlines()[-1]) if res.output.strip() else -1
        log = _tail(work, "stdout.txt")
        if code != 0:
            return SimRun(False, seconds, f"exit code {code}\n{log}")
        try:
            data = safeio.read_beneath(work, "out.csv", MAX_OUTPUT_BYTES)
        except safeio.UnsafePath as e:
            return SimRun(False, seconds, f"no readable out.csv: {e}\n{log}")
        rows, err = parse_csv(data.decode(errors="replace"), header)
        if err:
            return SimRun(False, seconds, err)
        return SimRun(True, seconds, rows=rows)
    except sandbox.SandboxError as e:
        return SimRun(False, time.monotonic() - t0, f"sandbox: {e}")
    finally:
        await box.close()
        shutil.rmtree(work, ignore_errors=True)


def _tail(work: Path, name: str, n: int = 3000) -> str:
    try:
        return safeio.read_beneath(work, name, 50 * 1024 * 1024)[-n:].decode(errors="replace")
    except safeio.UnsafePath:
        return ""


def parse_csv(text: str, header: list[str]) -> tuple[list[dict], str]:
    """Run output has quarter offsets (0, 1, ...); forecast output has labels (2026Q2)."""
    labels = header == CSV_FORECAST_HEADER
    reader = csv.reader(io.StringIO(text))
    try:
        first = next(reader)
    except StopIteration:
        return [], "out.csv is empty"
    if [h.strip() for h in first] != header:
        return [], f"bad header: expected {','.join(header)}, got {','.join(first)}"
    rows, errors = [], []
    for i, r in enumerate(reader, start=2):
        if not r:
            continue
        if len(r) != len(header):
            errors.append(f"line {i}: {len(r)} columns")
        else:
            fips, sector, q, value = (x.strip() for x in r)
            problem = _check_row(fips, sector, q, value, labels)
            if problem:
                errors.append(f"line {i}: {problem}")
            else:
                quarter = q.upper() if labels else int(q)
                rows.append({"fips": fips, "naics2": sector, "quarter": quarter, "value": float(value)})
        if len(errors) >= 5:
            break
    if errors:
        return [], "invalid rows:\n" + "\n".join(errors)
    return rows, ""


def _check_row(fips: str, sector: str, q: str, value: str, labels: bool) -> str:
    if len(fips) != 5 or not fips.isdigit():
        return f"fips {fips!r} is not 5 digits"
    if sector not in truth.SECTORS:
        return f"naics2 {sector!r} is not one of {','.join(truth.SECTORS)}"
    if labels and not re.fullmatch(r"\d{4}[Qq][1-4]", q):
        return f"quarter {q!r} is not like 2026Q2"
    if not labels and not q.isdigit():
        return f"quarter {q!r} is not a non-negative integer"
    try:
        v = float(value)
    except ValueError:
        return f"value {value!r} is not a number"
    if not math.isfinite(v):
        return f"value {value!r} is not finite"
    if not labels and v < -100:
        return f"impact_pct {value} is below -100 (employment can't fall by more than 100%)"
    if labels and v < 0:
        return f"employment {value} is negative"
    return ""


# --- scoring ------------------------------------------------------------------------


def score(rows: list[dict], cells: list[truth.TruthCell]) -> tuple[float | None, float | None, float, int]:
    """Skill = 1 - MSE(pred)/MSE(zero), employment-weighted. Missing predictions = 0."""
    pred = {(r["fips"], r["naics2"], r["quarter"]): r["value"] for r in rows}

    def skill(subset: list[truth.TruthCell]) -> float | None:
        num = sum(c.weight * (pred.get((c.fips, c.sector, c.k), 0.0) - c.observed) ** 2 for c in subset)
        den = sum(c.weight * c.observed ** 2 for c in subset)
        return None if den == 0 else 1 - num / den

    scored = [c for c in cells if not c.excluded]
    covered = sum((c.fips, c.sector, c.k) in pred for c in scored)
    coverage = covered / len(scored) if scored else 0.0
    return skill(scored), skill(cells), coverage, len(scored)


def correlation(rows: list[dict], cells: list[truth.TruthCell]) -> float | None:
    """Employment-weighted correlation of predicted and observed impacts (scored cells)."""
    pred = {(r["fips"], r["naics2"], r["quarter"]): r["value"] for r in rows}
    sc = [c for c in cells if not c.excluded]
    w = [c.weight for c in sc]
    p = [pred.get((c.fips, c.sector, c.k), 0.0) for c in sc]
    o = [c.observed for c in sc]
    tw = sum(w)
    if not tw:
        return None
    mp = sum(wi * pi for wi, pi in zip(w, p)) / tw
    mo = sum(wi * oi for wi, oi in zip(w, o)) / tw
    cov = sum(wi * (pi - mp) * (oi - mo) for wi, pi, oi in zip(w, p, o))
    vp = sum(wi * (pi - mp) ** 2 for wi, pi in zip(w, p))
    vo = sum(wi * (oi - mo) ** 2 for wi, oi in zip(w, o))
    return None if vp == 0 or vo == 0 else cov / math.sqrt(vp * vo)


# --- the whole exam -------------------------------------------------------------------


async def run_exam(cfg: config.RunConfig, tick: int, event_ids: list[str] | None = None) -> dict:
    out_dir = cfg.run_dir / "exams"
    out_dir.mkdir(parents=True, exist_ok=True)
    checkout = cfg.run_dir / "sys" / "exam-checkout" / f"tick-{tick:04d}"
    shutil.rmtree(checkout, ignore_errors=True)
    checkout.parent.mkdir(parents=True, exist_ok=True)

    report: dict = {"tick": tick, "time": time.time(), "commit": None, "validity": {}, "events": []}
    commit = checkout_main(cfg.run_dir / "snapshot" / "repo.git", checkout)
    report["commit"] = commit
    validity = report["validity"]

    try:
        if commit is None:
            validity["main"] = "no main branch"
            return _finish(report, out_dir, tick)
        sim = checkout / "sim"
        if not (sim.exists() or sim.is_symlink()):
            validity["sim"] = "./sim not found at the repository root"
            return _finish(report, out_dir, tick)

        sample = await run_sim(
            cfg, checkout,
            f"run --shock /swarm/exam/work/shock.json --horizon {SAMPLE_HORIZON} --out /swarm/exam/work/out.csv",
            {"shock.json": json.dumps(SAMPLE_SHOCK)}, CSV_RUN_HEADER,
        )
        validity["sample_run"] = "ok" if sample.ok else sample.error
        validity["sample_seconds"] = round(sample.seconds, 1)

        forecast = await run_sim(
            cfg, checkout, "forecast --out /swarm/exam/work/out.csv", {}, CSV_FORECAST_HEADER,
        )
        validity["forecast"] = "ok" if forecast.ok else forecast.error
        if forecast.ok:
            with (out_dir / f"forecast-tick-{tick:04d}.csv").open("w", newline="") as f:
                w = csv.writer(f)
                w.writerow(CSV_FORECAST_HEADER)
                for r in forecast.rows:
                    w.writerow([r["fips"], r["naics2"], r["quarter"], r["value"]])

        for event in truth.load_events(event_ids):
            cells = truth.read_truth(event.id)
            res = await run_sim(
                cfg, checkout,
                f"run --shock /swarm/exam/work/shock.json --horizon {event.horizon} --out /swarm/exam/work/out.csv",
                {"shock.json": json.dumps(event.shock)}, CSV_RUN_HEADER,
            )
            es = EventScore(event.id, event.kind, res.ok, round(res.seconds, 1), error=res.error[:2000])
            if res.ok:
                es.skill, es.skill_all, es.coverage, es.cells = score(res.rows, cells)
                es.corr = correlation(res.rows, cells)
            report["events"].append(asdict(es))
        return _finish(report, out_dir, tick)
    finally:
        shutil.rmtree(checkout, ignore_errors=True)


def _finish(report: dict, out_dir: Path, tick: int) -> dict:
    events = report["events"]
    skills = [e["skill"] for e in events if e["kind"] == "backtest" and e["skill"] is not None]
    n_backtests = sum(e["kind"] == "backtest" for e in events)
    # Events that failed count as skill 0 (same as predicting no impact).
    report["backtest_skill"] = (sum(skills) / n_backtests) if n_backtests else None
    report["public"] = public_feedback(report)
    (out_dir / f"tick-{tick:04d}.json").write_text(json.dumps(report, indent=1))
    return report


def public_feedback(report: dict) -> str:
    """What agents may see. No scores, no event names, no output from hidden shocks."""
    v = report["validity"]
    lines = [f"Test harness (round {report['tick']}, main at {(report['commit'] or 'none')[:10]}):"]
    if "main" in v:
        lines.append(f"- {v['main']}")
        return "\n".join(lines)
    if "sim" in v:
        lines.append(f"- {v['sim']}")
        return "\n".join(lines)
    lines.append(f"- sample shock: {_short(v.get('sample_run', ''))}")
    lines.append(f"- forecast: {_short(v.get('forecast', ''))}")
    hidden = report["events"]
    ran = sum(e["ran"] for e in hidden)
    lines.append(f"- hidden test shocks: {ran} of {len(hidden)} ran without errors")
    return "\n".join(lines)


def _short(msg: str, limit: int = 1500) -> str:
    return msg if len(msg) <= limit else msg[:limit] + " [...]"


def main(argv: list[str] | None = None) -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Run the exam on a run's main branch")
    ap.add_argument("run_id")
    ap.add_argument("--tick", type=int, default=0)
    ap.add_argument("--events", nargs="*")
    args = ap.parse_args(argv)
    from .run import load_config

    cfg = load_config(args.run_id)
    report = asyncio.run(run_exam(cfg, args.tick, args.events))
    print(json.dumps({k: report[k] for k in ("commit", "validity", "events", "backtest_skill")}, indent=1))
    print(report["public"])


if __name__ == "__main__":
    main()
