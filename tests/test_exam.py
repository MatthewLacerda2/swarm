"""Exam runner: checkout, sandboxed sim runs, validation, scoring, no leaks."""

from __future__ import annotations

import shutil
import subprocess
import uuid

import pytest

from swarm import config, exam, truth

# The exam's events and ground truth are not in the public repository (agents could find
# them online). Build them with `swarm truth` after adding exam/events/*.json.
needs_exam = pytest.mark.skipif(
    not (truth.TRUTH_DIR / "lordstown-2019.csv").exists(), reason="exam data not present"
)

ZERO_SIM = r'''#!/usr/bin/env python3
import sys, json
args = sys.argv[1:]
out = args[args.index("--out") + 1]
sectors = "11 21 22 23 31-33 42 44-45 48-49 51 52 53 54 55 56 61 62 71 72 81 gov".split()
with open(out, "w") as f:
    if args[0] == "run":
        h = int(args[args.index("--horizon") + 1])
        f.write("fips,naics2,quarter,impact_pct\n")
        for fips in ("39099", "39155", "42085"):
            for s in sectors:
                for q in range(h + 1):
                    f.write(f"{fips},{s},{q},0\n")
    else:
        f.write("fips,naics2,quarter,employment\n39099,72,2026Q2,1000\n")
'''

SNOOP_SIM = r'''#!/bin/sh
# tries to read hidden things and to leave the shock somewhere persistent
cat /home/*/.config/swarm/anthropic.key 2>/dev/null && exit 0
ls /home /mnt
cp /swarm/exam/work/shock.json /swarm/agents/agent-001/stolen.json 2>/dev/null && exit 0
cp /swarm/exam/work/shock.json /swarm/exam/repo/stolen.json 2>/dev/null && exit 0
curl -s -m 3 https://example.com >/dev/null 2>&1 && exit 0
ln -s /etc/passwd /swarm/exam/work/out.csv
exit 0
'''


def make_run(sim_source: str):
    run_id = f"test-{uuid.uuid4().hex[:8]}"
    cfg = config.RunConfig(run_id=run_id, sim_timeout=60)
    (cfg.run_dir / "agents" / "agent-001").mkdir(parents=True)
    (cfg.bulk_dir / "agent-001").mkdir(parents=True)
    (cfg.run_dir / "outputs").mkdir()
    snap = cfg.run_dir / "snapshot"
    snap.mkdir()
    bare = snap / "repo.git"
    work = cfg.run_dir / "sys" / "seed"
    subprocess.run(["git", "init", "-q", "--bare", "-b", "main", str(bare)], check=True)
    subprocess.run(["git", "init", "-q", "-b", "main", str(work)], check=True)
    (work / "sim").write_text(sim_source)
    (work / "sim").chmod(0o755)
    env = {"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}
    import os
    subprocess.run(["git", "add", "-A"], cwd=work, check=True)
    subprocess.run(["git", "commit", "-qm", "sim"], cwd=work, check=True, env=os.environ | env)
    subprocess.run(["git", "push", "-q", str(bare), "main"], cwd=work, check=True)
    return cfg


@pytest.fixture
def cleanup():
    cfgs = []
    yield cfgs
    for cfg in cfgs:
        shutil.rmtree(cfg.run_dir, ignore_errors=True)
        shutil.rmtree(cfg.bulk_dir, ignore_errors=True)


@needs_exam
def test_score_perfect_and_zero():
    cells = truth.read_truth("lordstown-2019")
    perfect = [{"fips": c.fips, "naics2": c.sector, "quarter": c.k, "value": c.observed} for c in cells]
    skill, skill_all, coverage, n = exam.score(perfect, cells)
    assert skill == pytest.approx(1) and coverage == 1 and n > 0
    skill, _, coverage, _ = exam.score([], cells)
    assert skill == pytest.approx(0) and coverage == 0


@needs_exam
async def test_zero_sim(cleanup):
    cfg = make_run(ZERO_SIM)
    cleanup.append(cfg)
    report = await exam.run_exam(cfg, tick=5, event_ids=["lordstown-2019"])
    assert report["validity"]["sample_run"] == "ok", report["validity"]
    assert report["validity"]["forecast"] == "ok", report["validity"]
    (ev,) = report["events"]
    assert ev["ran"] and ev["skill"] == pytest.approx(0) and ev["coverage"] == 1
    assert "1 of 1 ran" in report["public"]
    assert "lordstown" not in report["public"].lower()


@needs_exam
async def test_snooping_sim_gets_nothing(cleanup):
    cfg = make_run(SNOOP_SIM)
    cleanup.append(cfg)
    report = await exam.run_exam(cfg, tick=5, event_ids=["lordstown-2019"])
    (ev,) = report["events"]
    assert not ev["ran"]
    assert "root:" not in ev["error"] and "root:" not in report["public"]
    assert not (cfg.run_dir / "agents" / "agent-001" / "stolen.json").exists()
    assert "0 of 1 ran" in report["public"]


@needs_exam
async def test_no_main(cleanup):
    cfg = make_run(ZERO_SIM)
    cleanup.append(cfg)
    shutil.rmtree(cfg.run_dir / "snapshot" / "repo.git")
    subprocess.run(["git", "init", "-q", "--bare", str(cfg.run_dir / "snapshot" / "repo.git")], check=True)
    report = await exam.run_exam(cfg, tick=1, event_ids=["lordstown-2019"])
    assert report["validity"]["main"] == "no main branch"


def test_parse_csv_rejects_bad_rows():
    rows, err = exam.parse_csv("fips,naics2,quarter,impact_pct\n39099,72,0,-1.5\n", exam.CSV_RUN_HEADER)
    assert not err and rows[0]["value"] == -1.5
    for bad in ("3909,72,0,1", "39099,722,0,1", "39099,72,x,1", "39099,72,0,nan", "39099,72,-1,1", "39099,72,0,-130"):
        _, err = exam.parse_csv("fips,naics2,quarter,impact_pct\n" + bad + "\n", exam.CSV_RUN_HEADER)
        assert err, bad
    _, err = exam.parse_csv("fips,naics2,quarter,employment\n39099,72,2026Q2,10\n", exam.CSV_FORECAST_HEADER)
    assert not err
    _, err = exam.parse_csv("a,b\n", exam.CSV_RUN_HEADER)
    assert "bad header" in err
