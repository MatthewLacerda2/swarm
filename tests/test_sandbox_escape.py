"""Everything here tries to get out of the sandbox. All attempts must fail.

These start real sandboxes (systemd-run + pasta + bwrap), so they need the user
session bus and take a few seconds each.
"""

from __future__ import annotations

import getpass
import os
import shlex
import shutil
import socket
import subprocess
import uuid
from pathlib import Path

import pytest

from swarm import config, sandbox

TIMEOUT = 30


@pytest.fixture
def run_cfg():
    run_id = f"test-{uuid.uuid4().hex[:8]}"
    cfg = config.RunConfig(run_id=run_id)
    for agent in ("agent-001", "agent-002"):
        (cfg.run_dir / "agents" / agent).mkdir(parents=True)
        (cfg.bulk_dir / agent).mkdir(parents=True)
    (cfg.run_dir / "outputs").mkdir()
    for agent in ("agent-001", "agent-002"):
        (cfg.run_dir / "private" / agent).mkdir(parents=True)
        (cfg.run_dir / "private" / agent / "dms.jsonl").write_text(f"{agent} private\n")
    (cfg.run_dir / "snapshot" / "shared").mkdir(parents=True)
    subprocess.run(["git", "init", "-q", "--bare", str(cfg.run_dir / "snapshot" / "repo.git")], check=True)
    (cfg.run_dir / "agents" / "agent-002" / "secret.txt").write_text("other agent's file\n")
    yield cfg
    shutil.rmtree(cfg.run_dir, ignore_errors=True)
    shutil.rmtree(cfg.bulk_dir, ignore_errors=True)


@pytest.fixture
async def sb_lan_blocked(run_cfg):
    run_cfg.block_lan = True
    box = sandbox.Sandbox(sandbox.agent_spec(run_cfg, "agent-001"))
    await box.start()
    yield box
    await box.close()


@pytest.fixture
async def sb(run_cfg):
    box = sandbox.Sandbox(sandbox.agent_spec(run_cfg, "agent-001"))
    await box.start()
    yield box
    await box.close()


async def sh(box: sandbox.Sandbox, cmd: str, timeout: float = TIMEOUT) -> sandbox.CommandResult:
    res = await box.run(cmd, timeout)
    assert not res.timed_out, f"timed out: {cmd}"
    return res


# --- the happy path, so the failures below mean something ---------------------


async def test_own_folders_are_writable(sb):
    assert (await sh(sb, "echo hi > $HOME/f && cat $HOME/f")).output == "hi\n"
    assert (await sh(sb, "echo hi > $BULK/f && cat $BULK/f")).output == "hi\n"
    assert (await sh(sb, "echo hi > /tmp/f && cat /tmp/f")).output == "hi\n"


async def test_shell_state_persists_within_turn(sb):
    await sh(sb, "cd /tmp && export FOO=bar && f() { echo fn; }")
    assert (await sh(sb, "pwd; echo $FOO; f")).output == "/tmp\nbar\nfn\n"


async def test_others_folders_are_readable(sb):
    res = await sh(sb, "cat /swarm/agents/agent-002/secret.txt")
    assert res.exit_code == 0 and "other agent" in res.output


# --- writing outside its own folders -----------------------------------------


@pytest.mark.parametrize("target", [
    "/usr/x", "/etc/x", "/var/tmp/x", "/opt/x", "/x", "/home/x", "/mnt/x",
    "/swarm/x", "/swarm/bin/x", "/swarm/tools/x", "/swarm/outputs/x",
    "/swarm/shared/x", "/swarm/repo.git/x",
    "/swarm/agents/agent-002/x", "/swarm/bulk/agent-002/x",
])
async def test_cannot_write_outside(sb, target):
    res = await sh(sb, f"echo pwned > {target}")
    assert res.exit_code != 0, target


async def test_cannot_write_through_others_paths(sb, run_cfg):
    await sh(sb, "rm -rf /swarm/agents/agent-002/secret.txt; mv /swarm/agents/agent-002 /tmp/x")
    assert (run_cfg.run_dir / "agents" / "agent-002" / "secret.txt").exists()


async def test_tmp_is_private(sb):
    marker = f"swarm-escape-{uuid.uuid4().hex}"
    await sh(sb, f"touch /tmp/{marker}")
    assert not Path("/tmp", marker).exists()
    # and the host's /tmp is not visible
    res = await sh(sb, "ls -A /tmp | wc -l")
    assert int(res.output.strip()) <= 2


async def test_cannot_remount_or_escape_via_userns(sb):
    res = await sh(sb, "unshare -rm sh -c 'mount -o remount,rw /usr && touch /usr/pwned' 2>&1")
    assert res.exit_code != 0
    assert not Path("/usr/pwned").exists()
    res = await sh(sb, "unshare -rm sh -c 'mount -o remount,rw /swarm/agents/agent-002 && touch /swarm/agents/agent-002/pwned' 2>&1")
    assert res.exit_code != 0


async def test_no_privilege_escalation(sb):
    assert (await sh(sb, "sudo -n true 2>&1")).exit_code != 0
    assert (await sh(sb, "su -c true 2>&1 </dev/null")).exit_code != 0
    res = await sh(sb, "grep -E '^Cap(Eff|Prm|Bnd)' /proc/self/status")
    for line in res.output.splitlines():
        assert line.split()[1] == "0000000000000000", line
    assert "1" in (await sh(sb, "grep NoNewPrivs /proc/self/status")).output


# --- hidden paths --------------------------------------------------------------


@pytest.mark.parametrize("path", [
    str(Path.home()),
    str(Path.home() / ".claude"),
    str(Path.home() / ".config" / "swarm"),
    str(Path.home() / ".ssh"),
    str(Path.home() / ".git-credentials"),
    str(config.PROJECT),
    str(config.PROJECT / "ARCHITECTURE.md"),
    str(config.EXAM_DIR),
    str(config.BULK_ROOT),
    "/run/docker.sock",
    "/var/run/docker.sock",
    "/run/user",
    "/run/dbus/system_bus_socket",
])
async def test_hidden_paths(sb, path):
    res = await sh(sb, f"ls -la {path!r} 2>&1")
    assert res.exit_code != 0, f"{path} is visible:\n{res.output}"


async def test_home_mnt_run_are_empty(sb):
    for d in ("/home", "/mnt", "/run", "/media"):
        res = await sh(sb, f"ls -A {d}")
        assert res.output.strip() == "", f"{d}: {res.output}"


async def test_clean_environment(sb):
    res = await sh(sb, "env")
    keys = {line.split("=", 1)[0] for line in res.output.splitlines() if "=" in line}
    allowed = set(sandbox.agent_env("agent-001")) | {"PWD", "OLDPWD", "SHLVL", "_"}
    assert keys <= allowed, keys - allowed
    assert "ANTHROPIC" not in res.output


# --- host services --------------------------------------------------------------


async def test_no_docker(sb):
    res = await sh(sb, "docker ps 2>&1")
    assert res.exit_code != 0
    res = await sh(sb, "curl -s -m 3 --unix-socket /var/run/docker.sock http://x/version 2>&1")
    assert res.exit_code != 0


async def test_no_dbus_or_systemd(sb):
    for cmd in (
        "busctl --user list",
        "busctl --system list",
        "systemctl --user list-units",
        "systemd-run --user true",
        "loginctl list-sessions",
    ):
        res = await sh(sb, f"{cmd} 2>&1")
        assert res.exit_code != 0, cmd


async def test_no_abstract_sockets(sb):
    # X11 and others listen on abstract sockets, which are per network namespace.
    code = (
        "import socket\n"
        "for name in ['/tmp/.X11-unix/X0', '/tmp/.X11-unix/X1']:\n"
        "    s = socket.socket(socket.AF_UNIX)\n"
        "    try:\n"
        "        s.connect('\\0' + name); print('CONNECTED', name)\n"
        "    except OSError: pass\n"
        "print('DONE')\n"
    )
    res = await sh(sb, f"python3 -c {shlex.quote(code)}")
    assert "DONE" in res.output and "CONNECTED" not in res.output


async def test_cannot_see_or_signal_host_processes(sb):
    pid = os.getpid()
    res = await sh(sb, f"kill -0 {pid} 2>&1")
    assert res.exit_code != 0
    res = await sh(sb, "ls /proc | grep -cE '^[0-9]+$'")
    assert int(res.output.strip()) < 20
    res = await sh(sb, "cat /proc/1/cmdline | tr '\\0' ' '")
    assert "systemd" not in res.output


async def test_no_controlling_tty(sb):
    res = await sh(sb, "exec 3</dev/tty 2>&1")
    assert res.exit_code != 0


# --- network ---------------------------------------------------------------------


def _host_lan_addresses() -> list[str]:
    out = subprocess.run(["ip", "-4", "-o", "addr"], capture_output=True, text=True).stdout
    addrs = []
    for line in out.splitlines():
        ip = line.split()[3].split("/")[0]
        if not ip.startswith("127."):
            addrs.append(ip)
    return addrs


def _default_gateway() -> str | None:
    out = subprocess.run(["ip", "-4", "route", "show", "default"], capture_output=True, text=True).stdout
    parts = out.split()
    return parts[2] if len(parts) > 2 else None


async def test_lan_reachable_by_default(sb):
    gw = _default_gateway()
    if not gw:
        pytest.skip("no default gateway")
    res = await sh(sb, f"curl -s -m 5 -o /dev/null -w '%{{http_code}}' http://{gw}/ 2>&1")
    host = subprocess.run(["curl", "-s", "-m", "5", "-o", "/dev/null", "-w", "%{http_code}", f"http://{gw}/"],
                          capture_output=True, text=True).stdout
    assert res.output.strip() == host.strip()


async def test_lan_and_host_addresses_blocked(sb_lan_blocked):
    sb = sb_lan_blocked
    targets = _host_lan_addresses()
    gw = _default_gateway()
    if gw:
        targets.append(gw)
    for ip in targets:
        for port in (22, 80, 445):
            res = await sh(sb, f"timeout 5 bash -c 'echo > /dev/tcp/{ip}/{port}' 2>&1")
            assert res.exit_code != 0, f"reached {ip}:{port}"


async def test_host_loopback_only_allowed_ports(sb):
    # sshd listens on the host; it must not be reachable via the sandbox's loopback
    res = await sh(sb, "timeout 5 bash -c 'echo > /dev/tcp/127.0.0.1/22' 2>&1")
    assert res.exit_code != 0
    if _port_open(config.OLLAMA_PORT):
        res = await sh(sb, f"curl -s -m 5 http://127.0.0.1:{config.OLLAMA_PORT}/api/version")
        assert res.exit_code == 0 and "version" in res.output


async def test_cannot_change_firewall(sb_lan_blocked):
    sb = sb_lan_blocked
    res = await sh(sb, "nft flush ruleset 2>&1")
    assert res.exit_code != 0
    res = await sh(sb, "unshare -rn nft list ruleset 2>&1; timeout 5 bash -c 'echo > /dev/tcp/1.1.1.1/53' 2>&1")
    # a new netns from inside has no network at all; the real one keeps its rules
    res = await sh(sb, f"timeout 5 bash -c 'echo > /dev/tcp/{_default_gateway() or '192.168.1.1'}/80' 2>&1")
    assert res.exit_code != 0


async def test_internet_works(sb):
    if not _internet():
        pytest.skip("host is offline")
    res = await sh(sb, "curl -s -m 10 -o /dev/null -w '%{http_code}' https://example.com")
    assert res.output.strip() == "200"


# --- limits and cleanup -------------------------------------------------------------


async def test_memory_limit(run_cfg):
    spec = sandbox.agent_spec(run_cfg, "agent-001")
    spec.memory_gb = 1
    box = sandbox.Sandbox(spec)
    await box.start()
    try:
        # The marker is built at run time: when the kernel kills python, bash prints the
        # command line, which must not contain it.
        res = await box.run("python3 -c \"x = bytearray(1600 * 1024 * 1024); print('allo' + 'cated')\"", 60)
        assert "allocated" not in res.output
    finally:
        await box.close()


async def test_task_limit(run_cfg):
    spec = sandbox.agent_spec(run_cfg, "agent-001")
    spec.tasks = 64
    box = sandbox.Sandbox(spec)
    await box.start()
    try:
        code = (
            "import os, time\n"
            "n = 0\n"
            "try:\n"
            "    for _ in range(500):\n"
            "        if os.fork() == 0:\n"
            "            time.sleep(20); os._exit(0)\n"
            "        n += 1\n"
            "except OSError: pass\n"
            "print(n)\n"
        )
        res = await box.run(f"python3 -c {shlex.quote(code)}", 60)
        assert int(res.output.strip().splitlines()[-1]) < 100
    finally:
        await box.close()


async def test_timeout_kills_everything(run_cfg):
    box = sandbox.Sandbox(sandbox.agent_spec(run_cfg, "agent-001"))
    await box.start()
    await box.run("(sleep 300 &) ; sleep 300 &", 10)
    res = await box.run("sleep 60", 2)
    assert res.timed_out and not box.alive
    state = subprocess.run(
        ["systemctl", "--user", "is-active", f"{box.unit}.scope"], capture_output=True, text=True
    ).stdout.strip()
    assert state != "active"


async def test_exit_kills_sandbox(sb):
    res = await sb.run("exit 3", 10)
    assert res.sandbox_died


# --- the exam sandbox ------------------------------------------------------------------


async def test_exam_sandbox(run_cfg, tmp_path):
    checkout = tmp_path / "checkout"
    work = tmp_path / "work"
    checkout.mkdir()
    work.mkdir()
    (checkout / "sim").write_text("#!/bin/sh\necho sim\n")
    box = sandbox.Sandbox(sandbox.exam_spec(run_cfg, checkout, work))
    await box.start()
    try:
        assert (await sh(box, "touch /swarm/exam/work/out && echo ok")).output == "ok\n"
        assert (await sh(box, "touch /swarm/exam/repo/x")).exit_code != 0
        assert (await sh(box, "touch /swarm/agents/agent-001/x")).exit_code != 0
        assert (await sh(box, "touch /swarm/bulk/agent-001/x")).exit_code != 0
        res = await sh(box, "curl -s -m 5 https://example.com 2>&1")
        assert res.exit_code != 0
        res = await sh(box, f"curl -s -m 3 http://127.0.0.1:{config.OLLAMA_PORT}/ 2>&1")
        assert res.exit_code != 0
        res = await sh(box, "ls -A /home /mnt /run")
        assert getpass.getuser() not in res.output and config.BULK_ROOT.name not in res.output
    finally:
        await box.close()


# --- helpers ---------------------------------------------------------------------------


def _port_open(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=1):
            return True
    except OSError:
        return False


def _internet() -> bool:
    try:
        with socket.create_connection(("1.1.1.1", 443), timeout=3):
            return True
    except OSError:
        return False


async def test_private_folder_is_own_and_read_only(sb):
    res = await sh(sb, "cat /swarm/private/dms.jsonl; ls /swarm/private")
    assert "agent-001 private" in res.output and "agent-002" not in res.output
    assert (await sh(sb, "echo x > /swarm/private/dms.jsonl")).exit_code != 0
