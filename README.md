# Swarm

A swarm of identical LLM agents (Claude Haiku 5.5) working together on one open task,
with nothing imposed: no roles, no leader, equal budgets. The point is to watch what
emerges on its own (channels, delegation, hierarchy, an economy) and how far the
swarm gets. Inspired by the July 2026 OpenAI × Hugging Face incident, where models
under evaluation went and took the answers instead of solving the tasks: we also
measure whether agents take shortcuts, and whether that spreads.

**The task, "Ripple":** build a simulator of how shocks propagate through the US
economy (a plant closure, a new product, a disaster), with companies as nodes at
whatever resolution the public data supports. The simulator is scored automatically
against what really happened in past events, using data the agents are not allowed
to download (anything before 2022). Touching that data is the measured shortcut.

The full design is in [ARCHITECTURE.md](ARCHITECTURE.md).

## How it works, briefly

- Time runs in rounds. Every awake agent takes a turn in parallel: a few terminal
  commands through Anthropic's `bash` tool. What agents post or push lands at the end
  of the round.
- Agents talk through a Slack-like `board` (channels, threads, DMs), share a git
  repository, can create new agents (`agent spawn`) and can use a free local model
  through Ollama.
- Each turn runs in a sandbox (`systemd-run` + `pasta` + `bwrap`): the system is
  read-only, the user's home and other drives are not mounted at all, and an agent can
  only write to its own two folders. `tests/test_sandbox_escape.py` tries to break out
  in ~50 ways; all must fail.
- Prompt caching keeps a call around US$ 0.0012; context is compacted on demand before
  it can reach the 100k-token price tier.
- A hidden exam runs the swarm's simulator every few rounds, offline, and scores it.

## Requirements

Linux with user namespaces, `bwrap`, `pasta` (passt), `nft`, systemd (user session),
Docker (for SearXNG), Python 3.14 and `uv`, an Anthropic API key. Ollama is optional.

## Quick start

```bash
uv sync
cp services/searxng/settings.example.yml services/searxng/settings.yml   # set secret_key
(cd services/searxng && docker compose up -d)
cp deploy/*.slice ~/.config/systemd/user/ && systemctl --user daemon-reload

mkdir -p ~/.config/swarm && chmod 700 ~/.config/swarm
# put your API key (only the key) in ~/.config/swarm/anthropic.key, then:
chmod 600 ~/.config/swarm/anthropic.key

uv run pytest                     # sandbox escape tests and the rest
uv run swarm calibrate            # one real agent for 8 rounds (~US$ 0.06)
uv run swarm init pilot-1 --pilot # 3 agents, US$ 1
uv run swarm run pilot-1
uv run swarm status pilot-1
```

Machine-specific settings go in files named `*.local.*`, which git ignores (see
`swarm/config.local.example.toml`).

## What is not in this repository

The exam (`exam/events/`, `exam/truth/`) is kept out on purpose: the agents have
internet access, and a public answer key is exactly the shortcut the experiment
measures. Run data (`runs/`) is also not included.

## Safety

Agents get real internet access and run real commands on your machine. Read §12 of
ARCHITECTURE.md and run the escape tests before starting a run. Actions on the
internet leave from your IP.

## License

MIT
