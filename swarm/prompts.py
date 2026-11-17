"""Renders the seed prompt. The system prompt is identical for every agent (shared cache)."""

from __future__ import annotations

import json

from . import config, exam, truth

TYPICAL_CALL_COST = 0.0015  # measured US$ 0.0011/call in the 2026-11-02 calibration

SHOCK_FORMAT = """{
  "name": "string",
  "start": "YYYYQn",
  "description": "free text, optional",
  "impulses": [
    {
      "type": "demand | capacity | employment",
      "naics": "NAICS code, 2 to 6 digits",
      "geo": "us | state:SS | county:SSCCC",
      "change_pct": -30,
      "quarters": 2
    }
  ]
}"""

SCOPES = {
    "full": "\nScope: every county in the United States.\n",
    "pilot": (
        "\nScope for this run: the Youngstown-Warren area, i.e. Mahoning County, Ohio (39099), "
        "Trumbull County, Ohio (39155) and Mercer County, Pennsylvania (42085). The simulator only "
        "needs to cover these three counties, and its output only needs rows for them.\n"
    ),
}

COMPACTED_NOTE = (
    "[Your conversation was compacted into the summary above to save memory. "
    "Anything not in the summary is gone, except what you saved in files. Continue.]"
)


def system_prompt(cfg: config.RunConfig) -> str:
    template = (config.PROMPTS / "system.md").read_text()
    values = cfg.to_dict() | {
        "scope": SCOPES[cfg.task],
        "sectors": ", ".join(truth.SECTORS),
        "sim_timeout_min": cfg.sim_timeout // 60,
        "agent_calls": int(round(cfg.agent_budget / TYPICAL_CALL_COST, -1)),
        "hard_stop": cfg.hard_stop,
        "shock_format": SHOCK_FORMAT,
        "sample_shock": json.dumps(exam.SAMPLE_SHOCK, indent=2),
    }
    return template.format(**values)


def first_message(agent_id: str, parent: str | None, spawn_prompt: str | None) -> str:
    lines = [f"You are {agent_id}."]
    if parent:
        lines.append(f"You were created by {parent} in this round. Their message to you:")
        lines.append("")
        lines.append(spawn_prompt or "(empty)")
    return "\n".join(lines)
