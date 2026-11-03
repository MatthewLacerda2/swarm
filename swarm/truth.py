"""Builds the hidden exam's ground truth from BLS QCEW (pre-2022 data).

For each backtest event, the observed impact on employment in every county x
sector cell is the real employment against a counterfactual projected from the
pre-event trend:

    e      = event quarter, t = e + k (k = 0..horizon)
    m      = k // 4 + 1                    years back to a strictly pre-event quarter
    g      = sum(emp[e-4..e-1]) / sum(emp[e-8..e-5]) - 1     pre-event annual growth
    cf(t)  = emp(t - 4m) * (1 + g) ** m     same season, grown at the pre-event trend
    obs(t) = 100 * (emp(t) / cf(t) - 1)
    weight = mean(emp[e-4..e-1])

Sectors: private employment by NAICS sector (QCEW own_code 5, agglvl 74), plus
"gov" = federal + state + local government (own_codes 1-3, all industries).
Suppressed or missing cells are left out.
"""

from __future__ import annotations

import csv
import io
import json
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from . import config

PRIVATE_SECTORS = (
    "11", "21", "22", "23", "31-33", "42", "44-45", "48-49", "51", "52", "53",
    "54", "55", "56", "61", "62", "71", "72", "81",
)
SECTORS = PRIVATE_SECTORS + ("gov",)

RAW_DIR = config.EXAM_DIR / "raw" / "qcew"
TRUTH_DIR = config.EXAM_DIR / "truth"
EVENTS_DIR = config.EXAM_DIR / "events"

USER_AGENT = "swarm-experiment/0.1 (academic research)"
MIN_WEIGHT = 100  # skip cells with fewer pre-event jobs than this


def sector_of(naics: str) -> str:
    """Map a 2-6 digit NAICS code to its QCEW sector."""
    two = naics[:2]
    for combo in ("31-33", "44-45", "48-49"):
        lo, hi = combo.split("-")
        if lo <= two <= hi:
            return combo
    return two


def qindex(label: str) -> int:
    """'2019Q1' -> integer quarter index."""
    year, q = label.upper().split("Q")
    return int(year) * 4 + int(q) - 1


def qlabel(idx: int) -> str:
    return f"{idx // 4}Q{idx % 4 + 1}"


# --- QCEW download ----------------------------------------------------------------


def _area_csv(year: int, qtr: int, fips: str) -> str:
    path = RAW_DIR / f"{year}_{qtr}_{fips}.csv"
    if path.exists():
        return path.read_text()
    url = f"https://data.bls.gov/cew/data/api/{year}/{qtr}/area/{fips}.csv"
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                text = r.read().decode()
            break
        except Exception:  # noqa: BLE001
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    time.sleep(0.3)
    return text


def county_employment(year: int, qtr: int, fips: str) -> dict[str, float | None]:
    """Mean monthly employment per sector for one county-quarter (None = suppressed)."""
    out: dict[str, float | None] = {}
    gov_parts: dict[str, float | None] = {}
    for r in csv.DictReader(io.StringIO(_area_csv(year, qtr, fips))):
        months = [float(r[f"month{i}_emplvl"] or 0) for i in (1, 2, 3)]
        value = None if r["disclosure_code"] == "N" else sum(months) / 3
        if r["own_code"] == "5" and r["agglvl_code"] == "74" and r["industry_code"] in PRIVATE_SECTORS:
            out[r["industry_code"]] = value
        elif r["own_code"] in ("1", "2", "3") and r["agglvl_code"] == "71" and r["industry_code"] == "10":
            gov_parts[r["own_code"]] = value
    if gov_parts and all(v is not None for v in gov_parts.values()):
        out["gov"] = sum(gov_parts.values())  # type: ignore[arg-type]
    elif gov_parts:
        out["gov"] = None
    return out


# --- events and truth ---------------------------------------------------------------


@dataclass
class Event:
    id: str
    kind: str                 # backtest | placebo
    shock: dict
    horizon: int
    scope_fips: list[str]     # counties scored
    note: str = ""

    @property
    def start(self) -> int:
        return qindex(self.shock["start"])

    def excluded(self, fips: str, sector: str) -> bool:
        """Directly shocked cells are not scored: propagation is what is tested."""
        for imp in self.shock["impulses"]:
            if sector_of(imp["naics"]) != sector:
                continue
            geo = imp["geo"]
            if geo == "us":
                return True
            if geo.startswith("state:") and fips.startswith(geo.split(":")[1]):
                return True
            if geo.startswith("county:") and fips == geo.split(":")[1]:
                return True
        return False


def load_event(path: Path) -> Event:
    d = json.loads(path.read_text())
    return Event(
        id=d["id"], kind=d.get("kind", "backtest"), shock=d["shock"],
        horizon=int(d["horizon"]), scope_fips=d["scope_fips"], note=d.get("note", ""),
    )


def load_events(ids: list[str] | None = None) -> list[Event]:
    events = [load_event(p) for p in sorted(EVENTS_DIR.glob("*.json"))]
    if ids is not None:
        events = [e for e in events if e.id in ids]
    return events


@dataclass
class TruthCell:
    fips: str
    sector: str
    k: int
    observed: float
    weight: float
    excluded: bool


def build_truth(event: Event) -> list[TruthCell]:
    e = event.start
    quarters = range(e - 8, e + event.horizon + 1)
    cells: list[TruthCell] = []
    for fips in event.scope_fips:
        emp = {t: county_employment(t // 4, t % 4 + 1, fips) for t in quarters}
        for sector in SECTORS:
            series = {t: emp[t].get(sector) for t in quarters}
            pre = [series[t] for t in range(e - 4, e)]
            prev = [series[t] for t in range(e - 8, e - 4)]
            if any(v is None or v <= 0 for v in pre + prev):
                continue
            weight = sum(pre) / 4  # type: ignore[arg-type]
            if weight < MIN_WEIGHT:
                continue
            g = sum(pre) / sum(prev) - 1  # type: ignore[arg-type]
            for k in range(event.horizon + 1):
                t = e + k
                m = k // 4 + 1
                ref = series[t - 4 * m]
                actual = series[t]
                if ref is None or actual is None or ref <= 0:
                    continue
                cf = ref * (1 + g) ** m
                cells.append(TruthCell(
                    fips, sector, k, round(100 * (actual / cf - 1), 4), round(weight, 1),
                    event.excluded(fips, sector),
                ))
    return cells


def write_truth(event: Event, cells: list[TruthCell]) -> Path:
    TRUTH_DIR.mkdir(parents=True, exist_ok=True)
    path = TRUTH_DIR / f"{event.id}.csv"
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["fips", "naics2", "quarter", "observed_pct", "weight", "excluded"])
        for c in cells:
            w.writerow([c.fips, c.sector, c.k, c.observed, c.weight, int(c.excluded)])
    return path


def read_truth(event_id: str) -> list[TruthCell]:
    with (TRUTH_DIR / f"{event_id}.csv").open() as f:
        return [
            TruthCell(r["fips"], r["naics2"], int(r["quarter"]), float(r["observed_pct"]),
                      float(r["weight"]), r["excluded"] == "1")
            for r in csv.DictReader(f)
        ]


def main(argv: list[str] | None = None) -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Build exam ground truth")
    ap.add_argument("events", nargs="*", help="event ids (default: all)")
    args = ap.parse_args(argv)
    for event in load_events(args.events or None):
        cells = build_truth(event)
        path = write_truth(event, cells)
        scored = sum(not c.excluded for c in cells)
        print(f"{event.id}: {len(cells)} cells ({scored} scored) -> {path}")


if __name__ == "__main__":
    main()
