#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from lab.parity_cases import available_r_gold_cases
from lab.recommended_modes import RECOMMENDED_MODES, run_mode
from lab.replay_parity import replay_case
from lab.simulators import global_null

RESULTS_DIR = ROOT / "tmp" / "results"
AIRWAY_CANDIDATES = (
    ROOT / "data" / "correction" / "airway.csv",
    ROOT / "tmp" / "airway.csv",
)
ALPHA = 0.1


def _airway_path() -> Path | None:
    for path in AIRWAY_CANDIDATES:
        if path.is_file():
            return path
    return None


def _load_airway(path: Path) -> tuple[np.ndarray, np.ndarray]:
    import csv

    pvals: list[float] = []
    covs: list[float] = []
    with path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            pvals.append(float(row["pvalue"]))
            covs.append(float(row["basemean"]))
    return np.asarray(pvals, dtype=np.float64), np.asarray(covs, dtype=np.float64)


def _check_replay() -> list[dict]:
    rows = []
    for case in available_r_gold_cases():
        row = replay_case(case)
        rows.append(
            {
                "case_id": row.case_id,
                "ok": bool(row.highs_ok and row.bh_oracle_ok and row.error is None),
                "r_rejections": row.r_rejections,
                "py_rejections": row.highs_rejections,
                "delta": row.highs_delta,
                "bh_oracle_ok": row.bh_oracle_ok,
                "error": row.error,
            }
        )
    return rows


def _check_null_smoke(n: int = 3000, reps: int = 8) -> dict:
    mode = next(m for m in RECOMMENDED_MODES if m.mode_id == "ihw_inf_fast")
    rates: list[float] = []
    for rep in range(reps):
        draw = global_null(n, seed=7000 + rep)
        rej, _ = run_mode(mode, draw.pvalues, draw.covariates, alpha=ALPHA, seed=7000 + rep)
        rates.append(rej / n)
    max_rej_rate = float(np.max(rates))
    return {
        "reps": reps,
        "n": n,
        "max_rej_rate": max_rej_rate,
        "ok": max_rej_rate < ALPHA / 10,
    }


def _check_modes_airway() -> list[dict] | None:
    path = _airway_path()
    if path is None:
        return None
    p, cov = _load_airway(path)
    rows = []
    for mode in RECOMMENDED_MODES:
        rej, wall = run_mode(mode, p, cov, alpha=ALPHA, seed=42)
        rows.append(
            {
                "mode_id": mode.mode_id,
                "label": mode.label,
                "rejections": rej,
                "wall_s": wall,
                "ok": True,
            }
        )
    return rows


def _write_md(payload: dict, path: Path) -> None:
    lines = [
        "# IHW recommended checks",
        "",
        "Lab-only gate. Does not change adjust_ihw defaults.",
        "",
        "## Oracle replay",
        "",
        "| case | ok | R | highs | delta | BH@R |",
        "|---|:---:|---:|---:|---:|:---:|",
    ]
    for row in payload["replay"]:
        ok = "yes" if row.get("ok") else "no"
        bh = "yes" if row.get("bh_oracle_ok") else "no"
        if row.get("error"):
            lines.append(f"| {row['case_id']} | no |  |  | {row['error']} | no |")
        else:
            lines.append(
                f"| {row['case_id']} | {ok} | {row['r_rejections']} | "
                f"{row['py_rejections']} | {row['delta']:+d} | {bh} |"
            )
    null = payload["null_smoke"]
    lines.extend(
        [
            "",
            "## Null smoke",
            "",
            f"- max rej/n over {null['reps']} uniform-null reps: {null['max_rej_rate']:.5f} "
            f"(pass: {'yes' if null['ok'] else 'no'})",
            "",
            "## Airway",
            "",
        ]
    )
    airway = payload["airway"]
    if airway is None:
        lines.append("skipped (airway.csv not present)")
    else:
        lines.append("| mode | rejections | time (s) |")
        lines.append("|---|---:|---:|")
        for row in airway:
            lines.append(
                f"| {row['mode_id']} | {row['rejections']} | {row['wall_s']:.3f} |"
            )
    lines.extend(
        [
            "",
            f"Overall: {'PASS' if payload['all_ok'] else 'FAIL'}",
            "",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    replay_rows = _check_replay()
    null_smoke = _check_null_smoke()
    airway = _check_modes_airway()
    all_ok = all(r.get("ok") for r in replay_rows) and null_smoke["ok"]
    payload = {
        "replay": replay_rows,
        "null_smoke": null_smoke,
        "airway": airway,
        "all_ok": all_ok,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = RESULTS_DIR / "ihw_recommended_checks.json"
    md_path = RESULTS_DIR / "ihw_recommended_checks.md"
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    _write_md(payload, md_path)
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    print(f"Overall: {'PASS' if all_ok else 'FAIL'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
