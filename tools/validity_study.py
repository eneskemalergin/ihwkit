from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from ihw import _p_adjust, adjust_ihw
from tools.simulators import SCENARIO_BUILDERS

RESULTS_DIR = ROOT / "tmp" / "results"
ALPHA = 0.1
BASE_SEED = 2026


@dataclass(frozen=True)
class ScenarioSpec:
    """Describe one simulated validity-study scenario."""

    scenario_id: str
    n: int
    reps: int
    has_alternatives: bool


@dataclass
class AggregateRow:
    """Store aggregate validity and runtime metrics for one method."""

    scenario_id: str
    variant_id: str
    n: int
    reps: int
    mean_fdp: float
    se_fdp: float
    frac_fdp_above_alpha: float
    mean_null_rej_rate: float
    mean_power: float
    mean_rejections: float
    median_wall_s: float


def _fdp(adj: np.ndarray, is_null: np.ndarray, alpha: float) -> float:
    rej = adj <= alpha
    r = int(rej.sum())
    if r == 0:
        return 0.0
    v = int(np.logical_and(rej, is_null).sum())
    return v / r


def _power(adj: np.ndarray, is_null: np.ndarray, alpha: float) -> float:
    alt = ~is_null
    n_alt = int(alt.sum())
    if n_alt == 0:
        return float("nan")
    return float(np.logical_and(adj <= alpha, alt).sum()) / n_alt


def _run_bh(p: np.ndarray) -> tuple[np.ndarray, float]:
    t0 = time.perf_counter()
    adj = _p_adjust(p, "fdr_bh")
    return adj, time.perf_counter() - t0


def _run_ihw(p: np.ndarray, x: np.ndarray, seed: int) -> tuple[np.ndarray, float]:
    t0 = time.perf_counter()
    result = adjust_ihw(p, x, ALPHA, nfolds=5, seed=seed)
    return result.adj_pvalues, time.perf_counter() - t0


def _scenarios(*, quick: bool) -> list[ScenarioSpec]:
    n = 3000
    reps = 5 if quick else 15
    return [
        ScenarioSpec("global_null", n, reps, False),
        ScenarioSpec("null_covariate", n, reps, False),
        ScenarioSpec("mixture_mild", n, reps, True),
        ScenarioSpec("ignatiadis", n, reps, True),
    ]


def _aggregate(
    spec: ScenarioSpec,
    variant_id: str,
    fdps: list[float],
    powers: list[float],
    rejs: list[int],
    walls: list[float],
    n: int,
) -> AggregateRow:
    f = np.asarray(fdps, dtype=np.float64)
    se = float(np.std(f, ddof=1) / np.sqrt(len(f))) if len(f) > 1 else 0.0
    pows = np.asarray(powers, dtype=np.float64)
    mean_power = (
        float(np.nanmean(pows)) if np.any(np.isfinite(pows)) else float("nan")
    )
    return AggregateRow(
        scenario_id=spec.scenario_id,
        variant_id=variant_id,
        n=spec.n,
        reps=len(fdps),
        mean_fdp=float(np.mean(f)),
        se_fdp=se,
        frac_fdp_above_alpha=float(np.mean(f > ALPHA * 1.05)),
        mean_null_rej_rate=float(np.mean(np.asarray(rejs, dtype=np.float64) / n)),
        mean_power=mean_power,
        mean_rejections=float(np.mean(rejs)),
        median_wall_s=float(np.median(walls)),
    )


def _write_markdown(rows: list[AggregateRow], path: Path) -> None:
    """Write aggregate study metrics as Markdown."""

    lines = [
        "# IHW statistical validity study",
        "",
        f"Target FDR alpha = {ALPHA}. BH vs production IHW lambda=inf, nfolds=5.",
        "FDP is false discoveries / rejections per replicate.",
        "",
    ]
    for sid in sorted({r.scenario_id for r in rows}):
        lines.append(f"## {sid}")
        lines.append("")
        lines.append(
            "| variant | mean FDP | SE | frac FDP>alpha | mean rej | power | med time (s) |"
        )
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for row in rows:
            if row.scenario_id != sid:
                continue
            pow_s = f"{row.mean_power:.3f}" if np.isfinite(row.mean_power) else "nan"
            lines.append(
                f"| {row.variant_id} | {row.mean_fdp:.4f} | {row.se_fdp:.4f} | "
                f"{row.frac_fdp_above_alpha:.3f} | {row.mean_rejections:.1f} | "
                f"{pow_s} | {row.median_wall_s:.3f} |"
            )
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_study(*, quick: bool) -> list[AggregateRow]:
    """Run the configured validity study."""

    rows: list[AggregateRow] = []
    for spec in _scenarios(quick=quick):
        for variant_id in ("bh", "ihw_inf_cv"):
            print(f"  {spec.scenario_id} x {variant_id} ({spec.reps} reps)...")
            fdps: list[float] = []
            powers: list[float] = []
            rejs: list[int] = []
            walls: list[float] = []
            for rep in range(spec.reps):
                seed = BASE_SEED + rep
                draw = SCENARIO_BUILDERS[spec.scenario_id](spec.n, seed)
                if variant_id == "bh":
                    adj, wall = _run_bh(draw.pvalues)
                else:
                    adj, wall = _run_ihw(draw.pvalues, draw.covariates, seed)
                fdps.append(_fdp(adj, draw.is_null, ALPHA))
                powers.append(_power(adj, draw.is_null, ALPHA))
                rejs.append(int(np.sum(adj <= ALPHA)))
                walls.append(wall)
            row = _aggregate(spec, variant_id, fdps, powers, rejs, walls, spec.n)
            rows.append(row)
            pow_s = f"{row.mean_power:.3f}" if np.isfinite(row.mean_power) else "nan"
            print(
                f"    mean_fdp={row.mean_fdp:.4f} power={pow_s} "
                f"mean_rej={row.mean_rejections:.1f}"
            )
    return rows


def main() -> int:
    """Run the validity study and write JSON and Markdown reports."""

    quick = "--quick" in sys.argv
    print("Running validity study...")
    rows = run_study(quick=quick)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = RESULTS_DIR / "ihw_validity_study.json"
    md_path = RESULTS_DIR / "ihw_validity_study.md"
    payload = {"alpha": ALPHA, "quick": quick, "rows": [asdict(r) for r in rows]}
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    _write_markdown(rows, md_path)
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
