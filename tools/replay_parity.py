"""Replay fixed R references through the changing production method."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from ihw import _p_adjust, _safe_divide, adjust_ihw
from tools.peer import PARITY_GATE_IDS, REFERENCE_IDS, load_reference
from tools.simulators import SCENARIO_BUILDERS

ATOL = 1e-8
RTOL = 1e-6


@dataclass
class ReplayRow:
    """Report weighted-BH and production results for one fixed R reference."""

    reference_id: str
    dataset_id: str
    gate: bool
    r_rejections: int
    production_rejections: int | None
    production_delta: int | None
    production_max_adj: float | None
    production_max_w: float | None
    production_ok: bool
    bh_reference_ok: bool
    bh_max_adj: float | None
    error: str | None = None


@dataclass
class StressRow:
    """Record production behavior on one generated numerical stress case."""

    case_id: str
    scenario_id: str
    n: int
    seed: int
    nfolds: int
    status: str
    rejections: int | None
    false_rejections: int | None
    fdp: float | None
    weight_mean: float | None
    error: str | None


def _max_abs(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a - b)))


def _allclose(a: np.ndarray, b: np.ndarray) -> bool:
    return bool(np.allclose(a, b, atol=ATOL, rtol=RTOL))


def replay_case(reference_id: str) -> ReplayRow:
    """Replay one fixed R reference through the production implementation."""

    record = load_reference(reference_id)
    peer_input = record.peer_input
    if peer_input.groups is None or peer_input.folds is None:
        raise RuntimeError(f"reference {reference_id!r} has no fixed partition")
    replay_kwargs: dict[str, object] = {
        "nbins": int(np.max(peer_input.groups)) + 1,
        "nfolds": int(np.max(peer_input.folds)) + 1,
        "groups": peer_input.groups,
        "folds": peer_input.folds,
        "seed": peer_input.seed if peer_input.seed is not None else 1,
    }
    r_adj = record.adjusted_pvalues
    r_weights = record.weights
    r_rejections = record.r_rejections
    pvalues = peer_input.pvalues

    bh_adjusted = _p_adjust(_safe_divide(pvalues, r_weights), "fdr_bh")
    bh_ok = int(np.sum(bh_adjusted <= record.spec.alpha)) == r_rejections and _allclose(
        bh_adjusted, r_adj
    )
    bh_error = _max_abs(bh_adjusted, r_adj)

    try:
        production = adjust_ihw(
            pvalues,
            peer_input.covariates,
            record.spec.alpha,
            **replay_kwargs,
        )
    except Exception as exc:  # noqa: BLE001 - a failed fit is benchmark evidence
        return ReplayRow(
            reference_id=reference_id,
            dataset_id=record.spec.dataset_id,
            gate=record.spec.gate,
            r_rejections=r_rejections,
            production_rejections=None,
            production_delta=None,
            production_max_adj=None,
            production_max_w=None,
            production_ok=False,
            bh_reference_ok=bh_ok,
            bh_max_adj=bh_error,
            error=f"{type(exc).__name__}: {exc}",
        )
    production_rejections = int(np.sum(production.adj_pvalues <= record.spec.alpha))
    production_ok = (
        production_rejections == r_rejections
        and _allclose(production.adj_pvalues, r_adj)
        and _allclose(production.weights, r_weights)
    )
    return ReplayRow(
        reference_id=reference_id,
        dataset_id=record.spec.dataset_id,
        gate=record.spec.gate,
        r_rejections=r_rejections,
        production_rejections=production_rejections,
        production_delta=production_rejections - r_rejections,
        production_max_adj=_max_abs(production.adj_pvalues, r_adj),
        production_max_w=_max_abs(production.weights, r_weights),
        production_ok=production_ok,
        bh_reference_ok=bh_ok,
        bh_max_adj=bh_error,
    )


def run_known_stress_case() -> StressRow:
    """Exercise the smallest known generated false-infeasibility case."""

    scenario_id = "mixture_mild"
    n = 3_000
    seed = 2_034
    draw = SCENARIO_BUILDERS[scenario_id](n, seed)
    try:
        result = adjust_ihw(
            draw.pvalues,
            draw.covariates,
            0.1,
            nfolds=5,
            seed=seed,
        )
    except Exception as exc:  # noqa: BLE001 - a failed fit is benchmark evidence
        return StressRow(
            case_id="mixture_mild_n3000_seed2034_n5",
            scenario_id=scenario_id,
            n=n,
            seed=seed,
            nfolds=5,
            status="error",
            rejections=None,
            false_rejections=None,
            fdp=None,
            weight_mean=None,
            error=f"{type(exc).__name__}: {exc}",
        )
    rejected = result.adj_pvalues <= 0.1
    rejections = int(np.sum(rejected))
    false_rejections = int(np.sum(rejected & draw.is_null))
    return StressRow(
        case_id="mixture_mild_n3000_seed2034_n5",
        scenario_id=scenario_id,
        n=n,
        seed=seed,
        nfolds=5,
        status="ok",
        rejections=rejections,
        false_rejections=false_rejections,
        fdp=false_rejections / max(1, rejections),
        weight_mean=float(np.mean(result.weights)),
        error=None,
    )


def _format_number(value: float | None) -> str:
    return "" if value is None else f"{value:.2e}"


def _write_markdown(
    rows: list[ReplayRow], stress_rows: list[StressRow], path: Path
) -> None:
    lines = [
        "# IHW fixed-reference replay",
        "",
        "The stored R groups and folds are replayed through the current Python implementation. BH@R independently checks weighted BH using the stored R weights. Synthetic rows are release gates; airway rows are visible real-shape diagnostics.",
        "",
        "| reference | data | gate | R rej | production rej | delta | max abs w | production | BH@R |",
        "|---|---|:---:|---:|---:|---:|---:|:---:|:---:|",
    ]
    for row in rows:
        production_rejections = (
            "error" if row.production_rejections is None else row.production_rejections
        )
        delta = "" if row.production_delta is None else f"{row.production_delta:+d}"
        lines.append(
            f"| {row.reference_id} | {row.dataset_id} | "
            f"{'yes' if row.gate else 'no'} | {row.r_rejections} | "
            f"{production_rejections} | {delta} | "
            f"{_format_number(row.production_max_w)} | "
            f"{'yes' if row.production_ok else 'no'} | "
            f"{'yes' if row.bh_reference_ok else 'no'} |"
        )
    lines.append("")
    if stress_rows:
        lines.extend(
            [
                "## Generated numerical stress",
                "",
                "| case | status | rejections | FDP | mean weight | error |",
                "|---|:---:|---:|---:|---:|---|",
            ]
        )
        for row in stress_rows:
            rejections = "" if row.rejections is None else str(row.rejections)
            fdp = "" if row.fdp is None else f"{row.fdp:.4f}"
            weight_mean = "" if row.weight_mean is None else f"{row.weight_mean:.8f}"
            lines.append(
                f"| {row.case_id} | {row.status} | {rejections} | {fdp} | "
                f"{weight_mean} | {row.error or ''} |"
            )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: Sequence[str] | None = None, *, gates_only: bool = True) -> int:
    """Replay the parity gates or the complete robustness reference set."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, default=Path("tmp/results"))
    args = parser.parse_args(argv)
    result_dir = (ROOT / args.result_dir).resolve()
    if not result_dir.is_relative_to((ROOT / "tmp").resolve()):
        parser.error("--result-dir must be under tmp")
    reference_ids = PARITY_GATE_IDS if gates_only else REFERENCE_IDS
    rows: list[ReplayRow] = []
    for reference_id in reference_ids:
        print(f"replay {reference_id}...")
        row = replay_case(reference_id)
        rows.append(row)
        status = "PASS" if row.production_ok and row.bh_reference_ok else "FAIL"
        detail = f" error={row.error}" if row.error else ""
        print(
            f"  {status} R={row.r_rejections} "
            f"production={row.production_rejections} "
            f"BH@R={'yes' if row.bh_reference_ok else 'no'}{detail}"
        )
    stress_rows = [] if gates_only else [run_known_stress_case()]
    for row in stress_rows:
        detail = f" error={row.error}" if row.error else ""
        print(f"stress {row.case_id}: {row.status}{detail}")
    result_dir.mkdir(parents=True, exist_ok=True)
    stem = "ihw_replay_parity" if gates_only else "ihw_replay_robustness"
    json_path = result_dir / f"{stem}.json"
    markdown_path = result_dir / f"{stem}.md"
    payload = {
        "scope": "parity gates" if gates_only else "all robustness diagnostics",
        "rows": [asdict(row) for row in rows],
        "stress_rows": [asdict(row) for row in stress_rows],
        "pass": sum(row.production_ok and row.bh_reference_ok for row in rows)
        + sum(row.status == "ok" for row in stress_rows),
        "fail": sum(not (row.production_ok and row.bh_reference_ok) for row in rows)
        + sum(row.status != "ok" for row in stress_rows),
    }
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    _write_markdown(rows, stress_rows, markdown_path)
    print(f"wrote {json_path.relative_to(ROOT)}")
    print(f"wrote {markdown_path.relative_to(ROOT)}")
    rows_to_enforce = [row for row in rows if row.gate] if gates_only else rows
    passed = all(
        row.production_ok and row.bh_reference_ok for row in rows_to_enforce
    ) and all(row.status == "ok" for row in stress_rows)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
