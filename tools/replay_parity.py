from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from ihw import _p_adjust, _safe_divide, adjust_ihw
from tools.data_contract import load_oracle
from tools.parity_cases import ParityCase, available_r_gold_cases

RESULTS_DIR = ROOT / "tmp" / "results"
ATOL = 1e-8
RTOL = 1e-6


@dataclass
class ReplayRow:
    """Report weighted-BH and production results for one oracle case."""

    case_id: str
    r_rejections: int
    production_rejections: int
    production_delta: int
    production_max_adj: float
    production_max_w: float
    production_ok: bool
    bh_oracle_ok: bool
    bh_max_adj: float
    error: str | None = None


def _max_abs(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a - b)))


def _allclose(a: np.ndarray, b: np.ndarray) -> bool:
    return bool(np.allclose(a, b, atol=ATOL, rtol=RTOL))


def replay_case(case: ParityCase) -> ReplayRow:
    """Replay one frozen R oracle through the production implementation."""

    record = load_oracle(case.oracle_id, root=ROOT)
    peer_input = record.peer_input
    if peer_input.groups is None or peer_input.folds is None:
        raise RuntimeError(f"oracle {case.oracle_id!r} has no frozen partition")
    nbins = int(np.max(peer_input.groups)) + 1
    nfolds = int(np.max(peer_input.folds)) + 1
    replay_kwargs: dict[str, object] = {
        "nbins": nbins,
        "nfolds": nfolds,
        "groups": peer_input.groups,
        "folds": peer_input.folds,
        "seed": peer_input.seed if peer_input.seed is not None else 1,
    }
    if peer_input.fold_lambdas is not None:
        replay_kwargs["fold_lambdas"] = peer_input.fold_lambdas
    r_adj = record.adjusted_pvalues
    r_w = record.weights
    r_rej = record.r_rejections
    p = peer_input.pvalues
    try:
        bh_adj = _p_adjust(_safe_divide(p, r_w), "fdr_bh")
        bh_ok = int(np.sum(bh_adj <= 0.1)) == r_rej and _allclose(bh_adj, r_adj)
        bh_err = _max_abs(bh_adj, r_adj)

        production = adjust_ihw(p, peer_input.covariates, 0.1, **replay_kwargs)
        production_rej = int(np.sum(production.adj_pvalues <= 0.1))
        production_ok = (
            production_rej == r_rej
            and _allclose(production.adj_pvalues, r_adj)
            and _allclose(production.weights, r_w)
        )

        return ReplayRow(
            case_id=case.case_id,
            r_rejections=r_rej,
            production_rejections=production_rej,
            production_delta=production_rej - r_rej,
            production_max_adj=_max_abs(production.adj_pvalues, r_adj),
            production_max_w=_max_abs(production.weights, r_w),
            production_ok=production_ok,
            bh_oracle_ok=bh_ok,
            bh_max_adj=bh_err,
        )
    except Exception as exc:  # noqa: BLE001 - replay must record every fit failure
        return ReplayRow(
            case_id=case.case_id,
            r_rejections=r_rej,
            production_rejections=-1,
            production_delta=0,
            production_max_adj=float("nan"),
            production_max_w=float("nan"),
            production_ok=False,
            bh_oracle_ok=False,
            bh_max_adj=float("nan"),
            error=str(exc),
        )


def _fmt(x: float | None) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "nan"
    return f"{x:.2e}"


def _write_markdown(rows: list[ReplayRow], path: Path) -> None:
    """Write the replay rows as a compact metrics table."""

    lines = [
        "# IHW replay parity (frozen R oracles)",
        "",
        "Python-only: frozen groups/folds from stored IHW::ihw lambda=inf gold.",
        "The production NumPy+Numba path is compared with the stored R outputs.",
        "",
        "| case | R rej | production rej | delta | max abs w | production ok | BH@R |",
        "|---|---:|---:|---:|---:|:---:|:---:|",
    ]
    for row in rows:
        if row.error:
            lines.append(f"| {row.case_id} | {row.r_rejections} | err |  |  | no | no |")
            continue
        lines.append(
            f"| {row.case_id} | {row.r_rejections} | {row.production_rejections} | "
            f"{row.production_delta:+d} | {_fmt(row.production_max_w)} | "
            f"{'yes' if row.production_ok else 'no'} | "
            f"{'yes' if row.bh_oracle_ok else 'no'} |"
        )
    lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    """Replay all available frozen R oracle cases."""

    cases = available_r_gold_cases()
    if not cases:
        print("no frozen R gold cases on disk")
        return 2
    rows: list[ReplayRow] = []
    for case in cases:
        print(f"replay {case.case_id}...")
        row = replay_case(case)
        rows.append(row)
        if row.error:
            print(f"  ERR {row.error}")
            continue
        status = "PASS" if row.production_ok and row.bh_oracle_ok else "FAIL"
        print(
            f"  {status} R={row.r_rejections} production={row.production_rejections} "
            f"delta={row.production_delta:+d} BH@R={'yes' if row.bh_oracle_ok else 'no'} "
            f"max_adj={_fmt(row.production_max_adj)} "
            f"max_w={_fmt(row.production_max_w)}"
        )
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = RESULTS_DIR / "ihw_replay_parity.json"
    md_path = RESULTS_DIR / "ihw_replay_parity.md"
    payload = {
        "rows": [asdict(r) for r in rows],
        "pass": sum(
            1 for r in rows if r.production_ok and r.bh_oracle_ok and r.error is None
        ),
        "fail": sum(
            1
            for r in rows
            if not (r.production_ok and r.bh_oracle_ok) or r.error
        ),
    }
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    _write_markdown(rows, md_path)
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    ok = all(r.production_ok and r.bh_oracle_ok and r.error is None for r in rows)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
