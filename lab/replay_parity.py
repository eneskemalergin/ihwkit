#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from ihw import _numba_importable, _p_adjust, _safe_divide, adjust_ihw
from lab.parity_cases import available_r_gold_cases

RESULTS_DIR = ROOT / "tmp" / "results"
ATOL = 1e-8
RTOL = 1e-6


@dataclass
class ReplayRow:
    case_id: str
    r_rejections: int
    highs_rejections: int
    highs_delta: int
    highs_max_adj: float
    highs_max_w: float
    highs_ok: bool
    bh_oracle_ok: bool
    bh_max_adj: float
    numpy_rejections: int | None
    numpy_max_adj: float | None
    numpy_numba_rejections: int | None
    numpy_numba_max_adj: float | None
    error: str | None = None


def _max_abs(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a - b)))


def _allclose(a: np.ndarray, b: np.ndarray) -> bool:
    return bool(np.allclose(a, b, atol=ATOL, rtol=RTOL))


def replay_case(case) -> ReplayRow:
    data = np.load(case.oracle_path)
    p = np.asarray(data["p"], dtype=np.float64)
    x = np.asarray(data["x"], dtype=np.float64)
    groups = np.asarray(data["groups"], dtype=np.intp)
    r_adj = np.asarray(data["adj_pvalues"], dtype=np.float64)
    r_w = np.asarray(data["weights"], dtype=np.float64)
    r_rej = int(data["rejections"])
    kw = dict(case.replay_kwargs())
    kw["groups"] = groups
    if "folds" in data.files:
        kw["folds"] = np.asarray(data["folds"], dtype=np.intp)
    try:
        bh_adj = _p_adjust(_safe_divide(p, r_w), "fdr_bh")
        bh_ok = int(np.sum(bh_adj <= case.alpha)) == r_rej and _allclose(bh_adj, r_adj)
        bh_err = _max_abs(bh_adj, r_adj)

        highs = adjust_ihw(p, x, case.alpha, **kw)
        highs_rej = int(np.sum(highs.adj_pvalues <= case.alpha))
        highs_ok = (
            highs_rej == r_rej
            and _allclose(highs.adj_pvalues, r_adj)
            and _allclose(highs.weights, r_w)
        )

        numpy_kw = dict(kw)
        numpy_kw["lp_backend"] = "numpy"
        numpy_kw["use_numba"] = False
        numpy_fit = adjust_ihw(p, x, case.alpha, **numpy_kw)
        numpy_rej = int(np.sum(numpy_fit.adj_pvalues <= case.alpha))
        numpy_err = _max_abs(numpy_fit.adj_pvalues, r_adj)

        numba_rej = None
        numba_err = None
        if _numba_importable():
            numba_kw = dict(kw)
            numba_kw["lp_backend"] = "numpy"
            numba_kw["use_numba"] = True
            numba_fit = adjust_ihw(p, x, case.alpha, **numba_kw)
            numba_rej = int(np.sum(numba_fit.adj_pvalues <= case.alpha))
            numba_err = _max_abs(numba_fit.adj_pvalues, r_adj)

        return ReplayRow(
            case_id=case.case_id,
            r_rejections=r_rej,
            highs_rejections=highs_rej,
            highs_delta=highs_rej - r_rej,
            highs_max_adj=_max_abs(highs.adj_pvalues, r_adj),
            highs_max_w=_max_abs(highs.weights, r_w),
            highs_ok=highs_ok,
            bh_oracle_ok=bh_ok,
            bh_max_adj=bh_err,
            numpy_rejections=numpy_rej,
            numpy_max_adj=numpy_err,
            numpy_numba_rejections=numba_rej,
            numpy_numba_max_adj=numba_err,
        )
    except Exception as exc:
        return ReplayRow(
            case_id=case.case_id,
            r_rejections=r_rej,
            highs_rejections=-1,
            highs_delta=0,
            highs_max_adj=float("nan"),
            highs_max_w=float("nan"),
            highs_ok=False,
            bh_oracle_ok=False,
            bh_max_adj=float("nan"),
            numpy_rejections=None,
            numpy_max_adj=None,
            numpy_numba_rejections=None,
            numpy_numba_max_adj=None,
            error=str(exc),
        )


def _fmt(x: float | None) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "nan"
    return f"{x:.2e}"


def _write_markdown(rows: list[ReplayRow], path: Path) -> None:
    lines = [
        "# IHW replay parity (frozen R oracles)",
        "",
        "Python-only: frozen groups/folds from stored IHW::ihw lambda=inf gold.",
        "HiGHS is the quality line. numpy and numpy_numba are informational.",
        "",
        "| case | R rej | highs rej | delta | max abs w | highs ok | BH@R | numpy rej | numba rej |",
        "|---|---:|---:|---:|---:|:---:|:---:|---:|---:|",
    ]
    for row in rows:
        if row.error:
            lines.append(
                f"| {row.case_id} | {row.r_rejections} | err |  |  | no | no |  |  |"
            )
            continue
        np_s = "nan" if row.numpy_rejections is None else str(row.numpy_rejections)
        nb_s = (
            "nan" if row.numpy_numba_rejections is None else str(row.numpy_numba_rejections)
        )
        lines.append(
            f"| {row.case_id} | {row.r_rejections} | {row.highs_rejections} | "
            f"{row.highs_delta:+d} | {_fmt(row.highs_max_w)} | "
            f"{'yes' if row.highs_ok else 'no'} | {'yes' if row.bh_oracle_ok else 'no'} | "
            f"{np_s} | {nb_s} |"
        )
    lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
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
        status = "PASS" if row.highs_ok and row.bh_oracle_ok else "FAIL"
        print(
            f"  {status} R={row.r_rejections} highs={row.highs_rejections} "
            f"delta={row.highs_delta:+d} BH@R={'yes' if row.bh_oracle_ok else 'no'} "
            f"numpy_max_adj={_fmt(row.numpy_max_adj)} "
            f"numba_max_adj={_fmt(row.numpy_numba_max_adj)}"
        )
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = RESULTS_DIR / "ihw_replay_parity.json"
    md_path = RESULTS_DIR / "ihw_replay_parity.md"
    payload = {
        "rows": [asdict(r) for r in rows],
        "pass": sum(1 for r in rows if r.highs_ok and r.bh_oracle_ok and r.error is None),
        "fail": sum(1 for r in rows if not (r.highs_ok and r.bh_oracle_ok) or r.error),
    }
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    _write_markdown(rows, md_path)
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    ok = all(r.highs_ok and r.bh_oracle_ok and r.error is None for r in rows)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
