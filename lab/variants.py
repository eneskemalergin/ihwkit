from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from ihw import _p_adjust, adjust_ihw


@dataclass(frozen=True)
class MethodVariant:
    variant_id: str
    nfolds: int
    lambdas: np.ndarray | str | None
    nfolds_internal: int = 5
    nsplits_internal: int = 1
    lp_backend: str = "highs"
    is_bh: bool = False


VARIANTS: tuple[MethodVariant, ...] = (
    MethodVariant("bh", 1, None, is_bh=True),
    MethodVariant("ihw_inf_fast", 1, None),
    MethodVariant("ihw_inf_cv", 5, None),
    MethodVariant(
        "ihw_auto_fastgrid",
        5,
        np.array([0.0, np.inf], dtype=np.float64),
        nfolds_internal=3,
    ),
)


def run_variant(
    variant: MethodVariant,
    p: np.ndarray,
    cov: np.ndarray,
    *,
    alpha: float = 0.1,
    seed: int = 42,
) -> tuple[np.ndarray, float, dict]:
    t0 = time.perf_counter()
    diag: dict = {}
    if variant.is_bh:
        adj = _p_adjust(p, "fdr_bh")
        diag["rejections"] = int(np.sum(adj <= alpha))
        return adj, time.perf_counter() - t0, diag
    kw = {
        "nfolds": variant.nfolds,
        "nfolds_internal": variant.nfolds_internal,
        "nsplits_internal": variant.nsplits_internal,
        "lp_backend": variant.lp_backend,
        "seed": seed,
    }
    if variant.lambdas is not None:
        kw["lambdas"] = variant.lambdas
    result = adjust_ihw(p, cov, alpha, **kw)
    adj = result.adj_pvalues
    diag["rejections"] = int(np.sum(adj <= alpha))
    diag["weight_mean"] = float(np.mean(result.weights))
    return adj, time.perf_counter() - t0, diag
