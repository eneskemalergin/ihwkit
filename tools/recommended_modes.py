from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from ihw import adjust_ihw


@dataclass(frozen=True)
class RecommendedMode:
    """Describe one local diagnostic configuration."""

    mode_id: str
    label: str
    nfolds: int
    lambdas: np.ndarray | None
    nfolds_internal: int = 5


RECOMMENDED_MODES: tuple[RecommendedMode, ...] = (
    RecommendedMode(
        "ihw_inf_fast",
        "lambda=inf, nfolds=1",
        1,
        None,
    ),
    RecommendedMode(
        "ihw_inf_cv",
        "lambda=inf, nfolds=5",
        5,
        None,
    ),
    RecommendedMode(
        "ihw_auto_fastgrid",
        "lambda in {0, inf}, inner folds=3",
        5,
        np.array([0.0, np.inf], dtype=np.float64),
        3,
    ),
)


def run_mode(
    mode: RecommendedMode,
    p: np.ndarray,
    cov: np.ndarray,
    *,
    alpha: float = 0.1,
    seed: int = 42,
) -> tuple[int, float]:
    """Run one recommended mode and return rejections and elapsed seconds."""

    started = time.perf_counter()
    result = adjust_ihw(
        p,
        cov,
        alpha,
        nfolds=mode.nfolds,
        nfolds_internal=mode.nfolds_internal,
        lambdas=mode.lambdas,
        seed=seed,
    )
    return int(np.sum(result.adj_pvalues <= alpha)), time.perf_counter() - started
