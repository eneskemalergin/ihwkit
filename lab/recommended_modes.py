from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from lab.variants import MethodVariant, run_variant


@dataclass(frozen=True)
class RecommendedMode:
    mode_id: str
    label: str
    variant: MethodVariant
    nfolds: int
    lambdas: str
    parity_case: str | None = None


RECOMMENDED_MODES: tuple[RecommendedMode, ...] = (
    RecommendedMode(
        "ihw_inf_fast",
        "lambda=inf, nfolds=1",
        MethodVariant("ihw_inf_fast", 1, None),
        1,
        "inf",
        "sim_n5000_inf_n1",
    ),
    RecommendedMode(
        "ihw_inf_cv",
        "lambda=inf, nfolds=5",
        MethodVariant("ihw_inf_cv", 5, None),
        5,
        "inf",
        "sim_n5000_inf_n5",
    ),
    RecommendedMode(
        "ihw_auto_fastgrid",
        "lambda in {0, inf}, inner folds=3",
        MethodVariant(
            "ihw_auto_fastgrid",
            5,
            np.array([0.0, np.inf], dtype=np.float64),
            nfolds_internal=3,
        ),
        5,
        "fastgrid",
        None,
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
    adj, wall, diag = run_variant(mode.variant, p, cov, alpha=alpha, seed=seed)
    return int(diag.get("rejections", int(np.sum(adj <= alpha)))), wall
