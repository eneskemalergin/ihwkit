"""Generate small synthetic draws for local validity studies."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from math import erf, sqrt

import numpy as np


@dataclass(frozen=True)
class SimDraw:
    """Store p-values, covariates, truth labels, and simulation metadata."""

    pvalues: np.ndarray
    covariates: np.ndarray
    is_null: np.ndarray
    scenario: str
    seed: int


def _normal_survival(values: np.ndarray) -> np.ndarray:
    """Return the standard normal survival function for a one-dimensional array."""

    return np.fromiter(
        (0.5 * (1.0 - erf(float(value) / sqrt(2.0))) for value in values),
        dtype=np.float64,
        count=values.size,
    )


def global_null(n: int, *, seed: int = 0) -> SimDraw:
    """Generate a global-null draw with independent uniform p-values.

    Parameters
    ----------
    n : int
        Number of hypotheses.
    seed : int, optional
        Random seed.

    Returns
    -------
    SimDraw
        Simulated p-values, covariates, and all-null labels.
    """

    rng = np.random.default_rng(seed)
    p = rng.uniform(0.0, 1.0, size=n)
    cov = rng.uniform(0.0, 3.0, size=n)
    return SimDraw(
        pvalues=p.astype(np.float64),
        covariates=cov.astype(np.float64),
        is_null=np.ones(n, dtype=np.bool_),
        scenario="global_null",
        seed=seed,
    )


def null_with_covariate(n: int, *, seed: int = 0) -> SimDraw:
    """Generate null p-values with a rounded, skewed covariate."""

    rng = np.random.default_rng(seed)
    p = rng.uniform(0.0, 1.0, size=n)
    cov = np.exp(rng.normal(5.0, 2.0, size=n))
    cov = np.round(cov, 4)
    return SimDraw(
        pvalues=p.astype(np.float64),
        covariates=cov.astype(np.float64),
        is_null=np.ones(n, dtype=np.bool_),
        scenario="null_covariate",
        seed=seed,
    )


def mixture(
    n: int,
    *,
    pi0: float = 0.9,
    seed: int = 0,
    signal_frac: float | None = None,
) -> SimDraw:
    """Generate a mixture of null and covariate-dependent alternatives."""

    rng = np.random.default_rng(seed)
    alt_frac = 1.0 - pi0 if signal_frac is None else signal_frac
    alt_frac = float(np.clip(alt_frac, 0.0, 1.0))
    is_null = rng.uniform(size=n) >= alt_frac
    cov = rng.uniform(0.0, 3.0, size=n)
    p = np.empty(n, dtype=np.float64)
    p[is_null] = rng.uniform(0.0, 1.0, size=int(is_null.sum()))
    n_alt = int((~is_null).sum())
    if n_alt:
        z = rng.normal(loc=cov[~is_null])
        p[~is_null] = _normal_survival(z)
    return SimDraw(
        pvalues=p,
        covariates=cov.astype(np.float64),
        is_null=is_null,
        scenario=f"mixture_pi0_{pi0:.2f}",
        seed=seed,
    )


def ignatiadis(n: int, *, seed: int = 0, signal_frac: float = 0.12) -> SimDraw:
    """Generate the covariate-dependent alternative simulation."""

    rng = np.random.default_rng(seed)
    cov = rng.uniform(0.0, 3.0, size=n)
    signals = rng.binomial(1, signal_frac, size=n).astype(np.bool_)
    z = rng.normal(loc=signals * cov)
    p = _normal_survival(z)
    return SimDraw(
        pvalues=p.astype(np.float64),
        covariates=cov.astype(np.float64),
        is_null=~signals,
        scenario="ignatiadis",
        seed=seed,
    )


SCENARIO_BUILDERS: dict[str, Callable[[int, int], SimDraw]] = {
    "global_null": lambda n, seed: global_null(n, seed=seed),
    "null_covariate": lambda n, seed: null_with_covariate(n, seed=seed),
    "mixture_mild": lambda n, seed: mixture(n, pi0=0.9, seed=seed),
    "mixture_sparse": lambda n, seed: mixture(n, pi0=0.95, seed=seed),
    "ignatiadis": lambda n, seed: ignatiadis(n, seed=seed),
}
