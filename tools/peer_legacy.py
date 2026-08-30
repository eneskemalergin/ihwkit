"""Retained SciPy/HiGHS baseline for benchmark comparisons.

This is deliberately not a second ihwkit implementation. It preserves the
older dense linear-program route so the benchmark can compare the optimized
NumPy method with a genuinely different solver. Production code never imports
this module.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__version__ = "ihwkit-0.1.0-scipy-baseline"


@dataclass(frozen=True)
class IHWResult:
    """Only the normalized fields consumed by the peer runner."""

    adj_pvalues: np.ndarray
    weights: np.ndarray


def _isotonic_mean(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    size = values.shape[0]
    block_values = np.empty(size, dtype=np.float64)
    block_weights = np.empty(size, dtype=np.float64)
    block_counts = np.empty(size, dtype=np.intp)
    blocks = 0
    for index in range(size):
        block_values[blocks] = values[index]
        block_weights[blocks] = weights[index]
        block_counts[blocks] = 1
        blocks += 1
        while blocks >= 2 and block_values[blocks - 2] > block_values[blocks - 1]:
            total = block_weights[blocks - 2] + block_weights[blocks - 1]
            block_values[blocks - 2] = (
                block_weights[blocks - 2] * block_values[blocks - 2]
                + block_weights[blocks - 1] * block_values[blocks - 1]
            ) / total
            block_weights[blocks - 2] = total
            block_counts[blocks - 2] += block_counts[blocks - 1]
            blocks -= 1
    result = np.empty(size, dtype=np.float64)
    position = 0
    for block in range(blocks):
        count = int(block_counts[block])
        result[position : position + count] = block_values[block]
        position += count
    return result


def _grenander(
    sorted_pvalues: np.ndarray, m_total: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if sorted_pvalues.size == 0:
        return np.array([0.0]), np.array([0.0]), np.array([1.0])
    unique_p, counts = np.unique(sorted_pvalues, return_counts=True)
    ecdf = np.cumsum(counts, dtype=np.float64) / float(m_total)
    if unique_p[0] > 0.0:
        unique_p = np.concatenate(([0.0], unique_p))
        ecdf = np.concatenate(([0.0], ecdf))
    if unique_p[-1] < 1.0:
        unique_p = np.concatenate((unique_p, [1.0]))
        ecdf = np.concatenate((ecdf, [1.0]))
    widths = np.diff(unique_p)
    slopes = -_isotonic_mean(-np.diff(ecdf) / widths, widths)
    distinct = np.concatenate(([True], slopes[1:] != slopes[:-1]))
    x_knots = unique_p[np.concatenate((distinct, [True]))]
    slope_knots = slopes[distinct]
    y_knots = ecdf[0] + np.concatenate(
        ([0.0], np.cumsum(np.diff(x_knots) * slope_knots))
    )
    last_segment = int(slope_knots.shape[0]) - 1
    return (
        np.delete(x_knots, last_segment),
        np.delete(y_knots, last_segment),
        slope_knots,
    )


def _thresholds_to_weights(
    thresholds: np.ndarray, m_groups: np.ndarray
) -> np.ndarray:
    if np.all(thresholds == 0.0):
        return np.ones(thresholds.shape[0], dtype=np.float64)
    total = float(np.sum(m_groups))
    denominator = float(np.dot(m_groups, thresholds))
    if denominator == 0.0:
        raise RuntimeError("weight denominator is zero")
    return thresholds * total / denominator


def _dense_weights(
    split_sorted_pvalues: list[np.ndarray],
    alpha: float,
    m_groups: np.ndarray,
    m_groups_grenander: np.ndarray,
    adjustment_type: str,
) -> np.ndarray:
    """Solve the infinite-lambda allocation as the historical dense LP."""

    from scipy.optimize import linprog

    nbins = len(split_sorted_pvalues)
    grenander = [
        _grenander(np.where(values > 1e-20, values, 0.0), int(group_size))
        for values, group_size in zip(
            split_sorted_pvalues, m_groups_grenander, strict=True
        )
    ]
    constraint_count = sum(item[2].size for item in grenander)
    rows = np.zeros((constraint_count + 1, 2 * nbins), dtype=np.float64)
    rhs = np.zeros(constraint_count + 1, dtype=np.float64)
    row = 0
    for group, (x_knots, y_knots, slopes) in enumerate(grenander):
        for knot, slope in enumerate(slopes):
            rows[row, group] = 1.0
            rows[row, nbins + group] = -slope
            rhs[row] = y_knots[knot] - slope * x_knots[knot]
            row += 1
    if adjustment_type == "bh":
        rows[-1, :nbins] = -alpha * m_groups
        rows[-1, nbins:] = m_groups
    elif adjustment_type == "bonferroni":
        rows[-1, nbins:] = m_groups
        rhs[-1] = alpha
    else:
        raise ValueError(f"unknown adjustment_type: {adjustment_type!r}")
    objective = np.zeros(2 * nbins, dtype=np.float64)
    objective[:nbins] = m_groups / float(np.sum(m_groups)) * nbins
    fit = linprog(
        -objective,
        A_ub=rows,
        b_ub=rhs,
        bounds=[(0.0, 2.0)] * (2 * nbins),
        method="highs",
    )
    if not fit.success or fit.x is None:
        raise RuntimeError(f"SciPy weight LP did not solve: {fit.message}")
    return _thresholds_to_weights(
        np.maximum(np.asarray(fit.x[nbins:], dtype=np.float64), 0.0),
        m_groups,
    )


def _safe_divide(pvalues: np.ndarray, weights: np.ndarray) -> np.ndarray:
    result = np.ones_like(pvalues)
    np.divide(pvalues, weights, out=result, where=weights != 0.0)
    result[pvalues == 0.0] = 0.0
    np.minimum(result, 1.0, out=result)
    return result


def _adjust_pvalues(
    pvalues: np.ndarray, method: str, n_tests: int
) -> np.ndarray:
    if method == "bonferroni":
        return np.minimum(pvalues * n_tests, 1.0)
    order = np.argsort(pvalues)
    adjusted = pvalues[order] * n_tests / np.arange(1, pvalues.size + 1)
    np.minimum.accumulate(adjusted[::-1], out=adjusted[::-1])
    np.minimum(adjusted, 1.0, out=adjusted)
    result = np.empty_like(pvalues)
    result[order] = adjusted
    return result


def _groups_by_rank(
    covariates: np.ndarray, nbins: int, rng: np.random.Generator
) -> np.ndarray:
    size = covariates.size
    order = np.argsort(covariates, kind="mergesort")
    sorted_covariates = covariates[order]
    ranks = np.empty(size, dtype=np.intp)
    start = 0
    while start < size:
        end = start + 1
        while end < size and sorted_covariates[end] == sorted_covariates[start]:
            end += 1
        if end - start == 1:
            ranks[order[start]] = start
        else:
            ranks[order[start:end]] = start + rng.permutation(end - start)
        start = end
    return ((ranks + 1) * nbins - 1) // size


def _split_by_group(
    pvalues: np.ndarray, groups: np.ndarray, nbins: int
) -> list[np.ndarray]:
    return [np.sort(pvalues[groups == group]) for group in range(nbins)]


def _fit_cross_weights(
    sorted_pvalues: np.ndarray,
    sorted_groups: np.ndarray,
    sorted_folds: np.ndarray,
    alpha: float,
    m_groups: np.ndarray,
    nfolds: int,
    adjustment_type: str,
    folds_prespecified: bool,
) -> tuple[np.ndarray, np.ndarray]:
    nbins = m_groups.size
    available = np.bincount(sorted_groups, minlength=nbins).astype(np.intp)
    weights = np.full(sorted_pvalues.size, np.nan, dtype=np.float64)
    for fold in range(nfolds):
        holdout = sorted_folds == fold
        if not np.any(holdout):
            continue
        if nfolds == 1:
            training = np.ones(sorted_pvalues.size, dtype=bool)
            m_holdout = m_groups.copy()
            m_training = m_groups.copy()
        else:
            training = ~holdout
            training_counts = np.bincount(
                sorted_groups[training], minlength=nbins
            ).astype(np.intp)
            if folds_prespecified:
                m_holdout = np.bincount(
                    sorted_groups[holdout], minlength=nbins
                ).astype(np.intp)
            else:
                m_holdout = (
                    (m_groups - available) / nfolds + available - training_counts
                ).astype(np.intp)
            m_training = (m_groups - m_holdout).astype(np.intp)
        np.maximum(m_holdout, 0, out=m_holdout)
        np.maximum(m_training, 0, out=m_training)
        group_weights = _dense_weights(
            _split_by_group(
                sorted_pvalues[training], sorted_groups[training], nbins
            ),
            alpha,
            m_holdout,
            m_training,
            adjustment_type,
        )
        weights[holdout] = group_weights[sorted_groups[holdout]]
    weighted = _safe_divide(sorted_pvalues, weights)
    adjusted = _adjust_pvalues(weighted, adjustment_type, int(np.sum(m_groups)))
    return adjusted, weights


def adjust_ihw(
    pvalues: object,
    covariates: object,
    alpha: float,
    *,
    nbins: int | str = "auto",
    nfolds: int = 5,
    adjustment_type: str = "bh",
    folds: object | None = None,
    groups: object | None = None,
    m_groups: object | None = None,
    seed: int | None = 1,
) -> IHWResult:
    """Run the retained SciPy baseline on generated or frozen partitions."""

    p = np.asarray(pvalues, dtype=np.float64)
    x = np.asarray(covariates, dtype=np.float64)
    if p.ndim != 1 or x.ndim != 1 or p.size == 0 or p.shape != x.shape:
        raise ValueError("pvalues and covariates must be aligned non-empty vectors")
    size = p.size
    if groups is None:
        resolved_bins = (
            max(1, min(40, size // 1500)) if nbins == "auto" else int(nbins)
        )
        group_id = _groups_by_rank(
            x, resolved_bins, np.random.default_rng(seed)
        )
    else:
        group_id = np.asarray(groups, dtype=np.intp)
        resolved_bins = int(np.max(group_id)) + 1
    family_counts = (
        np.bincount(group_id, minlength=resolved_bins).astype(np.intp)
        if m_groups is None
        else np.asarray(m_groups, dtype=np.intp)
    )
    if resolved_bins == 1:
        return IHWResult(
            _adjust_pvalues(p, adjustment_type, int(np.sum(family_counts))),
            np.ones(size, dtype=np.float64),
        )
    order = np.argsort(p)
    if folds is None:
        fold_id = np.random.default_rng(seed).integers(
            0, nfolds, size=size, dtype=np.intp
        )
        prespecified = False
    else:
        fold_id = np.asarray(folds, dtype=np.intp)
        nfolds = int(np.max(fold_id)) + 1
        prespecified = True
    adjusted, weights = _fit_cross_weights(
        p[order],
        group_id[order],
        fold_id[order],
        alpha,
        family_counts,
        nfolds,
        adjustment_type,
        prespecified,
    )
    restored_adjusted = np.empty_like(adjusted)
    restored_weights = np.empty_like(weights)
    restored_adjusted[order] = adjusted
    restored_weights[order] = weights
    return IHWResult(restored_adjusted, restored_weights)
