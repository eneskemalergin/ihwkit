"""A small, NumPy-only implementation of Independent Hypothesis Weighting."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]
IntegerArray = NDArray[np.intp]


class IHWValidationError(ValueError):
    """Raised when an IHW input or option violates the public contract."""


@dataclass(frozen=True)
class IHWResult:
    """Return adjusted p-values, learned weights, and fit metadata."""

    pvalues: FloatArray
    adj_pvalues: FloatArray
    weights: FloatArray
    weighted_pvalues: FloatArray
    groups: IntegerArray
    folds: IntegerArray
    alpha: float
    nbins: int
    nfolds: int
    covariate_type: str
    adjustment_type: str
    m_groups: IntegerArray


def _isotonic_blocks(
    values: np.ndarray, weights: np.ndarray
) -> tuple[FloatArray, FloatArray, IntegerArray]:
    """Pool adjacent violations and return one value per final block.

    A few array-wide passes collapse the common long decreasing runs. The final stack pass preserves linear worst-case behavior when violations cascade one block at a time.
    """

    block_values = np.asarray(values, dtype=np.float64).copy()
    block_weights = np.asarray(weights, dtype=np.float64).copy()
    block_counts = np.ones(block_values.shape[0], dtype=np.intp)
    for _ in range(4):
        if block_values.shape[0] <= 1:
            return block_values, block_weights, block_counts
        violations = block_values[:-1] > block_values[1:]
        if not np.any(violations):
            return block_values, block_weights, block_counts
        starts = np.concatenate(([0], np.flatnonzero(~violations) + 1))
        next_weights = np.add.reduceat(block_weights, starts)
        block_values = (
            np.add.reduceat(block_values * block_weights, starts) / next_weights
        )
        block_weights = next_weights
        block_counts = np.add.reduceat(block_counts, starts)

    values_list = block_values.tolist()
    weights_list = block_weights.tolist()
    counts_list = block_counts.tolist()
    blocks = 0
    for index in range(len(values_list)):
        values_list[blocks] = values_list[index]
        weights_list[blocks] = weights_list[index]
        counts_list[blocks] = counts_list[index]
        blocks += 1
        while blocks >= 2 and values_list[blocks - 2] > values_list[blocks - 1]:
            total = weights_list[blocks - 2] + weights_list[blocks - 1]
            values_list[blocks - 2] = (
                weights_list[blocks - 2] * values_list[blocks - 2]
                + weights_list[blocks - 1] * values_list[blocks - 1]
            ) / total
            weights_list[blocks - 2] = total
            counts_list[blocks - 2] += counts_list[blocks - 1]
            blocks -= 1
    return (
        np.asarray(values_list[:blocks], dtype=np.float64),
        np.asarray(weights_list[:blocks], dtype=np.float64),
        np.asarray(counts_list[:blocks], dtype=np.intp),
    )


def _grenander(
    sorted_pvalues: FloatArray, m_total: int
) -> tuple[FloatArray, FloatArray, FloatArray]:
    size = sorted_pvalues.shape[0]
    if size == 0:
        return np.array([0.0]), np.array([0.0]), np.array([1.0])
    distinct = sorted_pvalues[1:] != sorted_pvalues[:-1]
    if np.all(distinct):
        unique_p = sorted_pvalues
        ecdf = np.arange(1, size + 1, dtype=np.float64) / float(m_total)
    else:
        starts = np.concatenate(([0], np.flatnonzero(distinct) + 1))
        unique_p = sorted_pvalues[starts]
        counts = np.diff(np.concatenate((starts, [size])))
        ecdf = np.cumsum(counts, dtype=np.float64) / float(m_total)
    if unique_p[0] > 0.0:
        unique_p = np.concatenate(([0.0], unique_p))
        ecdf = np.concatenate(([0.0], ecdf))
    if unique_p[-1] < 1.0:
        unique_p = np.concatenate((unique_p, [1.0]))
        ecdf = np.concatenate((ecdf, [1.0]))
    dx = np.diff(unique_p)
    slope_knots, block_widths, block_counts = _isotonic_blocks(
        -np.diff(ecdf) / dx, dx
    )
    slope_knots = -slope_knots
    distinct_slope = np.concatenate(
        ([True], slope_knots[1:] != slope_knots[:-1])
    )
    if not np.all(distinct_slope):
        starts = np.flatnonzero(distinct_slope)
        slope_knots = slope_knots[starts]
        block_widths = np.add.reduceat(block_widths, starts)
        block_counts = np.add.reduceat(block_counts, starts)
    ends = np.cumsum(block_counts)
    full_x = unique_p[np.concatenate(([0], ends))]
    full_y = ecdf[0] + np.concatenate(
        ([0.0], np.cumsum(block_widths * slope_knots))
    )
    last_segment = int(slope_knots.shape[0]) - 1
    return (
        np.delete(full_x, last_segment),
        np.delete(full_y, last_segment),
        slope_knots,
    )


def _thresholds_to_weights(thresholds: np.ndarray, m_groups: np.ndarray) -> np.ndarray:
    if np.all(thresholds == 0.0):
        return np.ones(thresholds.shape[0], dtype=np.float64)
    m = float(np.sum(m_groups))
    denom = float(np.sum(m_groups.astype(np.float64) * thresholds))
    if denom == 0.0:
        raise RuntimeError("weight denom is zero")
    return thresholds * m / denom


def _unregularized_weights(
    grenander_list: list[tuple[FloatArray, FloatArray, FloatArray]],
    alpha: float,
    m_groups: np.ndarray,
    adjustment_type: str,
) -> FloatArray:
    """Solve the separable infinite-lambda problem without a dense LP.

    Each Grenander estimate is a concave piecewise-linear curve. Without a finite regularization constraint, the bins interact only through the BH or Bonferroni budget. Taking all curve segments in decreasing slope order is therefore the exact continuous-knapsack solution.
    """

    nbins = len(grenander_list)
    base_values = np.empty(nbins, dtype=np.float64)
    slope_parts: list[FloatArray] = []
    group_parts: list[IntegerArray] = []
    width_parts: list[FloatArray] = []
    for group_idx, (x_knots, y_knots, slopes) in enumerate(grenander_list):
        intercepts = y_knots - slopes * x_knots
        base_values[group_idx] = np.clip(np.min(intercepts), 0.0, 2.0)
        starts = np.zeros(slopes.shape[0], dtype=np.float64)
        if slopes.shape[0] > 1:
            slope_drops = slopes[:-1] - slopes[1:]
            np.divide(
                intercepts[1:] - intercepts[:-1],
                slope_drops,
                out=starts[1:],
                where=slope_drops > 0.0,
            )
        np.clip(starts, 0.0, 2.0, out=starts)
        np.maximum.accumulate(starts, out=starts)
        ends = np.empty_like(starts)
        ends[:-1] = starts[1:]
        ends[-1] = 2.0
        positive = slopes > 0.0
        caps = np.full_like(slopes, 2.0)
        np.divide(2.0 - intercepts, slopes, out=caps, where=positive)
        np.minimum(ends, caps, out=ends)
        widths = np.maximum(0.0, ends - starts)
        valid = positive & (widths > 0.0) & (m_groups[group_idx] > 0)
        if np.any(valid):
            slope_parts.append(slopes[valid])
            group_parts.append(
                np.full(int(np.sum(valid)), group_idx, dtype=np.intp)
            )
            width_parts.append(widths[valid])
    if not slope_parts:
        return np.ones(nbins, dtype=np.float64)
    slopes = np.concatenate(slope_parts)
    groups = np.concatenate(group_parts)
    widths = np.concatenate(width_parts)
    order = np.argsort(-slopes, kind="stable")
    slopes = slopes[order]
    groups = groups[order]
    widths = widths[order]
    masses = m_groups[groups].astype(np.float64)
    if adjustment_type == "bh":
        residual = -alpha * float(np.dot(m_groups, base_values))
        costs = masses * (1.0 - alpha * slopes)
    elif adjustment_type == "bonferroni":
        residual = -alpha
        costs = masses
    else:
        raise IHWValidationError(f"Unknown adjustment_type: {adjustment_type!r}")
    cumulative = residual + np.cumsum(costs * widths)
    crossing = np.flatnonzero((costs > 0.0) & (cumulative >= 0.0))
    taken = widths.copy()
    if crossing.size:
        stop = int(crossing[0])
        before = residual if stop == 0 else float(cumulative[stop - 1])
        taken[stop] = min(widths[stop], max(0.0, -before) / costs[stop])
        taken[stop + 1 :] = 0.0
    thresholds = np.bincount(groups, weights=taken, minlength=nbins)
    return _thresholds_to_weights(thresholds, m_groups)


def _safe_divide(pvalues: np.ndarray, weights: np.ndarray) -> np.ndarray:
    out = np.ones_like(pvalues)
    np.divide(pvalues, weights, out=out, where=weights != 0.0)
    out[pvalues == 0.0] = 0.0
    np.minimum(out, 1.0, out=out)
    return out


def _fdr_bh(pvalues: np.ndarray, n_tests: int) -> np.ndarray:
    m = len(pvalues)
    order = np.argsort(pvalues)
    adjusted = pvalues[order]
    adjusted *= n_tests
    adjusted /= np.arange(1, m + 1)
    np.minimum.accumulate(adjusted[::-1], out=adjusted[::-1])
    np.minimum(adjusted, 1.0, out=adjusted)
    result = np.empty_like(pvalues)
    result[order] = adjusted
    return result


def _p_adjust(pvalues: np.ndarray, method: str, n_tests: int | None = None) -> np.ndarray:
    p = np.asarray(pvalues, dtype=np.float64)
    n = n_tests if n_tests is not None else len(p)
    if n <= 0:
        return p.copy()
    if method == "bonferroni":
        return np.minimum(p * n, 1.0)
    return _fdr_bh(p, n)


def _restore_order(sorted_values: np.ndarray, order: np.ndarray) -> np.ndarray:
    restored = np.empty_like(sorted_values)
    restored[order] = sorted_values
    return restored


def _groups_by_filter(covariates: np.ndarray, nbins: int, rng: np.random.Generator) -> np.ndarray:
    n = covariates.shape[0]
    if n == 0:
        return np.array([], dtype=np.intp)
    order = np.argsort(covariates, kind="mergesort")
    cov_sorted = covariates[order]
    zero_based_ranks = np.empty(n, dtype=np.intp)
    if n == 1 or bool(np.all(cov_sorted[1:] != cov_sorted[:-1])):
        zero_based_ranks[order] = np.arange(n, dtype=np.intp)
    else:
        i = 0
        while i < n:
            j = i + 1
            while j < n and cov_sorted[j] == cov_sorted[i]:
                j += 1
            block_len = j - i
            if block_len == 1:
                zero_based_ranks[order[i]] = i
            else:
                zero_based_ranks[order[i:j]] = i + rng.permutation(block_len)
            i = j
    return ((zero_based_ranks + 1) * nbins - 1) // n


def _assign_folds(n: int, nfolds: int, rng: np.random.Generator) -> np.ndarray:
    return rng.integers(0, nfolds, size=n, dtype=np.intp)


def _fit_group_weights(
    split_sorted_pvalues: list[np.ndarray],
    alpha: float,
    m_groups: np.ndarray,
    m_groups_grenander: np.ndarray,
    adjustment_type: str,
) -> FloatArray:
    """Fit the unregularized group weights used by the 0.1 method."""

    clipped = [
        pv
        if pv.shape[0] == 0 or pv[0] > 1e-20
        else np.where(pv > 1e-20, pv, 0.0).astype(np.float64)
        for pv in split_sorted_pvalues
    ]
    grenander_list = [
        _grenander(pv, int(mg))
        for pv, mg in zip(clipped, m_groups_grenander, strict=True)
    ]
    return _unregularized_weights(grenander_list, alpha, m_groups, adjustment_type)


def _ihw_infinite(
    sorted_groups: np.ndarray,
    sorted_pvalues: np.ndarray,
    alpha: float,
    m_groups: np.ndarray,
    nfolds: int,
    adjustment_type: str,
    rng: np.random.Generator,
    sorted_folds: np.ndarray | None,
) -> dict[str, object]:
    """Fit the default direct allocation from reusable group/fold slices."""

    size = sorted_pvalues.shape[0]
    nbins = m_groups.shape[0]
    folds_prespecified = sorted_folds is not None
    if sorted_folds is None:
        sorted_folds = _assign_folds(size, nfolds, rng)
    group_pvalues: list[FloatArray] = []
    group_folds: list[IntegerArray] = []
    for group in range(nbins):
        group_mask = sorted_groups == group
        group_pvalues.append(sorted_pvalues[group_mask])
        group_folds.append(sorted_folds[group_mask])
    m_groups_available = np.fromiter(
        (values.size for values in group_pvalues), dtype=np.intp, count=nbins
    )
    fold_counts = np.zeros((nfolds, nbins), dtype=np.intp)
    for group, folds_for_group in enumerate(group_folds):
        fold_counts[:, group] = np.bincount(folds_for_group, minlength=nfolds)

    sorted_weights = np.full(size, np.nan, dtype=np.float64)
    for fold in range(nfolds):
        fold_mask = sorted_folds == fold
        if not np.any(fold_mask):
            continue
        if nfolds == 1:
            train_split = group_pvalues
            m_holdout = m_groups.copy()
            m_train = m_groups.copy()
        else:
            train_split = [
                values[folds_for_group != fold]
                for values, folds_for_group in zip(
                    group_pvalues, group_folds, strict=True
                )
            ]
            train_counts = m_groups_available - fold_counts[fold]
            if folds_prespecified:
                m_holdout = fold_counts[fold].copy()
            else:
                m_holdout = (
                    (m_groups - m_groups_available) / nfolds
                    + m_groups_available
                    - train_counts
                ).astype(np.intp)
            m_train = (m_groups - m_holdout).astype(np.intp)
        np.maximum(m_holdout, 0, out=m_holdout)
        np.maximum(m_train, 0, out=m_train)
        weights = _fit_group_weights(
            train_split,
            alpha,
            m_holdout,
            m_train,
            adjustment_type,
        )
        sorted_weights[fold_mask] = weights[sorted_groups[fold_mask]]
    sorted_weighted = _safe_divide(sorted_pvalues, sorted_weights)
    m_total = int(np.sum(m_groups))
    pad_method = "fdr_bh" if adjustment_type == "bh" else "bonferroni"
    sorted_adjusted = _p_adjust(sorted_weighted, pad_method, n_tests=m_total)
    return {
        "sorted_weighted_pvalues": sorted_weighted,
        "sorted_adj_p": sorted_adjusted,
        "sorted_weights": sorted_weights,
        "sorted_folds": sorted_folds,
    }


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise IHWValidationError(f"{name} must be a positive integer, got {value!r}")
    result = int(value)
    if result <= 0:
        raise IHWValidationError(f"{name} must be positive, got {value}")
    return result


def _integer_vector(values: ArrayLike, name: str, length: int) -> IntegerArray:
    raw = np.asarray(values)
    if raw.ndim != 1:
        raise IHWValidationError(f"{name} must be a 1-d array")
    if raw.shape[0] != length:
        raise IHWValidationError(f"{name} length {raw.shape[0]} != {length}")
    try:
        numeric = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise IHWValidationError(f"{name} must contain integer values") from exc
    if np.any(~np.isfinite(numeric)) or np.any(numeric != np.floor(numeric)):
        raise IHWValidationError(f"{name} must contain integer values")
    return numeric.astype(np.intp)


def adjust_ihw(
    pvalues: ArrayLike,
    covariates: ArrayLike,
    alpha: float,
    *,
    exploratory: bool = False,
    covariate_type: str = "ordinal",
    nbins: int | str = "auto",
    nfolds: int = 5,
    adjustment_type: str = "bh",
    folds: ArrayLike | None = None,
    groups: ArrayLike | None = None,
    m_groups: ArrayLike | None = None,
    rng: np.random.Generator | None = None,
    seed: int | None = 1,
) -> IHWResult:
    """Run the NumPy-only, infinite-lambda IHW method.

    Parameters
    ----------
    pvalues : array-like
        One-dimensional p-values in the closed interval ``[0, 1]``.
    covariates : array-like
        One-dimensional finite covariates aligned with ``pvalues``.
    alpha : float
        Target false discovery rate or family-wise error level.
    exploratory : bool, optional
        Learn and apply weights on one fold for diagnostics. This is not the
        cross-weighted default and should not be used for confirmatory testing.
    covariate_type : {"ordinal", "nominal"}, optional
        Grouping rule for the covariates.
    nbins : int or {"auto"}, optional
        Number of groups for ordinal covariates.
    nfolds : int, optional
        Number of outer cross-validation folds.
    adjustment_type : {"bh", "bonferroni"}, optional
        Multiple-testing adjustment used by the weight optimization.
    folds, groups, m_groups : array-like or None, optional
        Optional frozen partitions or full-family group counts. ``m_groups``
        may exceed the observed counts when fitting a filtered subset, but may
        never be smaller.
    rng : numpy.random.Generator or None, optional
        Generator used for fold assignment.
    seed : int or None, optional
        Seed used for bin tie permutations and for the default fold generator.

    Returns
    -------
    IHWResult
        Adjusted p-values, weighted p-values, weights, partitions, and metadata.

    Raises
    ------
    IHWValidationError
        If an input, partition, or option violates the public contract.

    Notes
    -----
    Version 0.1 implements five-fold cross-weighting and the unregularized
    (infinite-lambda) allocation directly in NumPy. The returned adjusted
    p-values are produced for the requested ``alpha`` because the learned
    weights depend on that level; do not interpret one fit as an alpha-free
    q-value curve.
    """
    p = np.asarray(pvalues, dtype=np.float64)
    x = np.asarray(covariates, dtype=np.float64)
    if p.ndim != 1:
        raise IHWValidationError("pvalues must be a 1-d array")
    if x.ndim != 1:
        raise IHWValidationError("covariates must be a 1-d array")
    if p.shape[0] == 0:
        raise IHWValidationError("pvalues must not be empty")
    if np.any(~np.isfinite(p)):
        raise IHWValidationError("p-values must be finite")
    if np.any(~np.isfinite(x)):
        raise IHWValidationError("covariates must be finite")
    if np.any((p < 0.0) | (p > 1.0)):
        raise IHWValidationError("p-values must lie in [0, 1]")
    if not (0.0 < alpha < 1.0):
        raise IHWValidationError(f"alpha must be in (0, 1), got {alpha}")
    if p.shape[0] != x.shape[0]:
        raise IHWValidationError(f"Length mismatch: {p.shape[0]} p-values vs {x.shape[0]} covariates")
    if adjustment_type not in ("bh", "bonferroni"):
        raise IHWValidationError(f"Unknown adjustment_type: {adjustment_type!r}")
    if covariate_type not in ("ordinal", "nominal"):
        raise IHWValidationError(f"Unknown covariate_type: {covariate_type!r}")
    nfolds_i = _positive_integer(nfolds, "nfolds")
    n = p.shape[0]
    if rng is None:
        rng = np.random.default_rng(seed)
    if groups is not None:
        g = _integer_vector(groups, "groups", n)
        uniq_g = np.unique(g)
        nbins_i = int(uniq_g.size)
        if nbins_i == 0 or not np.array_equal(uniq_g, np.arange(nbins_i)):
            raise IHWValidationError("groups labels must be in 0 .. nbins-1 with no gaps")
        if not isinstance(nbins, str):
            requested_nbins = _positive_integer(nbins, "nbins")
            if requested_nbins != nbins_i:
                raise IHWValidationError(
                    f"nbins {requested_nbins} does not match groups"
                )
        group_id = g
    else:
        if isinstance(nbins, str):
            if nbins != "auto":
                raise IHWValidationError(f"nbins must be an integer or 'auto', got {nbins!r}")
            nbins_i = max(1, min(40, n // 1500))
        else:
            nbins_i = _positive_integer(nbins, "nbins")
        if covariate_type == "nominal":
            group_id = np.unique(x, return_inverse=True)[1].astype(np.intp)
            nbins_i = int(np.unique(group_id).size)
        else:
            bin_rng = np.random.default_rng(seed)
            group_id = _groups_by_filter(x, nbins_i, bin_rng)
    if m_groups is not None:
        mg = _integer_vector(m_groups, "m_groups", nbins_i)
        if np.any(mg < 0):
            raise IHWValidationError("m_groups must be nonnegative")
        observed_counts = np.bincount(group_id, minlength=nbins_i)
        if np.any(mg < observed_counts):
            raise IHWValidationError(
                "m_groups cannot be smaller than the observed group counts"
            )
        m_groups_arr = mg
    else:
        m_groups_arr = np.bincount(group_id, minlength=nbins_i).astype(np.intp)
    eff_nfolds = 1 if exploratory else nfolds_i
    validated_folds = None
    if folds is not None:
        validated_folds = _integer_vector(folds, "folds", n)
        uniq = np.unique(validated_folds)
        nfolds_f = int(uniq.size)
        if nfolds_f == 0 or not np.array_equal(uniq, np.arange(nfolds_f)):
            raise IHWValidationError("folds labels must be in 0 .. nfolds-1 with no gaps")
        if not exploratory:
            eff_nfolds = nfolds_f
        elif nfolds_f != 1:
            raise IHWValidationError("folds labels must be in 0 .. nfolds-1 with no gaps")
    pad_method = "fdr_bh" if adjustment_type == "bh" else "bonferroni"
    if nbins_i == 1:
        order = np.argsort(p)
        adj_sorted = _p_adjust(p[order], pad_method, n_tests=int(np.sum(m_groups_arr)))
        return IHWResult(
            pvalues=p,
            adj_pvalues=_restore_order(adj_sorted, order),
            weights=np.ones(n, dtype=np.float64),
            weighted_pvalues=p.copy(),
            groups=group_id,
            folds=np.zeros(n, dtype=np.intp),
            alpha=alpha,
            nbins=1,
            nfolds=1,
            covariate_type=covariate_type,
            adjustment_type=adjustment_type,
            m_groups=m_groups_arr,
        )
    order = np.argsort(p)
    sorted_folds = None
    if validated_folds is not None:
        sorted_folds = validated_folds[order]
    result = _ihw_infinite(
        group_id[order],
        p[order],
        alpha,
        m_groups_arr,
        eff_nfolds,
        adjustment_type,
        rng,
        sorted_folds,
    )
    return IHWResult(
        pvalues=p,
        adj_pvalues=_restore_order(
            np.asarray(result["sorted_adj_p"], dtype=np.float64), order
        ),
        weights=_restore_order(
            np.asarray(result["sorted_weights"], dtype=np.float64), order
        ),
        weighted_pvalues=_restore_order(
            np.asarray(result["sorted_weighted_pvalues"], dtype=np.float64), order
        ),
        groups=group_id,
        folds=_restore_order(
            np.asarray(result["sorted_folds"], dtype=np.intp), order
        ),
        alpha=alpha,
        nbins=nbins_i,
        nfolds=eff_nfolds,
        covariate_type=covariate_type,
        adjustment_type=adjustment_type,
        m_groups=m_groups_arr,
    )
