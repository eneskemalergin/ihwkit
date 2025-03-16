from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog


@dataclass(frozen=True)
class IHWResult:
    pvalues: np.ndarray
    adj_pvalues: np.ndarray
    weights: np.ndarray
    weighted_pvalues: np.ndarray
    groups: np.ndarray
    folds: np.ndarray
    alpha: float
    nbins: int
    nfolds: int
    penalty: str
    adjustment_type: str


def _iso_mean(y: np.ndarray, w: np.ndarray) -> np.ndarray:
    n = y.shape[0]
    if n == 1:
        return y.copy()
    values = y.astype(np.float64, copy=True)
    weights = w.astype(np.float64, copy=True)
    counts = np.ones(n, dtype=np.int64)
    size = n
    i = 0
    while i < size - 1:
        if values[i] <= values[i + 1]:
            i += 1
            continue
        tw = weights[i] + weights[i + 1]
        values[i] = (weights[i] * values[i] + weights[i + 1] * values[i + 1]) / tw
        weights[i] = tw
        counts[i] += counts[i + 1]
        values[i + 1 : size - 1] = values[i + 2 : size]
        weights[i + 1 : size - 1] = weights[i + 2 : size]
        counts[i + 1 : size - 1] = counts[i + 2 : size]
        size -= 1
        if i > 0:
            i -= 1
    out = np.empty(n, dtype=np.float64)
    pos = 0
    for k in range(size):
        out[pos : pos + int(counts[k])] = values[k]
        pos += int(counts[k])
    return out


def _grenander(sorted_pvalues: np.ndarray, m_total: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if sorted_pvalues.shape[0] == 0:
        return np.array([0.0]), np.array([0.0]), np.array([1.0])
    unique_p, counts = np.unique(sorted_pvalues, return_counts=True)
    ecdf = np.cumsum(counts, dtype=np.float64) / float(m_total)
    if unique_p[0] > 0.0:
        unique_p = np.concatenate(([0.0], unique_p))
        ecdf = np.concatenate(([0.0], ecdf))
    if unique_p[-1] < 1.0:
        unique_p = np.concatenate((unique_p, [1.0]))
        ecdf = np.concatenate((ecdf, [1.0]))
    dx = np.diff(unique_p)
    dy = np.diff(ecdf)
    rawslope = dy / dx
    rawslope = np.where(np.isposinf(rawslope), np.finfo(np.float64).max, rawslope)
    rawslope = np.where(np.isneginf(rawslope), -np.finfo(np.float64).max, rawslope)
    slope = -_iso_mean(-rawslope, dx)
    dup = np.concatenate(([True], slope[1:] != slope[:-1]))
    x_knots = unique_p[np.concatenate([dup, [True]])]
    dx_knots = np.diff(x_knots)
    slope_knots = slope[dup]
    y_knots = ecdf[0] + np.concatenate(([0.0], np.cumsum(dx_knots * slope_knots)))
    n_seg = int(slope_knots.shape[0])
    return np.delete(x_knots, n_seg - 1), np.delete(y_knots, n_seg - 1), slope_knots


def _thresholds_to_weights(thresholds: np.ndarray, m_groups: np.ndarray) -> np.ndarray:
    if np.all(thresholds == 0.0):
        return np.ones(thresholds.shape[0], dtype=np.float64)
    m = float(np.sum(m_groups))
    denom = float(np.sum(m_groups.astype(np.float64) * thresholds))
    return thresholds * m / denom


def _safe_divide(pvalues: np.ndarray, weights: np.ndarray) -> np.ndarray:
    out = np.empty_like(pvalues)
    out[pvalues == 0.0] = 0.0
    zero_w = (pvalues != 0.0) & (weights == 0.0)
    out[zero_w] = 1.0
    valid = (pvalues != 0.0) & (weights != 0.0)
    out[valid] = np.minimum(pvalues[valid] / weights[valid], 1.0)
    return out


def _fdr_bh(pvalues: np.ndarray, n_tests: int) -> np.ndarray:
    m = len(pvalues)
    order = np.argsort(pvalues)[::-1]
    steps = n_tests / np.arange(n_tests, n_tests - m, -1)
    q = np.minimum(1.0, np.minimum.accumulate(steps * pvalues[order]))
    result = np.empty_like(pvalues)
    result[order] = q
    return result


def _p_adjust(pvalues: np.ndarray, method: str, n_tests: int | None = None) -> np.ndarray:
    p = np.asarray(pvalues, dtype=np.float64)
    n = n_tests if n_tests is not None else len(p)
    if n <= 0:
        return p.copy()
    if method == "bonferroni":
        return np.minimum(p * n, 1.0)
    return _fdr_bh(p, n)


def _groups_by_filter(covariates: np.ndarray, nbins: int, rng: np.random.Generator) -> np.ndarray:
    n = covariates.shape[0]
    if n == 0:
        return np.array([], dtype=np.intp)
    order = np.argsort(covariates, kind="mergesort")
    cov_sorted = covariates[order]
    ranks = np.empty(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i + 1
        while j < n and cov_sorted[j] == cov_sorted[i]:
            j += 1
        base = float(i + 1)
        block_len = j - i
        if block_len == 1:
            ranks[order[i]] = base
        else:
            offs = rng.permutation(block_len).astype(np.float64)
            for k in range(block_len):
                ranks[order[i + k]] = base + offs[k]
        i = j
    groups = np.ceil((ranks / n) * nbins).astype(np.intp) - 1
    return np.clip(groups, 0, nbins - 1).astype(np.intp)


def _assign_folds(n: int, nfolds: int, rng: np.random.Generator) -> np.ndarray:
    return rng.integers(0, nfolds, size=n, dtype=np.intp)


def _split_pvalues_by_group(pvalues: np.ndarray, groups: np.ndarray, nbins: int) -> list[np.ndarray]:
    if pvalues.shape[0] == 0:
        return [np.array([], dtype=np.float64) for _ in range(nbins)]
    order = np.argsort(groups, kind="mergesort")
    g_sorted = groups[order]
    p_sorted = pvalues[order]
    counts = np.bincount(g_sorted, minlength=nbins)
    out: list[np.ndarray] = []
    pos = 0
    for cnt in counts:
        if cnt == 0:
            out.append(np.array([], dtype=np.float64))
        else:
            out.append(np.sort(p_sorted[pos : pos + cnt]))
            pos += cnt
    return out


def _solve_lp(objective: np.ndarray, a_ub: np.ndarray, b_ub: np.ndarray, lb: np.ndarray, ub: np.ndarray) -> np.ndarray:
    n = objective.shape[0]
    bounds = []
    for j in range(n):
        hi = float(ub[j]) if np.isfinite(ub[j]) else None
        bounds.append((float(lb[j]), hi))
    res = linprog(-objective, A_ub=a_ub, b_ub=b_ub, bounds=bounds, method="highs")
    if not res.success or res.x is None:
        detail = getattr(res, "message", "no solution")
        raise RuntimeError(f"weight LP did not solve: {detail}")
    x = np.asarray(res.x, dtype=np.float64)
    if not np.all(np.isfinite(x)):
        raise RuntimeError("weight LP did not solve: non-finite solution")
    return x


def _ihw_convex(
    split_sorted_pvalues: list[np.ndarray],
    alpha: float,
    m_groups: np.ndarray,
    m_groups_grenander: np.ndarray,
    penalty: str,
    lambda_: float,
    adjustment_type: str,
) -> np.ndarray:
    nbins = len(split_sorted_pvalues)
    if lambda_ == 0.0:
        return np.ones(nbins, dtype=np.float64)
    clipped = [np.where(pv > 1e-20, pv, 0.0).astype(np.float64) for pv in split_sorted_pvalues]
    m = int(np.sum(m_groups))
    grenander_list = [
        _grenander(pv, int(mg)) for pv, mg in zip(clipped, m_groups_grenander, strict=True)
    ]
    n_constraints = sum(len(g[2]) for g in grenander_list)
    rows = np.zeros((n_constraints, 2 * nbins), dtype=np.float64)
    rhs = np.empty(n_constraints, dtype=np.float64)
    row = 0
    for g_idx, (x_knots, y_knots, slope_knots) in enumerate(grenander_list):
        for k in range(len(slope_knots)):
            slope = slope_knots[k]
            rows[row, g_idx] = 1.0
            rows[row, nbins + g_idx] = -slope
            rhs[row] = y_knots[k] - slope * x_knots[k]
            row += 1
    objective = np.zeros(2 * nbins, dtype=np.float64)
    for g in range(nbins):
        objective[g] = float(m_groups[g]) / m * nbins
    n_base = 2 * nbins
    n_aux = 0
    if lambda_ < np.inf:
        if penalty == "total_variation":
            n_aux = nbins - 1
            aux_rows = np.zeros((2 * (nbins - 1) + 1, n_base + n_aux), dtype=np.float64)
            aux_rhs = np.zeros(2 * (nbins - 1) + 1, dtype=np.float64)
            for g in range(nbins - 1):
                aux_rows[g, nbins + g + 1] = 1.0
                aux_rows[g, nbins + g] = -1.0
                aux_rows[g, n_base + g] = -1.0
                r2 = (nbins - 1) + g
                aux_rows[r2, nbins + g + 1] = -1.0
                aux_rows[r2, nbins + g] = 1.0
                aux_rows[r2, n_base + g] = -1.0
            tv_row = 2 * (nbins - 1)
            for g in range(nbins - 1):
                aux_rows[tv_row, n_base + g] = 1.0
            for g in range(nbins):
                aux_rows[tv_row, nbins + g] = -lambda_ * float(m_groups[g]) / m
        elif penalty == "uniform_deviation":
            n_aux = nbins
            aux_rows = np.zeros((2 * nbins + 1, n_base + n_aux), dtype=np.float64)
            aux_rhs = np.zeros(2 * nbins + 1, dtype=np.float64)
            for g in range(nbins):
                for h in range(nbins):
                    coeff = float(m) if h == g else 0.0
                    coeff -= float(m_groups[h])
                    if coeff != 0.0:
                        aux_rows[g, nbins + h] = coeff
                aux_rows[g, n_base + g] = -1.0
                r2 = nbins + g
                for h in range(nbins):
                    coeff = -float(m) if h == g else 0.0
                    coeff += float(m_groups[h])
                    if coeff != 0.0:
                        aux_rows[r2, nbins + h] = coeff
                aux_rows[r2, n_base + g] = -1.0
            ud_row = 2 * nbins
            for g in range(nbins):
                aux_rows[ud_row, n_base + g] = 1.0
                aux_rows[ud_row, nbins + g] = -lambda_ * float(m_groups[g])
        else:
            raise ValueError(f"Unknown penalty: {penalty!r}")
        pad = np.zeros((rows.shape[0], n_aux), dtype=np.float64)
        rows = np.vstack([np.hstack([rows, pad]), aux_rows])
        rhs = np.concatenate([rhs, aux_rhs])
        objective = np.concatenate([objective, np.zeros(n_aux, dtype=np.float64)])
    n_vars = objective.shape[0]
    extra = np.zeros(n_vars, dtype=np.float64)
    if adjustment_type == "bh":
        for g in range(nbins):
            extra[g] = -alpha * float(m_groups[g])
            extra[nbins + g] = float(m_groups[g])
        rows = np.vstack([rows, extra])
        rhs = np.concatenate([rhs, [0.0]])
    elif adjustment_type == "bonferroni":
        for g in range(nbins):
            extra[nbins + g] = float(m_groups[g])
        rows = np.vstack([rows, extra])
        rhs = np.concatenate([rhs, [alpha]])
    else:
        raise ValueError(f"Unknown adjustment_type: {adjustment_type!r}")
    lb = np.zeros(n_vars, dtype=np.float64)
    ub = np.full(n_vars, 2.0, dtype=np.float64)
    if n_aux:
        ub[n_base:] = np.inf
    sol = _solve_lp(objective, rows, rhs, lb, ub)
    thresholds = np.maximum(sol[nbins : 2 * nbins], 0.0)
    return _thresholds_to_weights(thresholds, m_groups)


def _select_lambda(
    sorted_groups: np.ndarray,
    sorted_pvalues: np.ndarray,
    alpha: float,
    lambdas: np.ndarray,
    m_groups: np.ndarray,
    penalty: str,
    nfolds_internal: int,
    nsplits_internal: int,
    adjustment_type: str,
    rng: np.random.Generator,
) -> float:
    order = np.argsort(sorted_pvalues)
    internal_p = sorted_pvalues[order]
    internal_g = sorted_groups[order]
    n_internal = internal_p.shape[0]
    scores = np.zeros(lambdas.shape[0], dtype=np.float64)
    for _ in range(nsplits_internal):
        inner_folds = _assign_folds(n_internal, nfolds_internal, rng)
        for lam_idx, lam in enumerate(lambdas):
            result = _ihw_internal(
                internal_g,
                internal_p,
                alpha,
                np.array([lam], dtype=np.float64),
                m_groups,
                penalty,
                nfolds_internal,
                1,
                1,
                adjustment_type,
                rng,
                inner_folds,
            )
            scores[lam_idx] += float(result["rjs"])
    scores /= float(nsplits_internal)
    return float(lambdas[int(np.argmax(scores))])


def _ihw_internal(
    sorted_groups: np.ndarray,
    sorted_pvalues: np.ndarray,
    alpha: float,
    lambdas: np.ndarray,
    m_groups: np.ndarray,
    penalty: str,
    nfolds: int,
    nfolds_internal: int,
    nsplits_internal: int,
    adjustment_type: str,
    rng: np.random.Generator,
    sorted_folds: np.ndarray | None,
) -> dict:
    n = sorted_pvalues.shape[0]
    nbins = m_groups.shape[0]
    folds_prespecified = sorted_folds is not None
    if sorted_folds is None:
        sorted_folds = _assign_folds(n, nfolds, rng)
    m_groups_available = np.bincount(sorted_groups, minlength=nbins).astype(np.intp)
    sorted_weights = np.full(n, np.nan, dtype=np.float64)
    for fold_idx in range(nfolds):
        fold_mask = sorted_folds == fold_idx
        if not np.any(fold_mask):
            continue
        train_mask = ~fold_mask
        if nfolds == 1:
            train_mask = np.ones(n, dtype=bool)
            fold_weight_mask = np.ones(n, dtype=bool)
        else:
            fold_weight_mask = fold_mask
        train_groups = sorted_groups[train_mask]
        train_pvalues = sorted_pvalues[train_mask]
        if nfolds == 1:
            m_holdout = m_groups.copy()
            m_train = m_groups.copy()
        elif folds_prespecified:
            holdout_counts = np.bincount(sorted_groups[fold_mask], minlength=nbins).astype(np.intp)
            m_holdout = holdout_counts
            m_train = m_groups - m_holdout
        else:
            train_counts = np.bincount(train_groups, minlength=nbins).astype(np.intp)
            m_holdout = (
                (m_groups - m_groups_available) / nfolds + m_groups_available - train_counts
            ).astype(np.intp)
            m_train = (m_groups - m_holdout).astype(np.intp)
        m_holdout = np.maximum(m_holdout, 0)
        m_train = np.maximum(m_train, 0)
        train_split = _split_pvalues_by_group(train_pvalues, train_groups, nbins)
        if lambdas.shape[0] == 1:
            best_lambda = float(lambdas[0])
        else:
            best_lambda = _select_lambda(
                train_groups,
                train_pvalues,
                alpha,
                lambdas,
                m_train,
                penalty,
                nfolds_internal,
                nsplits_internal,
                adjustment_type,
                rng,
            )
        ws = _ihw_convex(
            train_split,
            alpha,
            m_holdout,
            m_train,
            penalty,
            best_lambda,
            adjustment_type,
        )
        sorted_weights[fold_weight_mask] = ws[sorted_groups[fold_weight_mask]]
    sorted_weighted = _safe_divide(sorted_pvalues, sorted_weights)
    m_total = int(np.sum(m_groups))
    pad_method = "fdr_bh" if adjustment_type == "bh" else "bonferroni"
    sorted_adj = _p_adjust(sorted_weighted, pad_method, n_tests=m_total)
    return {
        "rjs": int(np.sum(sorted_adj <= alpha)),
        "sorted_weighted_pvalues": sorted_weighted,
        "sorted_adj_p": sorted_adj,
        "sorted_weights": sorted_weights,
        "sorted_folds": sorted_folds,
    }


def adjust_ihw(
    pvalues,
    covariates,
    alpha: float,
    *,
    exploratory: bool = False,
    covariate_type: str = "ordinal",
    nbins: int | str = "auto",
    nfolds: int = 5,
    nfolds_internal: int = 5,
    nsplits_internal: int = 1,
    lambdas=None,
    adjustment_type: str = "bh",
    folds=None,
    rng: np.random.Generator | None = None,
    seed: int | None = 1,
) -> IHWResult:
    p = np.asarray(pvalues, dtype=np.float64)
    x = np.asarray(covariates, dtype=np.float64)
    if p.ndim != 1:
        raise ValueError("pvalues must be a 1-d array")
    if x.ndim != 1:
        raise ValueError("covariates must be a 1-d array")
    if p.shape[0] == 0:
        raise ValueError("pvalues must not be empty")
    if np.any(~np.isfinite(p)):
        raise ValueError("p-values must be finite")
    if np.any(~np.isfinite(x)):
        raise ValueError("covariates must be finite")
    if np.any((p < 0.0) | (p > 1.0)):
        raise ValueError("p-values must lie in [0, 1]")
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    if p.shape[0] != x.shape[0]:
        raise ValueError(f"Length mismatch: {p.shape[0]} p-values vs {x.shape[0]} covariates")
    if adjustment_type not in ("bh", "bonferroni"):
        raise ValueError(f"Unknown adjustment_type: {adjustment_type!r}")
    if covariate_type not in ("ordinal", "nominal"):
        raise ValueError(f"Unknown covariate_type: {covariate_type!r}")
    if nfolds <= 0:
        raise ValueError(f"nfolds must be positive, got {nfolds}")
    n = p.shape[0]
    if rng is None:
        rng = np.random.default_rng(seed)
    penalty = "total_variation" if covariate_type == "ordinal" else "uniform_deviation"
    if isinstance(nbins, str):
        if nbins != "auto":
            raise ValueError(f"nbins must be an integer or 'auto', got {nbins!r}")
        nbins_i = max(1, min(40, n // 1500))
    else:
        nbins_i = int(nbins)
        if nbins_i <= 0:
            raise ValueError(f"nbins must be positive, got {nbins}")
    groups = _groups_by_filter(x, nbins_i, rng)
    m_groups = np.bincount(groups, minlength=nbins_i).astype(np.intp)
    if exploratory:
        eff_nfolds = 1
        lam_grid = np.array([np.inf], dtype=np.float64)
    elif lambdas is None:
        eff_nfolds = nfolds
        lam_grid = np.array([np.inf], dtype=np.float64)
    elif isinstance(lambdas, str):
        if lambdas != "auto":
            raise ValueError(f"lambdas must be an array, 'auto', or None, got {lambdas!r}")
        eff_nfolds = nfolds
        lam_grid = np.array(
            sorted({0.0, 1.0, nbins_i / 8, nbins_i / 4, nbins_i / 2, nbins_i, np.inf}),
            dtype=np.float64,
        )
    else:
        eff_nfolds = nfolds
        lam_grid = np.asarray(lambdas, dtype=np.float64)
        if lam_grid.size == 0:
            raise ValueError("lambdas must not be empty")
        if np.any(np.isnan(lam_grid)):
            raise ValueError("lambdas must be finite or +inf")
        if np.any(lam_grid < 0.0):
            raise ValueError("lambdas must be nonnegative")
    pad_method = "fdr_bh" if adjustment_type == "bh" else "bonferroni"
    if nbins_i == 1:
        order = np.argsort(p)
        adj_sorted = _p_adjust(p[order], pad_method, n_tests=n)
        inv = np.argsort(order)
        return IHWResult(
            pvalues=p,
            adj_pvalues=adj_sorted[inv],
            weights=np.ones(n, dtype=np.float64),
            weighted_pvalues=p.copy(),
            groups=groups,
            folds=np.zeros(n, dtype=np.intp),
            alpha=alpha,
            nbins=1,
            nfolds=1,
            penalty=penalty,
            adjustment_type=adjustment_type,
        )
    order = np.argsort(p)
    sorted_folds = None
    if folds is not None:
        f = np.asarray(folds, dtype=np.intp)
        if f.shape[0] != n:
            raise ValueError(f"folds length {f.shape[0]} != {n}")
        if np.any((f < 0) | (f >= eff_nfolds)):
            raise ValueError(f"folds labels must be in 0 .. {eff_nfolds - 1}")
        sorted_folds = f[order]
    result = _ihw_internal(
        groups[order],
        p[order],
        alpha,
        lam_grid,
        m_groups,
        penalty,
        eff_nfolds,
        nfolds_internal,
        nsplits_internal,
        adjustment_type,
        rng,
        sorted_folds,
    )
    inv = np.argsort(order)
    return IHWResult(
        pvalues=p,
        adj_pvalues=np.asarray(result["sorted_adj_p"], dtype=np.float64)[inv],
        weights=np.asarray(result["sorted_weights"], dtype=np.float64)[inv],
        weighted_pvalues=np.asarray(result["sorted_weighted_pvalues"], dtype=np.float64)[inv],
        groups=groups,
        folds=np.asarray(result["sorted_folds"], dtype=np.intp)[inv],
        alpha=alpha,
        nbins=nbins_i,
        nfolds=eff_nfolds,
        penalty=penalty,
        adjustment_type=adjustment_type,
    )
