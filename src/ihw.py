"""Independent Hypothesis Weighting with a NumPy and Numba runtime."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

try:
    from numba import njit
except ImportError as exc:
    raise ImportError(
        "ihwkit requires Numba for its production implementation"
    ) from exc

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
    penalty: str
    adjustment_type: str
    fold_lambdas: FloatArray
    m_groups: IntegerArray


@njit(cache=False)
def _iso_mean_loops(y: FloatArray, w: FloatArray) -> FloatArray:
    n = y.shape[0]
    if n == 1:
        out0 = np.empty(1, dtype=np.float64)
        out0[0] = y[0]
        return out0
    values = np.empty(n, dtype=np.float64)
    weights = np.empty(n, dtype=np.float64)
    counts = np.empty(n, dtype=np.int64)
    k = 0
    for i in range(n):
        values[k] = y[i]
        weights[k] = w[i]
        counts[k] = 1
        k += 1
        while k >= 2 and values[k - 2] > values[k - 1]:
            tw = weights[k - 2] + weights[k - 1]
            values[k - 2] = (
                weights[k - 2] * values[k - 2] + weights[k - 1] * values[k - 1]
            ) / tw
            weights[k - 2] = tw
            counts[k - 2] += counts[k - 1]
            k -= 1
    out = np.empty(n, dtype=np.float64)
    pos = 0
    for i in range(k):
        ck = counts[i]
        vk = values[i]
        for t in range(ck):
            out[pos + t] = vk
        pos += ck
    return out


def _iso_mean(y: np.ndarray, w: np.ndarray) -> FloatArray:
    """Run the production Numba PAVA kernel on float64 arrays."""

    return _iso_mean_loops(
        np.asarray(y, dtype=np.float64), np.asarray(w, dtype=np.float64)
    )


def _grenander(
    sorted_pvalues: FloatArray, m_total: int
) -> tuple[FloatArray, FloatArray, FloatArray]:
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
    if denom == 0.0:
        raise RuntimeError("weight denom is zero")
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
    if n == 1 or bool(np.all(cov_sorted[1:] != cov_sorted[:-1])):
        ranks[order] = np.arange(1, n + 1, dtype=np.float64)
    else:
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


_LP_EPS = 1e-7
_LP_FEAS_TOL = 1e-5
_LP_MAX_ITER = 250000

@njit(cache=False)
def _simplex_tableau_loops(
    tableau: FloatArray, basis: IntegerArray, eps: float, max_iter: int
) -> tuple[FloatArray, IntegerArray]:
    m = tableau.shape[0] - 1
    n_tot = tableau.shape[1] - 1
    for _it in range(max_iter):
        enter = -1
        for j in range(n_tot):
            if tableau[m, j] < -eps:
                enter = j
                break
        if enter < 0:
            return tableau, basis
        min_ratio = np.inf
        any_pos = False
        for i in range(m):
            col_i = tableau[i, enter]
            if col_i > eps:
                any_pos = True
                ratio = tableau[i, n_tot] / col_i
                min_ratio = min(min_ratio, ratio)
        if not any_pos:
            raise RuntimeError("weight LP did not solve: unbounded")
        leave = -1
        leave_basis = 0
        for i in range(m):
            col_i = tableau[i, enter]
            if col_i <= eps:
                continue
            ratio = tableau[i, n_tot] / col_i
            if abs(ratio - min_ratio) <= eps and (
                leave < 0 or basis[i] < leave_basis
            ):
                leave = i
                leave_basis = basis[i]
        pivot = tableau[leave, enter]
        ncols = n_tot + 1
        for j in range(ncols):
            tableau[leave, j] /= pivot
        for i in range(m + 1):
            if i == leave:
                continue
            fac = tableau[i, enter]
            if fac != 0.0:
                for j in range(ncols):
                    tableau[i, j] -= fac * tableau[leave, j]
        basis[leave] = enter
    raise RuntimeError("weight LP did not solve: iteration limit")


def _simplex_tableau(
    tableau: np.ndarray,
    basis: list[int],
    eps: float,
    max_iter: int,
) -> tuple[FloatArray, list[int]]:
    basis_arr = np.asarray(basis, dtype=np.int64)
    tableau, basis_arr = _simplex_tableau_loops(
        np.asarray(tableau, dtype=np.float64), basis_arr, float(eps), int(max_iter)
    )
    return tableau, [int(v) for v in basis_arr]


def _clear_obj_basic(tableau: np.ndarray, basis: list[int], eps: float) -> None:
    for i, bi in enumerate(basis):
        if bi >= tableau.shape[1] - 1:
            continue
        coef = tableau[-1, bi]
        if abs(coef) > eps:
            tableau[-1] -= coef * tableau[i]


def _max_tableau(c: FloatArray, g: FloatArray, h: FloatArray) -> FloatArray:
    n = c.shape[0]
    m = g.shape[0]
    row_scale = np.max(np.abs(g), axis=1)
    zero_row = row_scale <= _LP_EPS
    if np.any(zero_row & (np.abs(h) > _LP_EPS)):
        raise RuntimeError("weight LP did not solve: infeasible")
    keep = ~zero_row
    g = g[keep]
    h = h[keep]
    if g.shape[0] == 0:
        if np.any(c > _LP_EPS):
            raise RuntimeError("weight LP did not solve: unbounded")
        return np.zeros(n, dtype=np.float64)
    row_scale = np.max(np.abs(g), axis=1)
    g = g / row_scale[:, None]
    h = h / row_scale
    col_scale = np.sqrt(np.maximum(np.max(np.abs(g), axis=0), 1e-300))
    g = g / col_scale
    c = c / col_scale
    m = g.shape[0]
    tableau = np.zeros((m + 1, n + m + 1), dtype=np.float64)
    tableau[:m, :n] = g
    tableau[:m, n : n + m] = np.eye(m)
    tableau[:m, -1] = h
    neg = tableau[:m, -1] < 0.0
    tableau[:m][neg] *= -1.0
    slack_diag = np.diag(tableau[:m, n : n + m])
    need_art = slack_diag < 0.5
    basis = list(range(n, n + m))
    n_struct = n + m
    if np.any(need_art):
        art_rows = np.flatnonzero(need_art)
        n_art = int(art_rows.shape[0])
        art = np.zeros((m + 1, n_art), dtype=np.float64)
        art[art_rows, np.arange(n_art)] = 1.0
        tableau = np.hstack([tableau[:, :-1], art, tableau[:, -1:]])
        tableau[-1, :] = 0.0
        tableau[-1] -= tableau[art_rows].sum(axis=0)
        k = 0
        basis = []
        for i in range(m):
            if need_art[i]:
                basis.append(n_struct + k)
                k += 1
            else:
                basis.append(n + i)
        tableau, basis = _simplex_tableau(tableau, basis, _LP_EPS, _LP_MAX_ITER)
        art_sum = 0.0
        for i, bi in enumerate(basis):
            if bi >= n_struct:
                art_sum += float(tableau[i, -1])
        if art_sum > 1e3 * _LP_EPS:
            raise RuntimeError("weight LP did not solve: infeasible")
        for i, bi in enumerate(basis):
            if bi < n_struct:
                continue
            col = tableau[i, :n_struct]
            enter_cands = np.flatnonzero(np.abs(col) > _LP_EPS)
            if enter_cands.size == 0:
                continue
            enter = int(enter_cands[0])
            pivot = tableau[i, enter]
            tableau[i] /= pivot
            factors = tableau[:, enter].copy()
            factors[i] = 0.0
            tableau -= factors[:, None] * tableau[i]
            basis[i] = enter
        tableau = np.hstack([tableau[:, :n_struct], tableau[:, -1:]])
        keep_rows = [i for i in range(m) if basis[i] < n_struct]
        if len(keep_rows) != m:
            tableau = np.vstack([tableau[keep_rows], tableau[-1:]])
            basis = [basis[i] for i in keep_rows]
            m = len(keep_rows)
        if m == 0:
            if np.any(c > _LP_EPS):
                raise RuntimeError("weight LP did not solve: unbounded")
            return np.zeros(n, dtype=np.float64)
    tableau[-1, :] = 0.0
    tableau[-1, :n] = -c
    _clear_obj_basic(tableau, basis, _LP_EPS)
    tableau, basis = _simplex_tableau(tableau, basis, _LP_EPS, _LP_MAX_ITER)
    y = np.zeros(n, dtype=np.float64)
    for i, bi in enumerate(basis):
        if bi < n:
            y[bi] = tableau[i, -1]
    return y / col_scale


def _solve_lp_numpy(
    objective: np.ndarray,
    a_ub: np.ndarray,
    b_ub: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
) -> FloatArray:
    c = np.asarray(objective, dtype=np.float64).ravel()
    a_ub = np.asarray(a_ub, dtype=np.float64)
    b_ub = np.asarray(b_ub, dtype=np.float64).ravel()
    lb = np.asarray(lb, dtype=np.float64).ravel()
    ub = np.asarray(ub, dtype=np.float64).ravel()
    n = c.shape[0]
    if a_ub.size == 0:
        a_ub = np.zeros((0, n), dtype=np.float64)
    else:
        a_ub = np.atleast_2d(a_ub)
    if a_ub.shape[1] != n or lb.shape[0] != n or ub.shape[0] != n:
        raise RuntimeError("weight LP did not solve: shape mismatch")
    if np.any(~np.isfinite(lb)):
        raise RuntimeError("weight LP did not solve: infinite lower bound")
    b_shift = b_ub - a_ub @ lb
    g = a_ub
    h = b_shift
    finite_ub = np.isfinite(ub)
    if np.any(finite_ub):
        ub_idx = np.flatnonzero(finite_ub)
        cap = ub[ub_idx] - lb[ub_idx]
        if np.any(cap < -_LP_EPS):
            raise RuntimeError("weight LP did not solve: empty bounds")
        ub_rows = np.zeros((ub_idx.shape[0], n), dtype=np.float64)
        ub_rows[np.arange(ub_idx.shape[0]), ub_idx] = 1.0
        g = np.vstack([g, ub_rows])
        h = np.concatenate([h, np.maximum(cap, 0.0)])
    if g.shape[0] == 0:
        y = np.zeros(n, dtype=np.float64)
        for j in range(n):
            if c[j] > _LP_EPS:
                if not np.isfinite(ub[j]):
                    raise RuntimeError("weight LP did not solve: unbounded")
                y[j] = ub[j] - lb[j]
            else:
                y[j] = 0.0
        return y + lb
    y = _max_tableau(c, g, h)
    y = np.maximum(y, 0.0)
    y[y < 1e-10] = 0.0
    x = y + lb
    snap = 1e-10
    near_lb = x - lb <= snap
    x[near_lb] = lb[near_lb]
    finite = np.isfinite(ub)
    near_ub = finite & (ub - x <= snap)
    x[near_ub] = ub[near_ub]
    if not np.all(np.isfinite(x)):
        raise RuntimeError("weight LP did not solve: non-finite solution")
    residual = a_ub @ x - b_ub
    residual_limit = _LP_FEAS_TOL * (1.0 + np.abs(b_ub))
    if np.any(residual > residual_limit):
        raise RuntimeError("weight LP did not solve: infeasible solution")
    if np.any(x < lb - _LP_FEAS_TOL) or np.any(x > ub + _LP_FEAS_TOL):
        raise RuntimeError("weight LP did not solve: bound violation")
    return x


def _ihw_convex(
    split_sorted_pvalues: list[np.ndarray],
    alpha: float,
    m_groups: np.ndarray,
    m_groups_grenander: np.ndarray,
    penalty: str,
    lambda_: float,
    adjustment_type: str,
) -> FloatArray:
    nbins = len(split_sorted_pvalues)
    if lambda_ == 0.0:
        return np.ones(nbins, dtype=np.float64)
    clipped = [np.where(pv > 1e-20, pv, 0.0).astype(np.float64) for pv in split_sorted_pvalues]
    m = int(np.sum(m_groups))
    grenander_list = [
        _grenander(pv, int(mg))
        for pv, mg in zip(clipped, m_groups_grenander, strict=True)
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
            raise IHWValidationError(f"Unknown penalty: {penalty!r}")
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
        raise IHWValidationError(f"Unknown adjustment_type: {adjustment_type!r}")
    lb = np.zeros(n_vars, dtype=np.float64)
    ub = np.full(n_vars, 2.0, dtype=np.float64)
    if n_aux:
        ub[n_base:] = np.inf
    sol = _solve_lp_numpy(objective, rows, rhs, lb, ub)
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
        for lam_idx, lam in enumerate(lambdas):
            inner_folds = _assign_folds(n_internal, nfolds_internal, rng)
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
    preset_fold_lambdas: np.ndarray | None = None,
) -> dict[str, object]:
    n = sorted_pvalues.shape[0]
    nbins = m_groups.shape[0]
    folds_prespecified = sorted_folds is not None
    if sorted_folds is None:
        sorted_folds = _assign_folds(n, nfolds, rng)
    m_groups_available = np.bincount(sorted_groups, minlength=nbins).astype(np.intp)
    sorted_weights = np.full(n, np.nan, dtype=np.float64)
    if preset_fold_lambdas is None:
        fold_lambdas = np.full(nfolds, np.inf, dtype=np.float64)
    else:
        fold_lambdas = np.asarray(preset_fold_lambdas, dtype=np.float64).copy()
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
        if preset_fold_lambdas is not None:
            best_lambda = float(preset_fold_lambdas[fold_idx])
        elif lambdas.shape[0] == 1:
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
        fold_lambdas[fold_idx] = best_lambda
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
        "fold_lambdas": fold_lambdas,
    }


def adjust_ihw(
    pvalues: ArrayLike,
    covariates: ArrayLike,
    alpha: float,
    *,
    exploratory: bool = False,
    covariate_type: str = "ordinal",
    nbins: int | str = "auto",
    nfolds: int = 5,
    nfolds_internal: int = 5,
    nsplits_internal: int = 1,
    lambdas: ArrayLike | str | None = None,
    adjustment_type: str = "bh",
    folds: ArrayLike | None = None,
    groups: ArrayLike | None = None,
    fold_lambdas: ArrayLike | None = None,
    m_groups: ArrayLike | None = None,
    rng: np.random.Generator | None = None,
    seed: int | None = 1,
) -> IHWResult:
    """Run independent hypothesis weighting with the production solver.

    Parameters
    ----------
    pvalues : array-like
        One-dimensional p-values in the closed interval ``[0, 1]``.
    covariates : array-like
        One-dimensional finite covariates aligned with ``pvalues``.
    alpha : float
        Target false discovery rate or family-wise error level.
    exploratory : bool, optional
        Use one fold and infinite lambda for weight inspection.
    covariate_type : {"ordinal", "nominal"}, optional
        Grouping rule for the covariates.
    nbins : int or {"auto"}, optional
        Number of groups for ordinal covariates.
    nfolds : int, optional
        Number of outer cross-validation folds.
    nfolds_internal : int, optional
        Number of inner folds for lambda selection.
    nsplits_internal : int, optional
        Number of inner cross-validation repetitions.
    lambdas : array-like, {"auto"} or None, optional
        Infinite-lambda default, an explicit lambda grid, or the built-in grid.
    adjustment_type : {"bh", "bonferroni"}, optional
        Multiple-testing adjustment used by the weight optimization.
    folds, groups, fold_lambdas, m_groups : array-like or None, optional
        Optional frozen partitions, regularization values, or family group counts.
    rng : numpy.random.Generator or None, optional
        Generator used for fold assignment and lambda selection.
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
    RuntimeError
        If the production dense simplex solver cannot find a finite solution.

    Notes
    -----
    NumPy stores the arrays and Numba executes the PAVA and dense simplex kernels. There is no runtime backend switch or optional solver fallback.
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
    if nfolds <= 0:
        raise IHWValidationError(f"nfolds must be positive, got {nfolds}")
    if nfolds_internal <= 0:
        raise IHWValidationError(f"nfolds_internal must be positive, got {nfolds_internal}")
    if nsplits_internal <= 0:
        raise IHWValidationError(f"nsplits_internal must be positive, got {nsplits_internal}")
    n = p.shape[0]
    if rng is None:
        rng = np.random.default_rng(seed)
    penalty = "total_variation" if covariate_type == "ordinal" else "uniform_deviation"
    if groups is not None:
        g = np.asarray(groups, dtype=np.intp)
        if g.ndim != 1:
            raise IHWValidationError("groups must be a 1-d array")
        if g.shape[0] != n:
            raise IHWValidationError(f"groups length {g.shape[0]} != {n}")
        uniq_g = np.unique(g)
        nbins_i = int(uniq_g.size)
        if nbins_i == 0 or not np.array_equal(uniq_g, np.arange(nbins_i)):
            raise IHWValidationError("groups labels must be in 0 .. nbins-1 with no gaps")
        if not isinstance(nbins, str) and int(nbins) != nbins_i:
            raise IHWValidationError(f"nbins {int(nbins)} does not match groups")
        group_id = g
    else:
        if isinstance(nbins, str):
            if nbins != "auto":
                raise IHWValidationError(f"nbins must be an integer or 'auto', got {nbins!r}")
            nbins_i = max(1, min(40, n // 1500))
        else:
            nbins_i = int(nbins)
            if nbins_i <= 0:
                raise IHWValidationError(f"nbins must be positive, got {nbins}")
        if covariate_type == "nominal":
            group_id = np.unique(x, return_inverse=True)[1].astype(np.intp)
            nbins_i = int(np.unique(group_id).size)
        else:
            bin_rng = np.random.default_rng(seed)
            group_id = _groups_by_filter(x, nbins_i, bin_rng)
    if m_groups is not None:
        mg = np.asarray(m_groups, dtype=np.intp)
        if mg.ndim != 1:
            raise IHWValidationError("m_groups must be a 1-d array")
        if mg.shape[0] != nbins_i:
            raise IHWValidationError(f"m_groups length {mg.shape[0]} != {nbins_i}")
        if np.any(mg < 0):
            raise IHWValidationError("m_groups must be nonnegative")
        m_groups_arr = mg
    else:
        m_groups_arr = np.bincount(group_id, minlength=nbins_i).astype(np.intp)
    if exploratory:
        eff_nfolds = 1
        lam_grid = np.array([np.inf], dtype=np.float64)
    elif lambdas is None:
        eff_nfolds = nfolds
        lam_grid = np.array([np.inf], dtype=np.float64)
    elif isinstance(lambdas, str):
        if lambdas != "auto":
            raise IHWValidationError(f"lambdas must be an array, 'auto', or None, got {lambdas!r}")
        eff_nfolds = nfolds
        lam_grid = np.array(
            sorted({0.0, 1.0, nbins_i / 8, nbins_i / 4, nbins_i / 2, nbins_i, np.inf}),
            dtype=np.float64,
        )
    else:
        eff_nfolds = nfolds
        lam_grid = np.asarray(lambdas, dtype=np.float64)
        if lam_grid.size == 0:
            raise IHWValidationError("lambdas must not be empty")
        if np.any(np.isnan(lam_grid)):
            raise IHWValidationError("lambdas must be finite or +inf")
        if np.any(lam_grid < 0.0):
            raise IHWValidationError("lambdas must be nonnegative")
    pad_method = "fdr_bh" if adjustment_type == "bh" else "bonferroni"
    if nbins_i == 1:
        order = np.argsort(p)
        adj_sorted = _p_adjust(p[order], pad_method, n_tests=int(np.sum(m_groups_arr)))
        inv = np.argsort(order)
        return IHWResult(
            pvalues=p,
            adj_pvalues=adj_sorted[inv],
            weights=np.ones(n, dtype=np.float64),
            weighted_pvalues=p.copy(),
            groups=group_id,
            folds=np.zeros(n, dtype=np.intp),
            alpha=alpha,
            nbins=1,
            nfolds=1,
            penalty=penalty,
            adjustment_type=adjustment_type,
            fold_lambdas=np.array([np.inf], dtype=np.float64),
            m_groups=m_groups_arr,
        )
    order = np.argsort(p)
    sorted_folds = None
    if folds is not None:
        f = np.asarray(folds, dtype=np.intp)
        if f.ndim != 1:
            raise IHWValidationError("folds must be a 1-d array")
        if f.shape[0] != n:
            raise IHWValidationError(f"folds length {f.shape[0]} != {n}")
        uniq = np.unique(f)
        nfolds_f = int(uniq.size)
        if nfolds_f == 0 or not np.array_equal(uniq, np.arange(nfolds_f)):
            raise IHWValidationError("folds labels must be in 0 .. nfolds-1 with no gaps")
        if not exploratory:
            eff_nfolds = nfolds_f
        elif nfolds_f != 1:
            raise IHWValidationError("folds labels must be in 0 .. nfolds-1 with no gaps")
        sorted_folds = f[order]
    preset_lams = None
    if fold_lambdas is not None and not exploratory:
        fl = np.asarray(fold_lambdas, dtype=np.float64)
        if fl.ndim != 1:
            raise IHWValidationError("fold_lambdas must be a 1-d array")
        if fl.shape[0] != eff_nfolds:
            raise IHWValidationError(f"fold_lambdas length {fl.shape[0]} != {eff_nfolds}")
        if fl.size == 0:
            raise IHWValidationError("fold_lambdas must not be empty")
        if np.any(np.isnan(fl)):
            raise IHWValidationError("fold_lambdas must be finite or +inf")
        if np.any(fl < 0.0):
            raise IHWValidationError("fold_lambdas must be nonnegative")
        preset_lams = fl
    result = _ihw_internal(
        group_id[order],
        p[order],
        alpha,
        lam_grid,
        m_groups_arr,
        penalty,
        eff_nfolds,
        nfolds_internal,
        nsplits_internal,
        adjustment_type,
        rng,
        sorted_folds,
        preset_lams,
    )
    inv = np.argsort(order)
    return IHWResult(
        pvalues=p,
        adj_pvalues=np.asarray(result["sorted_adj_p"], dtype=np.float64)[inv],
        weights=np.asarray(result["sorted_weights"], dtype=np.float64)[inv],
        weighted_pvalues=np.asarray(result["sorted_weighted_pvalues"], dtype=np.float64)[inv],
        groups=group_id,
        folds=np.asarray(result["sorted_folds"], dtype=np.intp)[inv],
        alpha=alpha,
        nbins=nbins_i,
        nfolds=eff_nfolds,
        penalty=penalty,
        adjustment_type=adjustment_type,
        fold_lambdas=np.asarray(result["fold_lambdas"], dtype=np.float64),
        m_groups=m_groups_arr,
    )
