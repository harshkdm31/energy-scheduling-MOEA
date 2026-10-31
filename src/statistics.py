
"""
Non-parametric statistical analysis engine for multi-objective experimental comparisons.
"""

from typing import Dict, List, Tuple, Union

import numpy as np
import scipy.stats


ArrayLike = Union[List[float], np.ndarray]


def _as_finite_array(data: ArrayLike, name: str) -> np.ndarray:
    """Convert input to a non-empty, finite one-dimensional float array."""
    arr = np.asarray(data, dtype=np.float64).ravel()

    if len(arr) == 0:
        raise ValueError(f"{name} must be non-empty.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values.")

    return arr


def paired_wilcoxon_test(
    data_a: ArrayLike,
    data_b: ArrayLike,
) -> Tuple[float, float]:
    """
    Paired two-tailed Wilcoxon signed-rank test across matching run seeds.

    Positive differences correspond to data_a > data_b.
    """
    a = _as_finite_array(data_a, "data_a")
    b = _as_finite_array(data_b, "data_b")

    if len(a) != len(b):
        raise ValueError(f"Array length mismatch: {len(a)} vs {len(b)}.")

    diffs = a - b

    if np.all(np.isclose(diffs, 0.0, atol=1e-12)):
        return 0.0, 1.0

    res = scipy.stats.wilcoxon(
        diffs,
        alternative="two-sided",
        zero_method="pratt",
        method="auto",
    )
    return float(res.statistic), float(res.pvalue)


def one_sample_wilcoxon_test(
    data: ArrayLike,
    reference: float,
) -> Tuple[float, float]:
    """
    Two-tailed one-sample Wilcoxon signed-rank test against a fixed reference.

    Example:
        data = stochastic algorithm HV values across independent runs.
        reference = the single deterministic baseline HV.

    Each stochastic run contributes one difference. The reference is not
    treated as a second sample or as repeated independent observations.
    """
    values = _as_finite_array(data, "data")
    reference = float(reference)

    if not np.isfinite(reference):
        raise ValueError("reference must be finite.")

    diffs = values - reference

    if np.all(np.isclose(diffs, 0.0, atol=1e-12)):
        return 0.0, 1.0

    res = scipy.stats.wilcoxon(
        diffs,
        alternative="two-sided",
        zero_method="pratt",
        method="auto",
    )
    return float(res.statistic), float(res.pvalue)


def calculate_paired_rank_biserial(
    data_a: ArrayLike,
    data_b: ArrayLike,
) -> float:
    """
    Calculate matched-pairs rank-biserial correlation.

    Positive values indicate data_a tends to exceed data_b.
    Negative values indicate data_b tends to exceed data_a.
    The result is bounded by [-1, 1].
    """
    a = _as_finite_array(data_a, "data_a")
    b = _as_finite_array(data_b, "data_b")

    if len(a) != len(b):
        raise ValueError(f"Array length mismatch: {len(a)} vs {len(b)}.")

    diffs = a - b
    nonzero = diffs[~np.isclose(diffs, 0.0, atol=1e-12)]

    if len(nonzero) == 0:
        return 0.0

    ranks = scipy.stats.rankdata(np.abs(nonzero), method="average")

    positive_rank_sum = float(np.sum(ranks[nonzero > 0]))
    negative_rank_sum = float(np.sum(ranks[nonzero < 0]))
    total_rank_sum = positive_rank_sum + negative_rank_sum

    if total_rank_sum == 0.0:
        return 0.0

    return float(
        (positive_rank_sum - negative_rank_sum) / total_rank_sum
    )


def holm_bonferroni_correction(
    p_values: List[float],
    alpha: float = 0.05,
) -> List[Tuple[float, bool]]:
    """
    Holm-Bonferroni step-down correction for multiple comparisons.

    Returns adjusted p-values and significance flags in original input order.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be between 0 and 1.")

    if len(p_values) == 0:
        return []

    values = np.asarray(p_values, dtype=np.float64)

    if not np.all(np.isfinite(values)):
        raise ValueError("p-values must be finite.")
    if np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("p-values must be between 0 and 1.")

    m = len(values)
    indexed_p = sorted(enumerate(values.tolist()), key=lambda x: x[1])

    rejected_ranks = set()
    for rank_idx, (_, p_val) in enumerate(indexed_p):
        threshold = alpha / (m - rank_idx)
        if p_val <= threshold:
            rejected_ranks.add(rank_idx)
        else:
            break

    adj_p_sorted = [0.0] * m
    running_max = 0.0

    for rank_idx, (_, p_val) in enumerate(indexed_p):
        curr_adj = min(1.0, (m - rank_idx) * p_val)
        running_max = max(running_max, curr_adj)
        adj_p_sorted[rank_idx] = running_max

    results = [None] * m

    for rank_idx, (orig_idx, _) in enumerate(indexed_p):
        results[orig_idx] = (
            float(adj_p_sorted[rank_idx]),
            rank_idx in rejected_ranks,
        )

    return results


def calculate_cliffs_delta(
    data_a: ArrayLike,
    data_b: ArrayLike,
) -> float:
    """
    Compute independent-sample Cliff's delta.

    d = (#(a_i > b_j) - #(a_i < b_j)) / (n_a * n_b)

    Use for independent samples, not matched-pairs comparisons.
    """
    a = _as_finite_array(data_a, "data_a")
    b = _as_finite_array(data_b, "data_b")

    diffs = a[:, None] - b[None, :]
    greater = np.sum(diffs > 1e-12)
    less = np.sum(diffs < -1e-12)

    return float((greater - less) / (len(a) * len(b)))


def summarize_metric(
    data: ArrayLike,
) -> Dict[str, float]:
    """Summarize a metric distribution using median and IQR."""
    arr = _as_finite_array(data, "data")

    q25 = float(np.percentile(arr, 25))
    q75 = float(np.percentile(arr, 75))

    return {
        "median": float(np.median(arr)),
        "iqr": float(q75 - q25),
        "q25": q25,
        "q75": q75,
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "n": len(arr),
    }
