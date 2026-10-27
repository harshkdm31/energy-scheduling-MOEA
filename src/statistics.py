"""
Non-parametric statistical analysis engine for multi-objective experimental comparisons.
"""

from typing import Any, Dict, List, Tuple, Union
import numpy as np
import scipy.stats


def paired_wilcoxon_test(
    data_a: Union[List[float], np.ndarray],
    data_b: Union[List[float], np.ndarray],
) -> Tuple[float, float]:
    """
    Paired two-tailed Wilcoxon signed-rank test across matching seed pairs (Deliverable B9).

    Args:
        data_a: Metric observations from algorithm A across seeds.
        data_b: Metric observations from algorithm B across matching seeds.

    Returns:
        (statistic, p_value): Test statistic and two-tailed p-value.
    """
    a = np.asarray(data_a, dtype=np.float64).ravel()
    b = np.asarray(data_b, dtype=np.float64).ravel()

    if len(a) != len(b):
        raise ValueError(f"Array length mismatch: {len(a)} vs {len(b)}.")
    if len(a) == 0:
        raise ValueError("Input arrays must not be empty.")

    # Handle exact ties or identical distributions
    diffs = a - b
    if np.all(np.isclose(diffs, 0.0, atol=1e-12)):
        return 0.0, 1.0

    # Non-zero differences check
    non_zero_diffs = diffs[~np.isclose(diffs, 0.0, atol=1e-12)]
    if len(non_zero_diffs) == 0:
        return 0.0, 1.0

    res = scipy.stats.wilcoxon(a, b, alternative="two-sided")
    return float(res.statistic), float(res.pvalue)


def holm_bonferroni_correction(
    p_values: List[float],
    alpha: float = 0.05,
) -> List[Tuple[float, bool]]:
    """
    Holm-Bonferroni step-down correction for multiple comparisons (Deliverable B9).

    Sorts p-values ascending and tests hypothesis k against threshold:
        p_(k) <= alpha / (m - k + 1)
    Testing terminates at the first failure to reject.

    Args:
        p_values: List of raw p-values.
        alpha: Target family-wise error rate (default: 0.05).

    Returns:
        List of tuples (adjusted_p_value, is_significant) preserving original input order.
    """
    m = len(p_values)
    if m == 0:
        return []

    indexed_p = sorted(enumerate(p_values), key=lambda x: x[1])

    # Step-down testing
    rejected_ranks = set()
    for rank_idx, (orig_idx, p_val) in enumerate(indexed_p):
        k = rank_idx + 1
        threshold = alpha / (m - k + 1)
        if p_val <= threshold:
            rejected_ranks.add(rank_idx)
        else:
            break

    # Adjusted p-values with monotonicity enforcement
    adj_p_sorted = [0.0] * m
    running_max = 0.0
    for rank_idx, (orig_idx, p_val) in enumerate(indexed_p):
        k = rank_idx + 1
        curr_adj = min(1.0, (m - k + 1) * p_val)
        running_max = max(running_max, curr_adj)
        adj_p_sorted[rank_idx] = running_max

    # Map back to original order
    results = [None] * m
    for rank_idx, (orig_idx, _) in enumerate(indexed_p):
        results[orig_idx] = (float(adj_p_sorted[rank_idx]), rank_idx in rejected_ranks)

    return results


def calculate_cliffs_delta(
    data_a: Union[List[float], np.ndarray],
    data_b: Union[List[float], np.ndarray],
) -> float:
    """
    Compute Cliff's Delta non-parametric effect size (Deliverable B9).

        d = (#(a_i > b_j) - #(a_i < b_j)) / (n_a * n_b)

    Returns:
        float: Cliff's delta in range [-1.0, 1.0].
    """
    a = np.asarray(data_a, dtype=np.float64).ravel()
    b = np.asarray(data_b, dtype=np.float64).ravel()

    n_a = len(a)
    n_b = len(b)

    if n_a == 0 or n_b == 0:
        raise ValueError("Both input arrays must be non-empty.")

    # Vectorized pairwise differences matrix (n_a, n_b)
    diffs = a[:, None] - b[None, :]
    greater = np.sum(diffs > 1e-12)
    less = np.sum(diffs < -1e-12)

    return float((greater - less) / (n_a * n_b))


def summarize_metric(
    data: Union[List[float], np.ndarray],
) -> Dict[str, float]:
    """
    Summarize a performance metric distribution using Median and IQR (Deliverable B9).

    Returns:
        dict containing 'median', 'iqr' (q75 - q25), 'q25', 'q75', 'min', 'max', 'n'.
    """
    arr = np.asarray(data, dtype=np.float64).ravel()

    if len(arr) == 0:
        raise ValueError("Cannot summarize an empty array.")

    q25 = float(np.percentile(arr, 25))
    q75 = float(np.percentile(arr, 75))
    median = float(np.median(arr))
    iqr = float(q75 - q25)

    return {
        "median": median,
        "iqr": iqr,
        "q25": q25,
        "q75": q75,
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "n": len(arr),
    }
