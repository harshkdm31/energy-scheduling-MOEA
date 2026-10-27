"""
Deterministic baseline heuristics and weighted-greedy sweep for energy-aware task scheduling.
"""

from typing import List, Optional, Tuple, Union
import numpy as np


class WeightedSweepResult:
    """
    Container for weighted greedy sweep results.
    Supports unpacking (schedules, objectives), attribute access, and dict-like key access.
    """

    def __init__(
        self,
        schedules: np.ndarray,
        objectives: np.ndarray,
        weights: np.ndarray,
        all_schedules: Optional[np.ndarray] = None,
        all_objectives: Optional[np.ndarray] = None,
    ):
        self.schedules = schedules
        self.objectives = objectives
        self.weights = weights
        self.all_schedules = all_schedules
        self.all_objectives = all_objectives

    def __iter__(self):
        yield self.schedules
        yield self.objectives

    def __getitem__(self, item):
        if isinstance(item, str):
            if item in ("schedules", "solutions"):
                return self.schedules
            elif item in ("objectives", "front", "pareto_front"):
                return self.objectives
            elif item == "weights":
                return self.weights
            elif item == "all_schedules":
                return self.all_schedules
            elif item == "all_objectives":
                return self.all_objectives
            raise KeyError(item)
        if item == 0:
            return self.schedules
        elif item == 1:
            return self.objectives
        raise IndexError(f"Index {item} out of range for WeightedSweepResult")

    def __len__(self):
        return 2

    def __repr__(self) -> str:
        return f"WeightedSweepResult(n_pareto={len(self.objectives)}, n_schedules={len(self.schedules)})"


def compute_lower_bounds(
    w: Union[List[float], np.ndarray],
    s: Union[List[float], np.ndarray],
    p: Union[List[float], np.ndarray],
) -> Tuple[float, float]:
    """
    Compute closed-form analytical lower bounds on makespan (LB_C) and energy (LB_E):
        LB_C = max(sum(w) / sum(s), max(w) / max(s))
        LB_E = sum(w) * min_j (P_j / s_j)
    """
    w_arr = np.asarray(w, dtype=np.float64)
    s_arr = np.asarray(s, dtype=np.float64)
    p_arr = np.asarray(p, dtype=np.float64)

    total_w = float(np.sum(w_arr))
    total_s = float(np.sum(s_arr))
    e = p_arr / s_arr

    lb_c = float(max(total_w / total_s, np.max(w_arr) / np.max(s_arr)))
    lb_e = float(total_w * np.min(e))
    return lb_c, lb_e


def evaluate_objectives(
    chromosome: Union[List[int], np.ndarray],
    w: Union[List[float], np.ndarray],
    s: Union[List[float], np.ndarray],
    p: Union[List[float], np.ndarray],
) -> Tuple[float, float]:
    """
    Evaluate makespan (f1) and modeled active energy (f2) for a given schedule chromosome:
        f1(X) = max_{1 <= j <= m} C_j(X)
        f2(X) = sum_{j=1}^m P_j * C_j(X) = sum_{i=1}^n w_i * (P_{x_i} / s_{x_i})
    """
    w_arr = np.asarray(w, dtype=np.float64)
    s_arr = np.asarray(s, dtype=np.float64)
    p_arr = np.asarray(p, dtype=np.float64)
    chrom = np.asarray(chromosome, dtype=np.int64)
    m = len(s_arr)

    C = np.zeros(m, dtype=np.float64)
    for i, j in enumerate(chrom):
        C[j] += w_arr[i] / s_arr[j]

    f1 = float(np.max(C))
    f2 = float(np.sum(p_arr * C))
    return f1, f2


def solve_mct(
    w: Union[List[float], np.ndarray],
    s: Union[List[float], np.ndarray],
) -> np.ndarray:
    """
    Sequential Minimum Completion Time (MCT) greedy heuristic (Deliverable B4).

    Assigns each incoming task i to the machine j* that minimizes the resulting
    completion time:
        j* = argmin_j (C_j + w_i / s_j)

    Deterministic tie-breaking: Selects lowest machine index on collision.

    Returns:
        np.ndarray: Integer chromosome array X in [0, m-1]^n.
    """
    w_arr = np.asarray(w, dtype=np.float64)
    s_arr = np.asarray(s, dtype=np.float64)
    n = len(w_arr)
    m = len(s_arr)

    if n == 0:
        raise ValueError("Task workload array w must not be empty.")
    if m == 0:
        raise ValueError("Machine speed array s must not be empty.")
    if np.any(w_arr <= 0):
        raise ValueError("All task workloads w must be strictly positive.")
    if np.any(s_arr <= 0):
        raise ValueError("All machine speeds s must be strictly positive.")

    C = np.zeros(m, dtype=np.float64)
    X = np.zeros(n, dtype=np.int64)

    for i in range(n):
        ct = C + (w_arr[i] / s_arr)
        # np.argmin deterministically returns the first (lowest) index on ties
        best_j = int(np.argmin(ct))
        X[i] = best_j
        C[best_j] += w_arr[i] / s_arr[best_j]

    return X


def solve_energy_greedy(
    w: Union[List[float], np.ndarray],
    s: Union[List[float], np.ndarray],
    p: Union[List[float], np.ndarray],
) -> np.ndarray:
    """
    Single-objective energy greedy anchor heuristic (Deliverable B4).

    Assigns all tasks to the machine j* that minimizes specific energy cost:
        e_j = P_j / s_j
        j* = argmin_j (P_j / s_j)

    Deterministic tie-breaking: Selects lowest machine index on collision.

    Attains exact analytical lower bound LB_E = sum(w) * min_j(e_j).

    Returns:
        np.ndarray: Integer chromosome array X in [0, m-1]^n.
    """
    w_arr = np.asarray(w, dtype=np.float64)
    s_arr = np.asarray(s, dtype=np.float64)
    p_arr = np.asarray(p, dtype=np.float64)
    n = len(w_arr)
    m = len(s_arr)

    if n == 0 or m == 0:
        raise ValueError("Tasks and machines arrays must not be empty.")
    if len(p_arr) != m:
        raise ValueError(f"Mismatch: {m} machine speeds but {len(p_arr)} machine powers.")
    if np.any(w_arr <= 0) or np.any(s_arr <= 0) or np.any(p_arr <= 0):
        raise ValueError("Workloads, speeds, and powers must be strictly positive.")

    e = p_arr / s_arr
    best_j = int(np.argmin(e))
    return np.full(n, best_j, dtype=np.int64)


def filter_pareto_front(
    schedules: List[np.ndarray],
    objectives: List[Tuple[float, float]],
    tol: float = 1e-9,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Filter unique, non-dominated Pareto objective pairs and their corresponding schedules.

    Returns:
        pareto_schedules: np.ndarray of shape (k, n)
        pareto_objectives: np.ndarray of shape (k, 2), sorted ascending by makespan (f1)
    """
    if len(objectives) == 0:
        return np.empty((0, 0), dtype=np.int64), np.empty((0, 2), dtype=np.float64)

    # 1. Deduplicate objective points within tolerance
    unique_candidates: List[Tuple[np.ndarray, Tuple[float, float]]] = []
    for sched, (f1, f2) in zip(schedules, objectives):
        is_dup = False
        for _, (u_f1, u_f2) in unique_candidates:
            if abs(f1 - u_f1) <= tol and abs(f2 - u_f2) <= tol:
                is_dup = True
                break
        if not is_dup:
            unique_candidates.append((sched, (f1, f2)))

    # 2. Non-dominated filtering
    non_dominated: List[Tuple[np.ndarray, List[float]]] = []
    for i, (sched_i, (f1_i, f2_i)) in enumerate(unique_candidates):
        is_dominated = False
        for j, (_, (f1_j, f2_j)) in enumerate(unique_candidates):
            if i == j:
                continue
            # Point j dominates point i if f1_j <= f1_i and f2_j <= f2_i with at least one strict inequality
            if (f1_j <= f1_i + tol) and (f2_j <= f2_i + tol) and (
                (f1_j < f1_i - tol) or (f2_j < f2_i - tol)
            ):
                is_dominated = True
                break
        if not is_dominated:
            non_dominated.append((sched_i, [f1_i, f2_i]))

    # 3. Sort ascending by makespan f1
    non_dominated.sort(key=lambda item: item[1][0])

    pareto_schedules = np.array([item[0] for item in non_dominated], dtype=np.int64)
    pareto_objectives = np.array([item[1] for item in non_dominated], dtype=np.float64)
    return pareto_schedules, pareto_objectives


def solve_weighted_greedy_sweep(
    w: Union[List[float], np.ndarray],
    s: Union[List[float], np.ndarray],
    p: Union[List[float], np.ndarray],
    num_weights: int = 21,
) -> WeightedSweepResult:
    """
    Weighted-Greedy Sweep heuristic (Deliverable B7).

    Algorithm:
    1. Compute closed-form analytical lower bounds:
       LB_C = max(sum(w) / sum(s), max(w) / max(s))
       LB_E = sum(w) * min_j (P_j / s_j)
    2. Sort tasks in descending workload (LPT) order: w_(1) >= w_(2) >= ... >= w_(n).
    3. Sweep lambda in linspace(0.0, 1.0, num_weights).
    4. For each lambda, sequentially assign incoming task i (in LPT order) to machine j minimizing:
       g_j = lambda * ((C_j + w_i / s_j) / LB_C) + (1 - lambda) * ((w_i * e_j) / LB_E)
       where e_j = P_j / s_j.
       Tie-breaker: lowest machine index.
    5. Evaluate makespan (f1) and modeled energy (f2) for each schedule.
    6. Filter and return unique non-dominated Pareto objective pairs and their schedules.

    Returns:
        WeightedSweepResult: Container with filtered pareto schedules and objectives,
                             unpackable as `schedules, objectives`.
    """
    w_arr = np.asarray(w, dtype=np.float64)
    s_arr = np.asarray(s, dtype=np.float64)
    p_arr = np.asarray(p, dtype=np.float64)
    n = len(w_arr)
    m = len(s_arr)

    if n == 0 or m == 0:
        raise ValueError("Tasks and machines arrays must not be empty.")
    if len(p_arr) != m:
        raise ValueError(f"Mismatch: {m} machine speeds but {len(p_arr)} machine powers.")
    if np.any(w_arr <= 0) or np.any(s_arr <= 0) or np.any(p_arr <= 0):
        raise ValueError("Workloads, speeds, and powers must be strictly positive.")
    if num_weights <= 0:
        raise ValueError(f"num_weights must be positive, got {num_weights}")

    # 1. Closed-form analytical lower bounds
    lb_c, lb_e = compute_lower_bounds(w_arr, s_arr, p_arr)
    e = p_arr / s_arr

    # 2. Stable LPT task ordering (descending workload)
    task_order = np.argsort(-w_arr, kind="stable")

    # 3. Sweep lambda over linspace(0.0, 1.0, num_weights)
    weights = np.linspace(0.0, 1.0, num_weights)

    all_schedules: List[np.ndarray] = []
    all_objectives: List[Tuple[float, float]] = []

    for lam in weights:
        C = np.zeros(m, dtype=np.float64)
        X = np.zeros(n, dtype=np.int64)

        for orig_idx in task_order:
            w_i = w_arr[orig_idx]
            # Scalarized cost for each machine j:
            # g_j = lam * ((C_j + w_i / s_j) / LB_C) + (1 - lam) * ((w_i * e_j) / LB_E)
            makespan_term = (C + (w_i / s_arr)) / lb_c
            energy_term = (w_i * e) / lb_e
            g = lam * makespan_term + (1.0 - lam) * energy_term

            # Tie-breaker: lowest machine index
            best_j = int(np.argmin(g))
            X[orig_idx] = best_j
            C[best_j] += w_i / s_arr[best_j]

        f1, f2 = evaluate_objectives(X, w_arr, s_arr, p_arr)
        all_schedules.append(X)
        all_objectives.append((f1, f2))

    # 4. Filter non-dominated Pareto front
    pareto_schedules, pareto_objectives = filter_pareto_front(all_schedules, all_objectives)

    return WeightedSweepResult(
        schedules=pareto_schedules,
        objectives=pareto_objectives,
        weights=weights,
        all_schedules=np.array(all_schedules, dtype=np.int64),
        all_objectives=np.array(all_objectives, dtype=np.float64),
    )
