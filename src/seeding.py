
"""Endpoint seed construction for multi-objective task scheduling."""

import numpy as np


def build_endpoint_schedules(w, s, p):
    """
    Construct makespan-oriented and energy-optimal schedules.

    Returns:
        A list containing unique integer chromosomes:
        - makespan-oriented schedule using LPT task ordering
        - energy-optimal schedule assigning all tasks to a minimum-energy-cost
          machine, where energy cost per unit work is p[j] / s[j]
    """
    w = np.asarray(w, dtype=np.float64)
    s = np.asarray(s, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)

    if w.ndim != 1 or len(w) == 0:
        raise ValueError("w must be a non-empty one-dimensional array.")
    if s.ndim != 1 or len(s) == 0:
        raise ValueError("s must be a non-empty one-dimensional array.")
    if p.shape != s.shape:
        raise ValueError("p and s must have the same shape.")
    if not np.all(np.isfinite(w)) or np.any(w <= 0):
        raise ValueError("Workloads must be finite and positive.")
    if not np.all(np.isfinite(s)) or np.any(s <= 0):
        raise ValueError("Speeds must be finite and positive.")
    if not np.all(np.isfinite(p)) or np.any(p <= 0):
        raise ValueError("Powers must be finite and positive.")

    n = len(w)
    m = len(s)

    # Makespan-oriented seed: LPT order, choosing the machine with
    # the smallest projected completion time for each task.
    loads = np.zeros(m, dtype=np.float64)
    makespan_seed = np.zeros(n, dtype=np.int64)

    for task in np.argsort(-w, kind="stable"):
        projected = loads + w[task] / s
        machine = int(np.argmin(projected))
        makespan_seed[task] = machine
        loads[machine] += w[task] / s[machine]

    # Energy-optimal seed: use the machine with minimum energy per
    # workload unit, e_j = p_j / s_j, for every task.
    energy_machine = int(np.argmin(p / s))
    energy_seed = np.full(n, energy_machine, dtype=np.int64)

    # Avoid duplicate endpoint chromosomes.
    seeds = []
    seen = set()

    for seed in (makespan_seed, energy_seed):
        key = tuple(seed.tolist())
        if key not in seen:
            seeds.append(seed)

            seen.add(key)

    return seeds
