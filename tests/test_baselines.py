"""
Unit tests for deterministic baselines: MCT, energy-greedy, and weighted-greedy sweep.
"""

import numpy as np
import pytest

from src.baselines import (
    compute_lower_bounds,
    evaluate_objectives,
    filter_pareto_front,
    solve_energy_greedy,
    solve_mct,
    solve_weighted_greedy_sweep,
)
from src.instance_generator import InstanceGenerator


def test_solve_energy_greedy_hits_lower_bound():
    """Test solve_energy_greedy: evaluated energy must equal LB_E within 1e-9 tolerance."""
    generator = InstanceGenerator()

    for seed in [42, 100, 2026]:
        instance = generator.generate(n_tasks=50, m_machines=5, alpha=2.0, kappa=1.0, seed=seed)
        chrom = solve_energy_greedy(instance.w, instance.s, instance.p)

        assert chrom.shape == (50,), f"Expected shape (50,), got {chrom.shape}"
        assert np.all(chrom >= 0) and np.all(chrom < 5)

        # Evaluated energy
        f1, f2 = evaluate_objectives(chrom, instance.w, instance.s, instance.p)

        # LB_E from instance and compute_lower_bounds
        _, lb_e = compute_lower_bounds(instance.w, instance.s, instance.p)

        assert pytest.approx(f2, abs=1e-9) == lb_e
        assert pytest.approx(f2, abs=1e-9) == instance.lower_bound_energy


def test_solve_mct_shape_and_alleles():
    """Test solve_mct: chromosome shape (n,) and all alleles in [0, m-1]."""
    generator = InstanceGenerator()
    test_cases = [
        (20, 3),
        (50, 5),
        (100, 10),
    ]

    for n_tasks, m_machines in test_cases:
        instance = generator.generate(n_tasks=n_tasks, m_machines=m_machines, seed=42)
        chrom = solve_mct(instance.w, instance.s)

        assert chrom.shape == (n_tasks,), f"Expected shape ({n_tasks},), got {chrom.shape}"
        assert chrom.dtype == np.int64
        assert np.all(chrom >= 0) and np.all(chrom < m_machines), (
            f"All alleles must be in [0, {m_machines - 1}]"
        )

        f1, f2 = evaluate_objectives(chrom, instance.w, instance.s, instance.p)
        assert f1 >= instance.lower_bound_makespan - 1e-9
        assert f2 >= instance.lower_bound_energy - 1e-9


def test_solve_weighted_greedy_sweep_pareto_optimality_and_bounds():
    """
    Test solve_weighted_greedy_sweep:
    - Returns non-dominated Pareto front.
    - All points satisfy analytical lower bounds: f1 >= LB_C, f2 >= LB_E.
    - Unpacks cleanly as `schedules, objectives`.
    """
    generator = InstanceGenerator()
    instance = generator.generate(n_tasks=60, m_machines=6, alpha=2.0, kappa=1.0, seed=123)

    res = solve_weighted_greedy_sweep(instance.w, instance.s, instance.p, num_weights=21)
    schedules, objectives = res

    assert len(schedules) == len(objectives)
    assert len(objectives) > 0, "Pareto front must not be empty"

    lb_c, lb_e = compute_lower_bounds(instance.w, instance.s, instance.p)

    # All points must satisfy analytical lower bounds
    for i in range(len(objectives)):
        f1, f2 = objectives[i]
        assert f1 >= lb_c - 1e-9, f"Makespan {f1} violates LB_C {lb_c}"
        assert f2 >= lb_e - 1e-9, f"Energy {f2} violates LB_E {lb_e}"

        # Cross-check objective evaluation with schedule
        eval_f1, eval_f2 = evaluate_objectives(schedules[i], instance.w, instance.s, instance.p)
        assert pytest.approx(eval_f1, abs=1e-9) == f1
        assert pytest.approx(eval_f2, abs=1e-9) == f2

    # Check non-domination property: no point in front dominates any other point
    k = len(objectives)
    for i in range(k):
        f1_i, f2_i = objectives[i]
        for j in range(k):
            if i == j:
                continue
            f1_j, f2_j = objectives[j]
            # Check if j strictly dominates i
            strictly_dominates = (
                (f1_j <= f1_i + 1e-9)
                and (f2_j <= f2_i + 1e-9)
                and ((f1_j < f1_i - 1e-9) or (f2_j < f2_i - 1e-9))
            )
            assert not strictly_dominates, f"Point {j} ({f1_j}, {f2_j}) dominates point {i} ({f1_i}, {f2_i})"

    # Check sorting: makespan non-decreasing, energy non-increasing
    for i in range(k - 1):
        assert objectives[i, 0] <= objectives[i + 1, 0] + 1e-9
        assert objectives[i, 1] >= objectives[i + 1, 1] - 1e-9


def test_determinism():
    """Assert identical inputs produce identical chromosomes and objectives across separate calls."""
    generator = InstanceGenerator()
    instance = generator.generate(n_tasks=45, m_machines=5, seed=999)

    # 1. MCT determinism
    mct1 = solve_mct(instance.w, instance.s)
    mct2 = solve_mct(instance.w, instance.s)
    assert np.array_equal(mct1, mct2)

    # 2. Energy greedy determinism
    eg1 = solve_energy_greedy(instance.w, instance.s, instance.p)
    eg2 = solve_energy_greedy(instance.w, instance.s, instance.p)
    assert np.array_equal(eg1, eg2)

    # 3. Weighted sweep determinism
    res1 = solve_weighted_greedy_sweep(instance.w, instance.s, instance.p, num_weights=15)
    res2 = solve_weighted_greedy_sweep(instance.w, instance.s, instance.p, num_weights=15)
    assert np.array_equal(res1.schedules, res2.schedules)
    assert np.allclose(res1.objectives, res2.objectives, atol=1e-12)


def test_deterministic_tie_breaking_lowest_machine_index():
    """Verify lowest machine index is consistently selected on collisions."""
    # Identical speeds and powers
    w = np.array([10.0, 10.0])
    s = np.array([2.0, 2.0])
    p = np.array([4.0, 4.0])

    # In MCT:
    # Task 0: C_0 + 5 = 5, C_1 + 5 = 5 -> Tie broken to machine 0
    # Task 1: C_0 + 5 = 10, C_1 + 5 = 5 -> Machine 1
    mct_chrom = solve_mct(w, s)
    assert np.array_equal(mct_chrom, np.array([0, 1]))

    # In Energy-Greedy:
    # e_0 = 2.0, e_1 = 2.0 -> Tie broken to machine 0 for all tasks
    eg_chrom = solve_energy_greedy(w, s, p)
    assert np.array_equal(eg_chrom, np.array([0, 0]))


def test_worked_example_from_docs():
    """Verify heuristics on the worked example from docs/mathematical_model.md."""
    w = np.array([4.0, 6.0])
    s = np.array([2.0, 4.0])
    p = np.array([3.0, 8.0])

    # Machine efficiencies: e_0 = 3/2 = 1.5, e_1 = 8/4 = 2.0
    # Energy greedy should assign both tasks to machine 0
    eg_chrom = solve_energy_greedy(w, s, p)
    assert np.array_equal(eg_chrom, np.array([0, 0]))
    f1_eg, f2_eg = evaluate_objectives(eg_chrom, w, s, p)
    assert f1_eg == 5.0
    assert f2_eg == 15.0  # Exactly LB_E

    # MCT should assign task 0 (w=4): M0: 4/2=2, M1: 4/4=1 -> M1
    # task 1 (w=6): M0: 0 + 6/2=3, M1: 1 + 6/4=2.5 -> M1
    mct_chrom = solve_mct(w, s)
    assert np.array_equal(mct_chrom, np.array([1, 1]))
    f1_mct, f2_mct = evaluate_objectives(mct_chrom, w, s, p)
    assert f1_mct == 2.5
    assert f2_mct == 20.0

    # Weighted sweep should produce non-dominated points
    res = solve_weighted_greedy_sweep(w, s, p, num_weights=11)
    for f1, f2 in res.objectives:
        assert f1 >= 10.0 / 6.0 - 1e-9
        assert f2 >= 15.0 - 1e-9


def test_input_validation():
    """Verify proper exceptions are raised for invalid inputs."""
    with pytest.raises(ValueError, match="must not be empty"):
        solve_mct([], [1.0, 2.0])

    with pytest.raises(ValueError, match="strictly positive"):
        solve_mct([-1.0, 5.0], [1.0, 2.0])

    with pytest.raises(ValueError, match="Mismatch"):
        solve_energy_greedy([5.0], [1.0, 2.0], [3.0])

    with pytest.raises(ValueError, match="num_weights must be positive"):
        solve_weighted_greedy_sweep([5.0], [1.0], [2.0], num_weights=0)
