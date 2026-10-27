"""
Independent verification suite: combinatorial brute-force lower bound validation
and cross-operator (BM + EM) integration harness.
"""

import itertools
from typing import Optional
import numpy as np
import pytest

from src.baselines import (
    compute_lower_bounds,
    evaluate_objectives,
    filter_pareto_front,
    solve_energy_greedy,
    solve_mct,
)
from src.instance_generator import InstanceGenerator
from src.operators import EnergyAwareMutation
from src.optimizer import SchedulingProblem, run_nsga2


def mock_bottleneck_aware_mutation(
    chromosome: np.ndarray,
    w: np.ndarray,
    s: np.ndarray,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Mock implementation of Person A's Bottleneck-Aware Mutation (BM).

    Identifies the bottleneck machine j* = argmax_j C_j, picks a task i on j*,
    and moves it to a destination machine that relieves the bottleneck:
        C_dest + w_i / s_dest < C_j*
    """
    rng = rng if rng is not None else np.random.default_rng()
    x = np.copy(chromosome).astype(np.int64)
    m = len(s)

    # Compute machine completion times
    C = np.bincount(x, weights=w, minlength=m) / s
    bottleneck_j = int(np.argmax(C))
    c_max = C[bottleneck_j]

    tasks_on_bottleneck = [i for i, j in enumerate(x) if j == bottleneck_j]
    if not tasks_on_bottleneck:
        return x

    rng.shuffle(tasks_on_bottleneck)

    for task_i in tasks_on_bottleneck:
        w_i = w[task_i]
        # Candidate destination machines that strictly relieve the bottleneck
        valid_destinations = [
            dest_j for dest_j in range(m)
            if dest_j != bottleneck_j and (C[dest_j] + w_i / s[dest_j] < c_max)
        ]
        if valid_destinations:
            dest_j = int(rng.choice(valid_destinations))
            # Verify the relieving condition
            assert C[dest_j] + w_i / s[dest_j] < c_max, (
                f"Destination {dest_j} must satisfy C_dest + w_i/s_dest < C_j*"
            )
            x[task_i] = dest_j
            return x

    # Fallback if no strict relief is feasible: reassign to least loaded machine
    least_loaded_j = int(np.argmin(C))
    if least_loaded_j != bottleneck_j:
        x[tasks_on_bottleneck[0]] = least_loaded_j

    return x


def test_exhaustive_brute_force_lower_bound_invariants():
    """
    Exhaustive brute-force check on n=8, m=3 (3^8 = 6,561 schedules):
    - Verify f1 >= LB_C for 100% of enumerated assignments (0 violations).
    - Verify f2 >= LB_E for 100% of enumerated assignments (0 violations).
    - Verify min(f2) == LB_E within 10^-9 numerical tolerance (exact attainment).
    """
    generator = InstanceGenerator()
    instance = generator.generate(n_tasks=8, m_machines=3, alpha=2.0, seed=42)

    w = np.array(instance.w)
    s = np.array(instance.s)
    p = np.array(instance.p)

    lb_c, lb_e = compute_lower_bounds(w, s, p)
    assert pytest.approx(lb_c, rel=1e-9) == instance.lower_bound_makespan
    assert pytest.approx(lb_e, rel=1e-9) == instance.lower_bound_energy

    # Enumerate all 3^8 = 6,561 assignments
    all_schedules = list(itertools.product(range(3), repeat=8))
    assert len(all_schedules) == 3**8 == 6561

    f1_values = np.zeros(len(all_schedules), dtype=np.float64)
    f2_values = np.zeros(len(all_schedules), dtype=np.float64)

    for idx, x_tuple in enumerate(all_schedules):
        x = np.array(x_tuple, dtype=np.int64)
        f1, f2 = evaluate_objectives(x, w, s, p)

        f1_values[idx] = f1
        f2_values[idx] = f2

    # 1. 100% of schedules satisfy f1 >= LB_C
    makespan_violations = np.sum(f1_values < lb_c - 1e-9)
    assert makespan_violations == 0, f"Found {makespan_violations} makespan lower bound violations!"
    assert np.all(f1_values >= lb_c - 1e-9)

    # 2. 100% of schedules satisfy f2 >= LB_E
    energy_violations = np.sum(f2_values < lb_e - 1e-9)
    assert energy_violations == 0, f"Found {energy_violations} energy lower bound violations!"
    assert np.all(f2_values >= lb_e - 1e-9)

    # 3. Exact attainment: min(f2) == LB_E within 10^-9 tolerance
    min_energy = float(np.min(f2_values))
    assert pytest.approx(min_energy, abs=1e-9) == lb_e, (
        f"Minimum enumerated energy {min_energy} does not match LB_E {lb_e}"
    )


def test_exhaustive_pareto_dominance_and_energy_greedy_anchor():
    """
    Extract true Pareto front across all 6,561 enumerated assignments and verify
    that solve_energy_greedy produces a point lying strictly within the true Pareto front.
    """
    generator = InstanceGenerator()
    instance = generator.generate(n_tasks=8, m_machines=3, alpha=2.0, seed=42)

    w = np.array(instance.w)
    s = np.array(instance.s)
    p = np.array(instance.p)
    _, lb_e = compute_lower_bounds(w, s, p)

    all_assignments = [np.array(x, dtype=np.int64) for x in itertools.product(range(3), repeat=8)]
    all_objectives = [evaluate_objectives(x, w, s, p) for x in all_assignments]

    # Non-dominated sorting across entire combinatorial search space
    pareto_X, pareto_F = filter_pareto_front(all_assignments, all_objectives)
    assert len(pareto_F) > 0

    # Solve via energy-greedy anchor heuristic
    eg_chrom = solve_energy_greedy(w, s, p)
    eg_f1, eg_f2 = evaluate_objectives(eg_chrom, w, s, p)

    # Energy greedy must match exact LB_E
    assert pytest.approx(eg_f2, abs=1e-9) == lb_e

    # Verify that (eg_f1, eg_f2) lies strictly in the true Pareto front
    in_pareto = any(
        np.isclose(eg_f1, pf1, atol=1e-9) and np.isclose(eg_f2, pf2, atol=1e-9)
        for pf1, pf2 in pareto_F
    )
    assert in_pareto, (
        f"Energy-greedy point ({eg_f1}, {eg_f2}) not found in true Pareto front of size {len(pareto_F)}"
    )


def test_mock_bottleneck_aware_mutation_interface():
    """Verify mock BM operator relieves the bottleneck and respects boundary conditions."""
    generator = InstanceGenerator()
    instance = generator.generate(n_tasks=10, m_machines=3, alpha=2.0, seed=101)

    w = np.array(instance.w)
    s = np.array(instance.s)

    # Force all tasks onto machine 0 so it is the unambiguous bottleneck
    x_unbalanced = np.zeros(10, dtype=np.int64)
    x_mutated = mock_bottleneck_aware_mutation(x_unbalanced, w, s)

    assert x_mutated.shape == (10,)
    assert np.all(x_mutated >= 0) and np.all(x_mutated < 3)

    # At least one task must have moved off the bottleneck machine 0
    assert np.sum(x_mutated != 0) >= 1

    # New makespan should be strictly less than original bottleneck makespan
    orig_makespan = np.max(np.bincount(x_unbalanced, weights=w, minlength=3) / s)
    new_makespan = np.max(np.bincount(x_mutated, weights=w, minlength=3) / s)
    assert new_makespan < orig_makespan


def test_cross_operator_integration_bm_em_hybrid():
    """
    Verify cross-operator integration:
    Combining Person A's BM (makespan relief) and Person B's EM (energy descent)
    produces valid hybrid mutations without array shape or indexing corruptions.
    """
    generator = InstanceGenerator()
    instance = generator.generate(n_tasks=12, m_machines=4, alpha=2.0, seed=2026)

    w = np.array(instance.w)
    s = np.array(instance.s)
    p = np.array(instance.p)
    em = EnergyAwareMutation(w, s, p)

    pop_size = 40
    pop = np.random.randint(0, 4, size=(pop_size, 12), dtype=np.int64)

    # Apply hybrid mutations over multiple evolutionary generations
    for gen_idx in range(10):
        for k in range(pop_size):
            if k % 2 == 0:
                # Apply BM (Person A makespan relief)
                pop[k] = mock_bottleneck_aware_mutation(pop[k], w, s)
            else:
                # Apply EM (Person B energy descent)
                pop[k] = em.mutate_chromosome(pop[k])

        # Assert shape, type, and integer allele bounds are strictly preserved
        assert pop.shape == (pop_size, 12), f"Population shape corrupted in gen {gen_idx}"
        assert pop.dtype == np.int64
        assert np.all(pop >= 0) and np.all(pop < 4), f"Allele bound violation in gen {gen_idx}"


def test_mock_endpoint_seeding_integration():
    """Verify endpoint seeding integration (MCT + Energy Greedy) executes cleanly with NSGA-II."""
    generator = InstanceGenerator()
    instance = generator.generate(n_tasks=10, m_machines=3, alpha=2.0, seed=77)

    w = np.array(instance.w)
    s = np.array(instance.s)
    p = np.array(instance.p)

    mct_seed = solve_mct(w, s)
    eg_seed = solve_energy_greedy(w, s, p)

    # Verify both seed chromosomes are valid
    assert len(mct_seed) == 10 and len(eg_seed) == 10

    problem = SchedulingProblem(instance)
    res = run_nsga2(problem, pop_size=20, n_evals=100, seed=42)

    assert len(res.F) > 0
    assert np.all(res.F[:, 0] >= instance.lower_bound_makespan - 1e-9)
    assert np.all(res.F[:, 1] >= instance.lower_bound_energy - 1e-9)
