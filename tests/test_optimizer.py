"""
Unit tests for SchedulingProblem, EnergyAwareMutation, and NSGA-II optimizer engine.
"""

import numpy as np
import pytest

from src.instance_generator import InstanceGenerator
from src.operators import EnergyAwareMutation, UniformIntegerMutation
from src.optimizer import SchedulingProblem, run_nsga2


@pytest.fixture
def toy_instance():
    """Toy problem instance: n=12, m=3, alpha=2.0, seed=42."""
    generator = InstanceGenerator()
    return generator.generate(n_tasks=12, m_machines=3, alpha=2.0, seed=42)


def test_energy_aware_mutation_alleles_and_transition(toy_instance):
    """
    Test EnergyAwareMutation:
    - Verify all mutated alleles stay within [0, m-1].
    - When a task on an inefficient machine is mutated, verify it transitions to lower energy.
    """
    w = np.array(toy_instance.w)
    s = np.array(toy_instance.s)
    p = np.array(toy_instance.p)
    e = p / s
    m = toy_instance.m_machines
    n = toy_instance.n_tasks

    mutation = EnergyAwareMutation(w, s, p)

    # 1. Verify alleles stay within [0, m-1] over multiple population mutations
    pop_X = np.random.randint(0, m, size=(30, n))
    mut_X = mutation._do(None, pop_X)
    assert mut_X.shape == (30, n)
    assert np.all(mut_X >= 0) and np.all(mut_X < m), f"Alleles out of [0, {m-1}] bounds"

    # 2. Test transition from inefficient machine to lower energy state
    inefficient_machine = int(np.argmax(e))
    efficient_machine = int(np.argmin(e))

    # All tasks on the most inefficient machine
    x_inefficient = np.full(n, inefficient_machine, dtype=np.int64)

    for task_idx in range(n):
        x_mut = mutation.mutate_chromosome(x_inefficient, task_idx=task_idx)
        mutated_machine = x_mut[task_idx]

        assert 0 <= mutated_machine < m
        # Energy must strictly decrease since better machines exist
        assert e[mutated_machine] < e[inefficient_machine], (
            f"Expected lower energy state: {e[mutated_machine]} < {e[inefficient_machine]}"
        )

    # 3. Test fallback when on most efficient machine (no strictly lower machine exists)
    x_efficient = np.full(n, efficient_machine, dtype=np.int64)
    x_fallback = mutation.mutate_chromosome(x_efficient, task_idx=0)
    assert 0 <= x_fallback[0] < m


def test_run_nsga2_miniature_test(toy_instance):
    """
    Test run_nsga2:
    - Run a miniature test (N_pop=20, evals=100).
    - Verify non-empty res.F and res.X returned.
    - Verify all returned points satisfy analytical lower bounds (f1 >= LB_C, f2 >= LB_E).
    """
    problem = SchedulingProblem(toy_instance)

    res = run_nsga2(problem, pop_size=20, n_evals=100, seed=42)

    # 1. Non-empty return verification
    assert len(res.F) > 0, "res.F must not be empty"
    assert len(res.X) > 0, "res.X must not be empty"
    assert len(res.F) == len(res.X), "Length mismatch between objectives and schedules"
    assert res.F.shape[1] == 2, "Objectives must have 2 dimensions (makespan, energy)"
    assert res.X.shape[1] == toy_instance.n_tasks, "Schedule chromosome length must equal n_tasks"
    assert res.runtime > 0.0, "Wall-clock runtime must be positive"

    # 2. Integer alleles verification
    assert np.all(res.X >= 0) and np.all(res.X < toy_instance.m_machines), "Alleles must be in [0, m-1]"

    # 3. Analytical lower bound satisfaction
    lb_c = toy_instance.lower_bound_makespan
    lb_e = toy_instance.lower_bound_energy

    assert np.all(res.F[:, 0] >= lb_c - 1e-9), f"Makespan violates LB_C {lb_c}"
    assert np.all(res.F[:, 1] >= lb_e - 1e-9), f"Energy violates LB_E {lb_e}"

    # 4. Container unpacking and key access
    F, X, runtime = res
    assert np.array_equal(F, res.F)
    assert np.array_equal(X, res.X)
    assert res["F"] is res.F
    assert res["X"] is res.X


def test_run_nsga2_with_energy_aware_mutation(toy_instance):
    """Verify run_nsga2 functions seamlessly when explicitly configured with EnergyAwareMutation."""
    problem = SchedulingProblem(toy_instance)
    em = EnergyAwareMutation(toy_instance.w, toy_instance.s, toy_instance.p)

    res = run_nsga2(problem, pop_size=20, n_evals=100, mutation=em, seed=123)

    assert len(res.F) > 0
    assert len(res.X) > 0
    assert np.all(res.F[:, 0] >= toy_instance.lower_bound_makespan - 1e-9)
    assert np.all(res.F[:, 1] >= toy_instance.lower_bound_energy - 1e-9)


def test_scheduling_problem_input_validation():
    """Verify input validation and initialization of SchedulingProblem."""
    with pytest.raises(ValueError, match="must be non-empty"):
        SchedulingProblem([], [1.0], [2.0])

    with pytest.raises(ValueError, match="Mismatch"):
        SchedulingProblem([10.0, 20.0], [1.0, 2.0], [3.0])
