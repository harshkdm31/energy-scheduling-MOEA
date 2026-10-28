
import numpy as np
import pytest

from src.instance_generator import InstanceGenerator
from src.optimizer import SchedulingProblem, run_nsga2
from src.seeding import build_endpoint_schedules


def test_endpoint_seeds_are_valid():
    instance = InstanceGenerator().generate(
        n_tasks=12, m_machines=3, alpha=2.0, seed=7
    )

    seeds = build_endpoint_schedules(
        instance.w, instance.s, instance.p
    )

    assert len(seeds) >= 1

    for seed in seeds:
        assert seed.shape == (instance.n_tasks,)
        assert np.issubdtype(seed.dtype, np.integer)
        assert np.all(seed >= 0)
        assert np.all(seed < instance.m_machines)


def test_energy_seed_attains_energy_lower_bound():
    instance = InstanceGenerator().generate(
        n_tasks=12, m_machines=3, alpha=2.0, seed=7
    )

    seeds = build_endpoint_schedules(
        instance.w, instance.s, instance.p
    )
    problem = SchedulingProblem(instance)

    energy_seed = np.full(
        instance.n_tasks,
        int(np.argmin(np.asarray(instance.p) / np.asarray(instance.s))),
    )

    out = {}
    problem._evaluate(energy_seed, out)

    assert out["F"][1] == pytest.approx(
        problem.lb_energy, rel=1e-10, abs=1e-10
    )


def test_seeded_nsga2_runs():
    instance = InstanceGenerator().generate(
        n_tasks=12, m_machines=3, alpha=2.0, seed=7
    )

    problem = SchedulingProblem(instance)
    seeds = build_endpoint_schedules(
        instance.w, instance.s, instance.p
    )

    result = run_nsga2(
        problem=problem,
        pop_size=10,
        n_evals=40,
        seed=7,
        initial_population=np.asarray(seeds),
    )

    assert result.F.ndim == 2
    assert result.F.shape[1] == 2
    assert result.X.ndim == 2
    assert result.X.shape[1] == instance.n_tasks
    assert len(result.F) == len(result.X) > 0
    assert np.all(np.isfinite(result.F))
    assert np.all(result.X >= 0)
    assert np.all(result.X < instance.m_machines)
