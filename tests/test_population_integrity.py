
import numpy as np

from src.instance_generator import InstanceGenerator
from src.optimizer import (
    EndpointSeededSampling,
    SchedulingProblem,
    run_nsga2,
)
from src.seeding import build_endpoint_schedules


def make_instance():
    return InstanceGenerator().generate(
        n_tasks=12,
        m_machines=3,
        alpha=2.0,
        seed=7,
    )


def test_endpoint_seeds_are_unique():
    instance = make_instance()

    seeds = build_endpoint_schedules(
        instance.w,
        instance.s,
        instance.p,
    )

    seed_matrix = np.asarray(seeds, dtype=np.int64)

    assert seed_matrix.ndim == 2
    assert len(seed_matrix) >= 1
    assert len({tuple(row) for row in seed_matrix}) == len(seed_matrix)


def test_seeded_sampling_preserves_seeds_and_population_size():
    instance = make_instance()
    problem = SchedulingProblem(instance)

    seeds = np.asarray(
        build_endpoint_schedules(
            instance.w,
            instance.s,
            instance.p,
        ),
        dtype=np.int64,
    )

    population_size = 10
    sampler = EndpointSeededSampling(seeds)

    population = sampler.do(
        problem,
        population_size,
        random_state=np.random.default_rng(123),
    )

    population = np.asarray(population.get("X"), dtype=np.int64)

    assert population.shape == (population_size, instance.n_tasks)
    assert np.all(population >= 0)
    assert np.all(population < instance.m_machines)

    # Endpoint seeds should occupy the first population slots.
    assert np.array_equal(population[:len(seeds)], seeds)


def test_nsga2_final_population_has_no_duplicate_chromosomes():
    instance = make_instance()
    problem = SchedulingProblem(instance)

    seeds = np.asarray(
        build_endpoint_schedules(
            instance.w,
            instance.s,
            instance.p,
        ),
        dtype=np.int64,
    )

    result = run_nsga2(
        problem=problem,
        pop_size=10,
        n_evals=100,
        seed=7,
        initial_population=seeds,
    )

    final_population = np.asarray(
        result.raw_result.algorithm.pop.get("X"),
        dtype=np.int64,
    )

    assert final_population.ndim == 2
    assert final_population.shape[1] == instance.n_tasks
    assert len(final_population) == 10

    unique_rows = {
        tuple(row.tolist()) for row in final_population
    }
    assert len(unique_rows) == len(final_population)
