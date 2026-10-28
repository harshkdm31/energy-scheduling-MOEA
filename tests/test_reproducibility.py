import numpy as np

from src.instance_generator import InstanceGenerator
from src.optimizer import SchedulingProblem, run_nsga2


def test_same_seed_reproduces_nsga2_objectives():
    instance = InstanceGenerator().generate(
        n_tasks=10,
        m_machines=3,
        alpha=2.0,
        seed=7,
    )

    def run_once():
        return run_nsga2(
            problem=SchedulingProblem(instance),
            pop_size=20,
            n_evals=100,
            seed=123,
        )

    first = run_once()
    second = run_once()

    # Compare the objective vectors without depending on their row order.
    first_F = first.F[np.lexsort((first.F[:, 1], first.F[:, 0]))]
    second_F = second.F[np.lexsort((second.F[:, 1], second.F[:, 0]))]

    assert first_F.shape == second_F.shape
    assert np.allclose(first_F, second_F)
    assert first.eval_count == second.eval_count