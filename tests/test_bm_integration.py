
import numpy as np

from src.instance_generator import InstanceGenerator
from src.operators import BottleneckAwareMutation
from src.optimizer import SchedulingProblem, run_nsga2


def test_bm_runs_through_nsga2():
    instance = InstanceGenerator().generate(
        n_tasks=12,
        m_machines=3,
        alpha=2.0,
        seed=7,
    )

    problem = SchedulingProblem(instance)
    mutation = BottleneckAwareMutation(
        w=instance.w,
        s=instance.s,
        prob=1.0,
    )

    result = run_nsga2(
        problem=problem,
        pop_size=10,
        n_evals=40,
        mutation=mutation,
        seed=7,
    )

    assert result.F.ndim == 2
    assert result.F.shape[1] == 2
    assert result.X.ndim == 2
    assert result.X.shape[1] == instance.n_tasks
    assert len(result.F) == len(result.X)
    assert len(result.F) > 0

    assert np.all(result.X >= 0)
    assert np.all(result.X < instance.m_machines)
    assert np.all(np.isfinite(result.F))

    assert np.all(
        result.F[:, 0] >= instance.lower_bound_makespan - 1e-9
    )
    assert np.all(
        result.F[:, 1] >= instance.lower_bound_energy - 1e-9
    )
