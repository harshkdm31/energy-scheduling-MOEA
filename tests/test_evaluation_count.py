
from src.instance_generator import InstanceGenerator
from src.optimizer import SchedulingProblem, run_nsga2


def test_optimizer_reports_actual_evaluation_count():
    instance = InstanceGenerator().generate(
        n_tasks=12,
        m_machines=3,
        alpha=2.0,
        seed=7,
    )
    problem = SchedulingProblem(instance)

    result = run_nsga2(
        problem=problem,
        pop_size=10,
        n_evals=40,
        seed=7,
    )

    assert result.eval_count > 0
    assert result.eval_count == result.raw_result.algorithm.evaluator.n_eval
    assert result.eval_count >= 10