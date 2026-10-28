
from src.instance_generator import InstanceGenerator
from src.runner import load_run_result, run_experiment_batch


def make_instance():
    return InstanceGenerator().generate(
        n_tasks=8,
        m_machines=2,
        alpha=2.0,
        seed=7,
    )


def test_random_baseline_metadata(tmp_path):
    instance = make_instance()

    paths = run_experiment_batch(
        instance=instance,
        variants=["random"],
        seeds=[11, 22],
        output_dir=tmp_path,
        n_evals=50,
    )

    assert len(paths) == 2

    for path in paths:
        result = load_run_result(path)
        metadata = result["metadata"]

        assert metadata["variant"] == "random"
        assert metadata["instance_id"] == instance.instance_id
        assert metadata["seed"] in {11, 22}
        assert metadata["eval_count"] == 50
        assert result["eval_count"] == 50
        assert metadata["requested_eval_budget"] == 50
        assert metadata["eval_count_type"] == "objective_evaluations"
        assert metadata["stochastic"] is True
        assert metadata["deterministic_per_instance"] is False
        assert metadata["statistical_unit"] == "instance_seed"


def test_weighted_greedy_baseline_metadata(tmp_path):
    instance = make_instance()

    paths = run_experiment_batch(
        instance=instance,
        variants=["weighted_greedy"],
        seeds=[11, 22],
        output_dir=tmp_path,
        n_evals=50,
    )

    assert len(paths) == 2

    results = [load_run_result(path) for path in paths]

    for result in results:
        metadata = result["metadata"]

        assert metadata["variant"] == "weighted_greedy"
        assert metadata["instance_id"] == instance.instance_id
        assert metadata["seed"] in {11, 22}
        assert metadata["eval_count"] == 21
        assert result["eval_count"] == 21
        assert metadata["requested_eval_budget"] == 50
        assert metadata["eval_count_type"] == "scalarization_candidates"
        assert metadata["stochastic"] is False
        assert metadata["deterministic_per_instance"] is True
        assert metadata["statistical_unit"] == "instance"

    # Seed-labelled archives should contain the same deterministic front.
    first = results[0]["non_dominated_F"]
    second = results[1]["non_dominated_F"]

    assert first.shape == second.shape
    assert set(map(tuple, first)) == set(map(tuple, second))
