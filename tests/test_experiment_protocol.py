
from src.instance_generator import InstanceGenerator
from src.runner import load_run_result, run_experiment_batch


def test_nsga2_metadata_matches_saved_run(tmp_path):
    instance = InstanceGenerator().generate(
        n_tasks=10,
        m_machines=3,
        alpha=2.0,
        seed=42,
    )

    variants = ["nsga2_std", "nsga2_seeded"]
    seeds = [11, 22]

    paths = run_experiment_batch(
        instance=instance,
        variants=variants,
        seeds=seeds,
        output_dir=tmp_path,
        n_evals=100,
        pop_size=20,
    )

    assert len(paths) == len(variants) * len(seeds)

    observed = set()

    for path in paths:
        result = load_run_result(path)
        metadata = result["metadata"]

        assert metadata["instance_id"] == instance.instance_id
        assert metadata["variant"] in variants
        assert metadata["seed"] in seeds
        assert metadata["requested_eval_budget"] == 100
        assert metadata["eval_count"] == result["eval_count"]
        assert 0 < result["eval_count"] <= 100

        observed.add((metadata["variant"], metadata["seed"]))

    assert observed == {
        (variant, seed)
        for variant in variants
        for seed in seeds
    }
