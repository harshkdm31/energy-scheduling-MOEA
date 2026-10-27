
import numpy as np
import pytest

from src.instance_generator import InstanceGenerator
from src.runner import load_run_result, run_experiment_batch


@pytest.mark.parametrize(
    "variant",
    [
        "nsga2_seeded",
        "nsga2_seeded_bm",
        "nsga2_seeded_em",
    ],
)
def test_seeded_runner_variants(tmp_path, variant):
    instance = InstanceGenerator().generate(
        n_tasks=12,
        m_machines=3,
        alpha=2.0,
        seed=7,
    )

    paths = run_experiment_batch(
        instance=instance,
        variants=[variant],
        seeds=[7],
        output_dir=tmp_path,
        n_evals=40,
        pop_size=10,
    )

    assert len(paths) == 1
    assert paths[0].exists()

    result = load_run_result(paths[0])
    F = result["non_dominated_F"]
    X = result["non_dominated_X"]
    metadata = result["metadata"]

    assert F.ndim == 2
    assert F.shape[1] == 2
    assert X.ndim == 2
    assert X.shape[1] == instance.n_tasks
    assert len(F) == len(X) > 0
    assert np.all(np.isfinite(F))
    assert np.all(X >= 0)
    assert np.all(X < instance.m_machines)

    assert metadata["variant"] == variant
    assert metadata["endpoint_seeding"] is True
    assert metadata["endpoint_seed_count"] >= 1
