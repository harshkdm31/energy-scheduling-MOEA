"""
Unit tests for experiment runner, npz persistence, and statistical analysis engine.
"""

from pathlib import Path
import numpy as np
import pytest

from src.instance_generator import InstanceGenerator
from src.runner import load_run_result, run_experiment_batch
from src.statistics import (
    calculate_cliffs_delta,
    holm_bonferroni_correction,
    paired_wilcoxon_test,
    summarize_metric,
    calculate_paired_rank_biserial,
    one_sample_wilcoxon_test,
)


def test_batch_runner_execution_and_persistence(tmp_path: Path):
    """
    Test run_experiment_batch:
    - Miniature problem (n=10, m=3, evals=100).
    - 2 seeds for "nsga2_std" and "weighted_greedy".
    - Verify .npz files on disk and check non_dominated_F, non_dominated_X, wall_clock_time_sec keys.
    """
    generator = InstanceGenerator()
    instance = generator.generate(n_tasks=10, m_machines=3, alpha=2.0, seed=42)

    seeds = [42, 100]
    variants = ["nsga2_std", "weighted_greedy"]

    generated_paths = run_experiment_batch(
        instance=instance,
        variants=variants,
        seeds=seeds,
        output_dir=tmp_path,
        n_evals=100,
        pop_size=20,
    )

    assert len(generated_paths) == 4, f"Expected 4 .npz files, got {len(generated_paths)}"

    for file_path in generated_paths:
        assert file_path.exists(), f"File {file_path} was not created"
        assert file_path.suffix == ".npz"

        # Load with load_run_result helper
        res = load_run_result(file_path)

        assert "non_dominated_F" in res
        assert "non_dominated_X" in res
        assert "wall_clock_time_sec" in res
        assert "eval_count" in res
        assert "metadata" in res

        F = res["non_dominated_F"]
        X = res["non_dominated_X"]
        runtime = res["wall_clock_time_sec"]

        assert F.ndim == 2 and F.shape[1] == 2
        assert X.ndim == 2 and X.shape[1] == 10
        assert isinstance(runtime, float) and runtime >= 0.0
        assert res["eval_count"] > 0
        assert res["metadata"]["instance_id"] == instance.instance_id

        # Also load directly via np.load to ensure raw compatibility
        with np.load(file_path, allow_pickle=True) as raw:
            assert "non_dominated_F" in raw
            assert "non_dominated_X" in raw
            assert "wall_clock_time_sec" in raw


def test_batch_runner_random_and_em_variants(tmp_path: Path):
    """Verify random sampling and nsga2_em variants execute and save valid archives."""
    generator = InstanceGenerator()
    instance = generator.generate(n_tasks=8, m_machines=2, alpha=1.0, seed=99)

    generated_paths = run_experiment_batch(
        instance=instance,
        variants=["random", "nsga2_em"],
        seeds=[123],
        output_dir=tmp_path,
        n_evals=50,
        pop_size=10,
    )

    assert len(generated_paths) == 2
    for path in generated_paths:
        data = load_run_result(path)
        assert len(data["non_dominated_F"]) > 0
        assert len(data["non_dominated_X"]) > 0


def test_paired_wilcoxon_test():
    """Verify paired Wilcoxon test: identical arrays return p=1.0, distinct return significant p."""
    # 1. Identical arrays -> p = 1.0
    a = np.array([10.5, 12.3, 14.1, 9.8, 11.2, 13.0, 15.4, 8.9])
    stat_tie, p_tie = paired_wilcoxon_test(a, a)
    assert p_tie == 1.0
    assert stat_tie == 0.0

    # 2. Shifted arrays -> significant p-value
    b = a + 5.0
    stat_diff, p_diff = paired_wilcoxon_test(a, b)
    assert p_diff < 0.05

    # 3. Validation error on length mismatch
    with pytest.raises(ValueError, match="mismatch"):
        paired_wilcoxon_test([1.0, 2.0], [1.0])

def test_one_sample_wilcoxon_test():
    """Verify comparison against one fixed deterministic baseline."""
    # All stochastic runs exceed the fixed baseline.
    values = np.array([12.0, 13.0, 14.0, 15.0, 16.0, 17.0])
    statistic, p_value = one_sample_wilcoxon_test(values, reference=10.0)

    assert statistic == 0.0
    assert p_value < 0.05

    # All runs equal the fixed baseline.
    statistic, p_value = one_sample_wilcoxon_test(
        [10.0, 10.0, 10.0],
        reference=10.0,
    )
    assert statistic == 0.0
    assert p_value == 1.0

    # Invalid inputs.
    with pytest.raises(ValueError, match="must be non-empty"):
        one_sample_wilcoxon_test([], reference=10.0)

    with pytest.raises(ValueError, match="finite"):
        one_sample_wilcoxon_test([10.0, np.nan], reference=10.0)


def test_calculate_paired_rank_biserial():
    """Verify direction, magnitude, and ties for paired effect size."""
    a = np.array([10.0, 20.0, 30.0])
    b = np.array([1.0, 2.0, 3.0])

    # Every A observation exceeds its matching B observation.
    assert calculate_paired_rank_biserial(a, b) == 1.0

    # Reverse direction.
    assert calculate_paired_rank_biserial(b, a) == -1.0

    # Identical paired observations.
    assert calculate_paired_rank_biserial(a, a) == 0.0

    # Mixed positive and negative differences.
    effect = calculate_paired_rank_biserial(
        [10.0, 5.0, 8.0],
        [5.0, 10.0, 8.0],
    )
    assert -1.0 <= effect <= 1.0

    # Length mismatch.
    with pytest.raises(ValueError, match="mismatch"):
        calculate_paired_rank_biserial([1.0, 2.0], [1.0])


def test_holm_bonferroni_correction():
    """Verify Holm-Bonferroni correction rejects strictly according to step-down order."""
    # Example: m=3, alpha=0.05
    # p = [0.01, 0.04, 0.03]
    # Sorted: p_(1)=0.01 <= 0.05/3=0.01667 -> Reject (True)
    #         p_(2)=0.03 >  0.05/2=0.025   -> Fail to reject (False)
    #         p_(3)=0.04 -> Step-down stops -> (False)
    raw_p = [0.01, 0.04, 0.03]
    results = holm_bonferroni_correction(raw_p, alpha=0.05)

    assert len(results) == 3
    # Check original order preservation:
    # index 0 (0.01): True
    # index 1 (0.04): False
    # index 2 (0.03): False
    assert results[0][1] is True
    assert results[1][1] is False
    assert results[2][1] is False

    # Check step-down stop: even if a subsequent p < alpha, if rank 1 fails, all fail
    # p = [0.03, 0.04], alpha=0.05 -> p_(1)=0.03 > 0.05/2=0.025 -> both False
    stop_results = holm_bonferroni_correction([0.03, 0.04], alpha=0.05)
    assert all(not is_sig for _, is_sig in stop_results)

    # Empty list edge case
    assert holm_bonferroni_correction([]) == []


def test_calculate_cliffs_delta():
    """Verify Cliff's delta returns values bounded in [-1.0, 1.0] and handles extremes."""
    # 1. Bounded in [-1.0, 1.0]
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 100)
    y = rng.normal(0.5, 1, 100)
    d = calculate_cliffs_delta(x, y)
    assert -1.0 <= d <= 1.0

    # 2. Perfect positive dominance -> 1.0
    assert calculate_cliffs_delta([10, 20, 30], [1, 2, 3]) == 1.0

    # 3. Perfect negative dominance -> -1.0
    assert calculate_cliffs_delta([1, 2, 3], [10, 20, 30]) == -1.0

    # 4. Completely identical distributions -> 0.0
    assert calculate_cliffs_delta([5, 5, 5], [5, 5, 5]) == 0.0

    # Validation
    with pytest.raises(ValueError, match="non-empty"):
        calculate_cliffs_delta([], [1.0])


def test_summarize_metric():
    """Verify metric summarizer returns median and IQR rather than mean/std."""
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
    summary = summarize_metric(data)

    assert summary["median"] == 5.0
    assert summary["q25"] == 3.0
    assert summary["q75"] == 7.0
    assert summary["iqr"] == 4.0
    assert summary["min"] == 1.0
    assert summary["max"] == 9.0
    assert summary["n"] == 9

    with pytest.raises(ValueError, match="empty"):
        summarize_metric([])
