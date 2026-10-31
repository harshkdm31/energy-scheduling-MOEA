
"""
Integration tests for Stage 3 multi-instance execution,
archive validation, and instance-level resume.
"""

from pathlib import Path

from scripts.run_stage3_batch import run_stage3_batch
from src.runner import load_run_result


def test_stage3_smoke_execution_and_archive_integrity(tmp_path: Path):
    """Verify smoke execution generates all expected valid archives."""

    output_dir = tmp_path / "stage3_smoke"
    instances_dir = tmp_path / "instances"

    result = run_stage3_batch(
        output_dir=str(output_dir),
        instances_dir=str(instances_dir),
        smoke_test=True,
    )

    assert result["status"] == "PASS"
    assert result["total_instances"] == 1
    assert result["completed_instances"] == 1
    assert result["skipped_instances"] == 0
    assert result["total_files"] == 8

    archives = list(output_dir.rglob("*.npz"))
    assert len(archives) == 8

    expected_pairs = {
        (variant, seed)
        for variant in [
            "random",
            "nsga2_std",
            "nsga2_em",
            "weighted_greedy",
        ]
        for seed in [201, 202]
    }

    observed_pairs = set()

    for archive in archives:
        data = load_run_result(archive)
        metadata = data["metadata"]

        assert metadata["instance_id"] == "inst_n20_m4_a2_s42020"
        assert metadata["n_tasks"] == 20
        assert metadata["m_machines"] == 4
        assert metadata["variant"] in {
            "random",
            "nsga2_std",
            "nsga2_em",
            "weighted_greedy",
        }
        assert metadata["seed"] in [201, 202]
        assert metadata["requested_eval_budget"] == 100
        assert 0 < data["eval_count"] <= 100

        assert data["non_dominated_F"].ndim == 2
        assert data["non_dominated_F"].shape[1] == 2
        assert len(data["non_dominated_F"]) > 0

        assert data["non_dominated_X"].shape[0] == len(
            data["non_dominated_F"]
        )
        assert data["non_dominated_X"].shape[1] == 20

        observed_pairs.add(
            (metadata["variant"], metadata["seed"])
        )

    assert observed_pairs == expected_pairs

    instance_files = list(instances_dir.glob("*.json"))
    assert len(instance_files) == 1


def test_stage3_skips_completed_instance(tmp_path: Path):
    """Verify a fully validated instance is skipped on a second run."""

    output_dir = tmp_path / "stage3_resume"
    instances_dir = tmp_path / "instances"

    first = run_stage3_batch(
        output_dir=str(output_dir),
        instances_dir=str(instances_dir),
        smoke_test=True,
    )

    assert first["completed_instances"] == 1
    assert first["skipped_instances"] == 0

    second = run_stage3_batch(
        output_dir=str(output_dir),
        instances_dir=str(instances_dir),
        smoke_test=True,
    )

    assert second["status"] == "PASS"
    assert second["completed_instances"] == 0
    assert second["skipped_instances"] == 1
    assert second["rerun_instances"] == 0
    assert second["total_files"] == 8


def test_stage3_recovers_incomplete_instance(tmp_path: Path):
    """Verify an incomplete instance is regenerated in full."""

    output_dir = tmp_path / "stage3_recovery"
    instances_dir = tmp_path / "instances"

    run_stage3_batch(
        output_dir=str(output_dir),
        instances_dir=str(instances_dir),
        smoke_test=True,
    )

    instance_dir = (
        output_dir
        / "smoke"
        / "alpha_2"
        / "inst_n20_m4_a2_s42020"
    )

    missing_archive = instance_dir / "nsga2_std_seed201.npz"

    assert missing_archive.exists()

    missing_archive.unlink()

    assert not missing_archive.exists()

    recovered = run_stage3_batch(
        output_dir=str(output_dir),
        instances_dir=str(instances_dir),
        smoke_test=True,
    )

    assert recovered["status"] == "PASS"
    assert recovered["completed_instances"] == 1
    assert recovered["skipped_instances"] == 0
    assert recovered["rerun_instances"] == 1
    assert recovered["total_files"] == 8

    archives = list(instance_dir.glob("*.npz"))
    assert len(archives) == 8
    assert missing_archive.exists()
