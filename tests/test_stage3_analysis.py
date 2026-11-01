from pathlib import Path

import numpy as np
import pytest

import scripts.analyze_stage3 as stage3


# ---------------------------------------------------------
# Hypervolume tests
# ---------------------------------------------------------

def test_hypervolume_single_point():
    front = np.array([[2.0, 3.0]])
    reference = (5.0, 6.0)

    assert stage3.compute_hypervolume_2d(front, reference) == pytest.approx(9.0)


def test_hypervolume_multiple_non_dominated_points():
    front = np.array([[1.0, 4.0], [2.0, 2.0]])
    reference = (5.0, 6.0)

    assert stage3.compute_hypervolume_2d(front, reference) == pytest.approx(14.0)


def test_hypervolume_ignores_dominated_points():
    front = np.array([[1.0, 4.0], [2.0, 5.0], [2.0, 2.0]])
    reference = (5.0, 6.0)

    assert stage3.compute_hypervolume_2d(front, reference) == pytest.approx(14.0)


def test_hypervolume_empty_front():
    front = np.empty((0, 2))
    reference = (5.0, 6.0)

    assert stage3.compute_hypervolume_2d(front, reference) == 0.0


# ---------------------------------------------------------
# Archive discovery tests
# ---------------------------------------------------------

def test_discover_archives_recursively(tmp_path):
    instance_id = "inst_small_a1_s1001"
    instance_dir = tmp_path / "small" / "alpha_1.0" / instance_id
    instance_dir.mkdir(parents=True)

    archive = instance_dir / "nsga2_em_seed201.npz"
    archive.touch()

    result = stage3.discover_archives(tmp_path)

    assert instance_id in result
    assert result[instance_id]["nsga2_em"][201] == archive


def test_discover_archives_ignores_unknown_variants(tmp_path):
    instance_dir = tmp_path / "small" / "alpha_1.0" / "instance1"
    instance_dir.mkdir(parents=True)

    (instance_dir / "unknown_algorithm_seed201.npz").touch()

    result = stage3.discover_archives(tmp_path)

    assert result == {}


def test_discover_archives_missing_directory(tmp_path):
    missing_dir = tmp_path / "does_not_exist"

    with pytest.raises(FileNotFoundError):
        stage3.discover_archives(missing_dir)


# ---------------------------------------------------------
# Archive validation tests
# ---------------------------------------------------------

def test_validate_instance_archives_complete():
    seeds = [201, 202]
    variants = {
        variant: {seed: Path(f"{variant}_{seed}.npz") for seed in seeds}
        for variant in stage3.EXPECTED_VARIANTS
    }

    stage3.validate_instance_archives("instance1", variants, seeds)


def test_validate_instance_archives_missing_variant():
    variants = {
        variant: {201: Path(f"{variant}.npz")}
        for variant in stage3.EXPECTED_VARIANTS
        if variant != "nsga2_em"
    }

    with pytest.raises(ValueError, match="missing variant"):
        stage3.validate_instance_archives("instance1", variants, [201])


def test_validate_instance_archives_missing_seed():
    variants = {
        variant: {201: Path(f"{variant}.npz")}
        for variant in stage3.EXPECTED_VARIANTS
    }

    variants["nsga2_em"] = {}

    with pytest.raises(ValueError, match="incomplete seeds"):
        stage3.validate_instance_archives("instance1", variants, [201])


# ---------------------------------------------------------
# Instance analysis tests
# ---------------------------------------------------------

def build_mock_instance(tmp_path):
    variants = {}

    for variant in stage3.EXPECTED_VARIANTS:
        variants[variant] = {}

        for seed in [201, 202, 203]:
            path = tmp_path / f"{variant}_seed{seed}.npz"
            path.touch()
            variants[variant][seed] = path

    return variants


def test_analyze_instance_uses_median_across_seeds(tmp_path, monkeypatch):
    variants = build_mock_instance(tmp_path)

    fronts = {}

    for seed, value in zip([201, 202, 203], [1.0, 2.0, 3.0]):
        fronts[variants["nsga2_std"][seed]] = np.array([[value, value]])

    for variant in ["random", "nsga2_em"]:
        for seed in [201, 202, 203]:
            fronts[variants[variant][seed]] = np.array([[2.0, 2.0]])

    for seed in [201, 202, 203]:
        fronts[variants["weighted_greedy"][seed]] = np.array([[3.0, 3.0]])

    monkeypatch.setattr(
        stage3,
        "load_run_result",
        lambda path: {"non_dominated_F": fronts[path]},
    )

    result = stage3.analyze_instance("instance1", variants)

    expected_median = np.median(
        list(result["run_hv"]["nsga2_std"].values())
    )

    assert result["instance_values"]["nsga2_std"] == pytest.approx(expected_median)


def test_analyze_instance_rejects_inconsistent_greedy(tmp_path, monkeypatch):
    variants = build_mock_instance(tmp_path)

    fronts = {}

    for variant in ["random", "nsga2_std", "nsga2_em"]:
        for seed in [201, 202, 203]:
            fronts[variants[variant][seed]] = np.array([[2.0, 2.0]])

    for seed, value in zip([201, 202, 203], [2.0, 3.0, 4.0]):
        fronts[variants["weighted_greedy"][seed]] = np.array([[value, value]])

    monkeypatch.setattr(
        stage3,
        "load_run_result",
        lambda path: {"non_dominated_F": fronts[path]},
    )

    with pytest.raises(ValueError, match="Weighted Greedy archives disagree"):
        stage3.analyze_instance("instance1", variants)