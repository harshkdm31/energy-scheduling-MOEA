
import numpy as np
import pytest

from src.metrics import (
    MultiObjectiveMetrics,
    ObjectiveNormalization,
    build_empirical_reference_front,
    nondominated_unique,
)


def test_normalization_is_shared_and_consistent():
    front_a = np.array([[2.0, 8.0], [4.0, 5.0]])
    front_b = np.array([[3.0, 6.0], [5.0, 3.0]])

    norm = ObjectiveNormalization.from_objective_sets([front_a, front_b])

    transformed_a = norm.transform(front_a)
    transformed_b = norm.transform(front_b)

    assert transformed_a.shape == front_a.shape
    assert transformed_b.shape == front_b.shape
    assert np.allclose(norm.ideal, [1.85, 2.75])
    assert np.allclose(norm.nadir, [5.15, 8.25])


def test_reference_front_is_pooled_unique_and_nondominated():
    front_a = np.array([[1.0, 5.0], [2.0, 4.0]])
    front_b = np.array([[2.0, 4.0], [3.0, 6.0]])

    reference = build_empirical_reference_front([front_a, front_b])

    assert len(reference) == 2
    assert len(np.unique(reference, axis=0)) == len(reference)
    assert any(np.allclose(row, [1.0, 5.0]) for row in reference)
    assert any(np.allclose(row, [2.0, 4.0]) for row in reference)


def test_dominated_points_are_removed():
    front = np.array([
        [1.0, 5.0],
        [2.0, 4.0],
        [3.0, 6.0],
        [2.0, 4.0],
    ])

    result = nondominated_unique(front)

    assert len(result) == 2
    assert not any(np.allclose(row, [3.0, 6.0]) for row in result)


def test_all_algorithms_use_same_metric_configuration():
    front_a = np.array([[2.0, 8.0], [4.0, 5.0]])
    front_b = np.array([[3.0, 6.0], [5.0, 3.0]])

    metrics = MultiObjectiveMetrics.fit([front_a, front_b])

    result_a = metrics.evaluate(front_a)
    result_b = metrics.evaluate(front_b)

    assert set(result_a) == {"hypervolume", "igd"}
    assert set(result_b) == {"hypervolume", "igd"}
    assert result_a["hypervolume"] >= 0.0
    assert result_b["hypervolume"] >= 0.0
    assert result_a["igd"] >= 0.0
    assert result_b["igd"] >= 0.0


def test_invalid_or_empty_reference_data_is_rejected():
    with pytest.raises(ValueError):
        MultiObjectiveMetrics.fit([np.empty((0, 2))])

    with pytest.raises(ValueError):
        ObjectiveNormalization.from_objective_sets([
            np.array([[1.0, np.nan]])
        ])


def test_empty_algorithm_front_has_defined_output():
    front = np.array([[1.0, 5.0], [2.0, 4.0]])
    metrics = MultiObjectiveMetrics.fit([front])

    result = metrics.evaluate(np.empty((0, 2)))

    assert result["hypervolume"] == 0.0
    assert np.isinf(result["igd"])
