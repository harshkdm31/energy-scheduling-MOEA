
import numpy as np
import pytest

from src.operators import BottleneckAwareMutation


def makespan(x, w, s):
    loads = np.bincount(x, weights=w, minlength=len(s)) / s
    return float(np.max(loads))


def test_bm_finds_strict_makespan_improvement():
    w = np.array([4.0, 4.0, 1.0])
    s = np.array([1.0, 1.0])
    x = np.array([0, 0, 1])

    bm = BottleneckAwareMutation(w, s)
    result = bm.mutate_chromosome(x, rng=np.random.default_rng(7))

    assert result.shape == x.shape
    assert np.all((result >= 0) & (result < len(s)))
    assert makespan(result, w, s) < makespan(x, w, s)


def test_bm_preserves_chromosome_with_one_machine():
    w = np.array([2.0, 3.0])
    s = np.array([1.0])
    x = np.array([0, 0])

    bm = BottleneckAwareMutation(w, s)
    result = bm.mutate_chromosome(x, rng=np.random.default_rng(1))

    np.testing.assert_array_equal(result, x)


def test_bm_does_not_worsen_makespan():
    w = np.array([5.0, 3.0, 2.0, 1.0])
    s = np.array([1.0, 2.0, 3.0])
    x = np.array([0, 0, 1, 2])

    bm = BottleneckAwareMutation(w, s)
    result = bm.mutate_chromosome(x, rng=np.random.default_rng(11))

    assert makespan(result, w, s) <= makespan(x, w, s) + 1e-12


def test_bm_rejects_invalid_machine_index():
    bm = BottleneckAwareMutation(
        w=np.array([2.0, 3.0]),
        s=np.array([1.0, 2.0]),
    )

    with pytest.raises(ValueError):
        bm.mutate_chromosome(np.array([0, 2]))


def test_bm_rejects_nonpositive_workloads():
    with pytest.raises(ValueError):
        BottleneckAwareMutation(
            w=np.array([2.0, 0.0]),
            s=np.array([1.0, 2.0]),
        )
