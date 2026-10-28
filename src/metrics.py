
"""Shared normalization, empirical reference fronts, Hypervolume, and IGD."""

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from pymoo.indicators.hv import HV
from pymoo.indicators.igd import IGD


@dataclass(frozen=True)
class ObjectiveNormalization:
    """Fixed objective-space bounds shared by all algorithms in a comparison."""

    ideal: np.ndarray
    nadir: np.ndarray

    def __post_init__(self):
        ideal = np.asarray(self.ideal, dtype=np.float64).reshape(-1).copy()
        nadir = np.asarray(self.nadir, dtype=np.float64).reshape(-1).copy()

        if ideal.size == 0 or ideal.shape != nadir.shape:
            raise ValueError("ideal and nadir must be non-empty vectors of equal shape.")
        if not np.all(np.isfinite(ideal)) or not np.all(np.isfinite(nadir)):
            raise ValueError("Normalization bounds must be finite.")
        if np.any(nadir <= ideal):
            raise ValueError("Each nadir bound must be greater than its ideal bound.")

        ideal.setflags(write=False)
        nadir.setflags(write=False)
        object.__setattr__(self, "ideal", ideal)
        object.__setattr__(self, "nadir", nadir)

    @classmethod
    def from_objective_sets(
        cls,
        objective_sets: Iterable[np.ndarray],
        padding: float = 0.05,
    ):
        """Fit fixed bounds to the pooled objective vectors for one benchmark."""
        if not 0.0 <= padding:
            raise ValueError("padding must be non-negative.")

        arrays = []
        for values in objective_sets:
            F = _validate_front(values, allow_empty=True)
            if len(F):
                arrays.append(F)

        if not arrays:
            raise ValueError("At least one non-empty objective set is required.")

        pooled = np.vstack(arrays)
        ideal = np.min(pooled, axis=0)
        maximum = np.max(pooled, axis=0)
        span = maximum - ideal

        # A constant objective still needs a nonzero scale.
        scale = np.where(span > 0.0, span, np.maximum(np.abs(ideal), 1.0))
        ideal = ideal - padding * scale
        nadir = maximum + padding * scale

        return cls(ideal=ideal, nadir=nadir)

    def transform(self, F: np.ndarray) -> np.ndarray:
        """Normalize objectives using this fixed instance-level scale."""
        F = _validate_front(F, allow_empty=True)
        if F.shape[1] != len(self.ideal):
            raise ValueError("Objective dimension does not match normalization bounds.")
        return (F - self.ideal) / (self.nadir - self.ideal)


def _validate_front(F: np.ndarray, allow_empty: bool = False) -> np.ndarray:
    F = np.asarray(F, dtype=np.float64)
    if F.ndim != 2:
        raise ValueError("Objective front must be a 2D array.")
    if F.shape[1] == 0:
        raise ValueError("Objective front must have at least one objective.")
    if len(F) == 0 and not allow_empty:
        raise ValueError("Objective front must not be empty.")
    if not np.all(np.isfinite(F)):
        raise ValueError("Objective values must all be finite.")
    return F


def nondominated_unique(F: np.ndarray) -> np.ndarray:
    """Return unique nondominated objective vectors for minimization."""
    F = _validate_front(F)
    F = np.unique(F, axis=0)

    keep = np.ones(len(F), dtype=bool)
    for i, point in enumerate(F):
        # Another point dominates this point if it is no worse in all
        # objectives and strictly better in at least one.
        dominates_i = np.all(F <= point, axis=1) & np.any(F < point, axis=1)
        if np.any(dominates_i):
            keep[i] = False

    return F[keep]


def build_empirical_reference_front(
    objective_sets: Iterable[np.ndarray],
) -> np.ndarray:
    """Build one pooled empirical nondominated front for a benchmark instance."""
    arrays = []
    for values in objective_sets:
        F = _validate_front(values, allow_empty=True)
        if len(F):
            arrays.append(F)

    if not arrays:
        raise ValueError("Cannot build a reference front from only empty sets.")

    return nondominated_unique(np.vstack(arrays))


@dataclass
class MultiObjectiveMetrics:
    """Fixed, shared metric configuration for one benchmark instance."""

    normalization: ObjectiveNormalization
    reference_front: np.ndarray
    hv_reference_point: np.ndarray

    @classmethod
    def fit(
        cls,
        objective_sets: Iterable[np.ndarray],
        padding: float = 0.05,
    ):
        """Fit once on pooled runs; reuse for every algorithm and seed."""
        sets = [
            _validate_front(values, allow_empty=True)
            for values in objective_sets
        ]
        nonempty = [values for values in sets if len(values)]
        if not nonempty:
            raise ValueError("At least one non-empty objective set is required.")

        normalization = ObjectiveNormalization.from_objective_sets(
            nonempty, padding=padding
        )
        reference_front = build_empirical_reference_front(nonempty)
        normalized_reference = normalization.transform(reference_front)

        # All objective values are normalized against shared bounds. A
        # reference point beyond the normalized empirical range avoids
        # accidental clipping of the best observed points.
        hv_reference_point = np.ones(reference_front.shape[1]) * (
            1.0 + padding
        )

        return cls(
            normalization=normalization,
            reference_front=normalized_reference,
            hv_reference_point=hv_reference_point,
        )

    def evaluate(self, F: np.ndarray) -> dict:
        """Calculate normalized HV and IGD using the shared configuration."""
        F = _validate_front(F, allow_empty=True)

        if F.shape[1] != len(self.normalization.ideal):
            raise ValueError("Objective dimension does not match fitted metrics.")

        if len(F) == 0:
            return {"hypervolume": 0.0, "igd": float("inf")}

        normalized = self.normalization.transform(F)
        normalized = nondominated_unique(normalized)

        hv = HV(ref_point=self.hv_reference_point)
        igd = IGD(self.reference_front)

        return {
            "hypervolume": float(hv(normalized)),
            "igd": float(igd(normalized)),
        }
