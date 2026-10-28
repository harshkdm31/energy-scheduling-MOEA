"""
Problem-aware genetic operators for multi-objective energy-aware scheduling.
"""

from typing import List, Optional, Union
import numpy as np
from pymoo.core.mutation import Mutation
from pymoo.core.variable import get


class EnergyAwareMutation(Mutation):
    """
    Energy-Aware Mutation (EM) operator (Deliverable B6).

    Reassigns a selected task to a machine with strictly lower specific energy
    consumption e_j = P_j / s_j. If the task is already on the most energy-efficient
    machine, it falls back to a uniform random reset across all m machines.
    """

    def __init__(
        self,
        w: Union[List[float], np.ndarray],
        s: Union[List[float], np.ndarray],
        p: Union[List[float], np.ndarray],
        prob: Optional[float] = None,
        **kwargs,
    ):
        self.w = np.asarray(w, dtype=np.float64)
        self.s = np.asarray(s, dtype=np.float64)
        self.p = np.asarray(p, dtype=np.float64)
        self.n = len(self.w)
        self.m = len(self.s)

        if self.n == 0 or self.m == 0:
            raise ValueError("Task workload and machine speed arrays must not be empty.")
        if len(self.p) != self.m:
            raise ValueError(f"Mismatch: {self.m} speeds but {len(self.p)} powers.")

        # Precompute specific energy consumption per unit of workload: e_j = P_j / s_j
        self.e = self.p / self.s

        prob_val = float(prob) if prob is not None else 1.0 / self.n
        self.prob_val = prob_val
        super().__init__(prob=1.0, prob_var=prob_val, **kwargs)

    def mutate_chromosome(
        self,
        x: np.ndarray,
        task_idx: Optional[int] = None,
        rng=None,
    ) -> np.ndarray:
        """
        Mutate a single chromosome x in [0, m-1]^n.

        If task_idx is provided, mutates that specific task.
        Otherwise, uniformly chooses a task i in {0, ..., n-1}.
        """
        rng = rng if rng is not None else np.random.default_rng()
        x_mut = np.copy(x).astype(np.int64)

        if task_idx is not None:
            i = int(task_idx)
        else:
            i = int(rng.integers(0, self.n)) if hasattr(rng, "integers") else int(rng.randint(0, self.n))

        curr_machine = x_mut[i]
        curr_e = self.e[curr_machine]

        # Target machines with strictly lower energy per work unit: e_j < e_{x_i}
        better_machines = np.where(self.e < curr_e)[0]

        if len(better_machines) > 0:
            new_machine = int(rng.choice(better_machines))
        else:
            # Fallback: uniform random reset across all m machines
            new_machine = int(rng.integers(0, self.m)) if hasattr(rng, "integers") else int(rng.randint(0, self.m))

        x_mut[i] = new_machine
        return x_mut

    def _do(self, problem, X, random_state=None, **kwargs):
        """
        Mutate a population matrix X of shape (n_individuals, n_tasks).
        """
        rng = random_state if random_state is not None else np.random.default_rng()
        X_out = np.copy(X).astype(np.int64)

        is_1d = X_out.ndim == 1
        if is_1d:
            X_out = X_out[np.newaxis, :]

        n_ind, n_var = X_out.shape

        for k in range(n_ind):
            # Select task i
            i = int(rng.integers(0, n_var)) if hasattr(rng, "integers") else int(rng.randint(0, n_var))
            curr_machine = X_out[k, i]
            curr_e = self.e[curr_machine]

            better_machines = np.where(self.e < curr_e)[0]
            if len(better_machines) > 0:
                new_machine = int(rng.choice(better_machines))
            else:
                # Fallback: uniform random reset across all m machines
                new_machine = int(rng.integers(0, self.m)) if hasattr(rng, "integers") else int(rng.randint(0, self.m))

            X_out[k, i] = new_machine

        return X_out[0] if is_1d else X_out


class UniformIntegerMutation(Mutation):
    """
    Standard uniform random reset baseline mutation for integer chromosomes.
    """

    def __init__(self, prob: float = 1.0, prob_var: Optional[float] = None, **kwargs):
        super().__init__(prob=prob, prob_var=prob_var, **kwargs)

    def _do(self, problem, X, random_state=None, **kwargs):
        rng = random_state if random_state is not None else np.random.default_rng()
        X_out = np.copy(X).astype(np.int64)

        is_1d = X_out.ndim == 1
        if is_1d:
            X_out = X_out[np.newaxis, :]

        n_ind, n_var = X_out.shape
        if self.prob_var is not None:
            prob_var = float(get(self.prob_var))
        else:
            prob_var = 1.0 / n_var if n_var > 0 else 0.1

        m = int(problem.xu[0] + 1) if (problem is not None and hasattr(problem, "xu")) else int(np.max(X_out) + 1)

        for k in range(n_ind):
            for i in range(n_var):
                rand_val = rng.random() if hasattr(rng, "random") else rng.rand()
                if rand_val < prob_var:
                    X_out[k, i] = int(rng.integers(0, m)) if hasattr(rng, "integers") else int(rng.randint(0, m))

        return X_out[0] if is_1d else X_out


class BottleneckAwareMutation(Mutation):
    """
    Bottleneck-Aware Mutation (BM).

    Attempts to move one task from a critical machine to another machine.
    Strict makespan improvements are preferred. If none exist, a move that
    does not worsen global makespan may be accepted.

    If no suitable move exists, the chromosome remains unchanged.
    """

    def __init__(
        self,
        w,
        s,
        prob=1.0,
        tol=1e-12,
        **kwargs,
    ):
        self.w = np.asarray(w, dtype=np.float64)
        self.s = np.asarray(s, dtype=np.float64)
        self.n = len(self.w)
        self.m = len(self.s)
        self.tol = float(tol)

        if self.n == 0 or self.m == 0:
            raise ValueError("Workloads and machine speeds must be non-empty.")
        if not np.all(np.isfinite(self.w)) or np.any(self.w <= 0):
            raise ValueError("Workloads must be finite and positive.")
        if not np.all(np.isfinite(self.s)) or np.any(self.s <= 0):
            raise ValueError("Machine speeds must be finite and positive.")
        if self.tol < 0:
            raise ValueError("tol must be non-negative.")

        super().__init__(prob=prob, **kwargs)

    @staticmethod
    def _randint(rng, high):
        """Support NumPy Generator and legacy RandomState APIs."""
        if hasattr(rng, "integers"):
            return int(rng.integers(0, high))
        return int(rng.randint(0, high))

    def _makespan(self, x):
        """Calculate the makespan of a chromosome."""
        loads = np.bincount(
            x, weights=self.w, minlength=self.m
        ) / self.s
        return float(np.max(loads))

    def mutate_chromosome(self, x, rng=None):
        """Mutate one chromosome and return a new chromosome."""
        rng = rng if rng is not None else np.random.default_rng()
        x = np.asarray(x, dtype=np.int64)

        if x.shape != (self.n,):
            raise ValueError(
                f"Expected chromosome shape {(self.n,)}, got {x.shape}."
            )
        if np.any(x < 0) or np.any(x >= self.m):
            raise ValueError("Chromosome contains an invalid machine index.")

        original = x.copy()
        old_makespan = self._makespan(original)

        loads = np.bincount(
            original, weights=self.w, minlength=self.m
        ) / self.s

        # Consider every machine tied for the maximum completion time.
        critical_machines = np.flatnonzero(
            np.isclose(
                loads, old_makespan, rtol=0.0, atol=self.tol
            )
        )

        improving = []
        non_worsening = []

        for source in critical_machines:
            for task in np.flatnonzero(original == source):
                for destination in range(self.m):
                    if destination == source:
                        continue

                    candidate = original.copy()
                    candidate[task] = destination
                    new_makespan = self._makespan(candidate)

                    if new_makespan < old_makespan - self.tol:
                        improving.append(candidate)
                    elif new_makespan <= old_makespan + self.tol:
                        non_worsening.append(candidate)

        # Prefer strict improvements; otherwise allow non-worsening moves.
        choices = improving if improving else non_worsening

        if not choices:
            return original

        return choices[self._randint(rng, len(choices))]

    def _do(self, problem, X, random_state=None, **kwargs):
        """Apply BM to a population of chromosomes."""
        rng = (
            random_state
            if random_state is not None
            else np.random.default_rng()
        )

        X_out = np.asarray(X, dtype=np.int64).copy()
        was_1d = X_out.ndim == 1

        if was_1d:
            X_out = X_out.reshape(1, -1)

        if X_out.ndim != 2 or X_out.shape[1] != self.n:
            raise ValueError(
                f"Expected population shape (n_individuals, {self.n}), "
                f"got {X_out.shape}."
            )

        for k in range(len(X_out)):
            X_out[k] = self.mutate_chromosome(X_out[k], rng=rng)

        return X_out[0] if was_1d else X_out
