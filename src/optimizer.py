"""
NSGA-II problem definition and execution engine for energy-aware task scheduling.
"""

import time
from typing import List, Optional, Tuple, Union
import numpy as np

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import ElementwiseProblem
from pymoo.operators.crossover.pntx import PointCrossover
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from pymoo.core.sampling import Sampling

from src.instance_generator import ProblemInstance
from src.operators import (
    UniformIntegerMutation,
    BottleneckAwareMutation
)


class OptimizationResult:
    """
    Container for NSGA-II optimization results.
    Unpackable as `F, X, runtime` or accessible via attributes and dictionary keys.
    """

    def __init__(
        self,
        F: np.ndarray,
        X: np.ndarray,
        runtime: float,
        raw_result=None,
         eval_count: int = 0,
    ):
        self.F = F
        self.X = X
        self.runtime = runtime
        self.raw_result = raw_result
        self.eval_count = int(eval_count)

    def __iter__(self):
        yield self.F
        yield self.X
        yield self.runtime

    def __getitem__(self, item):
        if isinstance(item, str):
            if item in ("F", "objectives", "front"):
                return self.F
            elif item in ("X", "schedules", "solutions"):
                return self.X
            elif item in ("runtime", "wall_clock", "time"):
                return self.runtime
            elif item == "raw_result":
                return self.raw_result
            elif item in ("eval_count", "n_eval", "evaluations"):
                return self.eval_count
            raise KeyError(item)
        if item == 0:
            return self.F
        elif item == 1:
            return self.X
        elif item == 2:
            return self.runtime
        raise IndexError(f"Index {item} out of range for OptimizationResult")

    def __len__(self):
        return 3

    def __repr__(self) -> str:
        return f"OptimizationResult(n_solutions={len(self.F)}, runtime={self.runtime:.3f}s)"


class SchedulingProblem(ElementwiseProblem):
    """
    Elementwise problem definition for multi-objective energy-aware scheduling (Deliverable B5).

    Variables:
        Integer chromosome X in [0, m-1]^n representing task-to-machine assignment.

    Objectives:
        f1: Makespan max_j C_j(X)
        f2: Active Energy sum_j P_j * C_j(X)

    Hard assertions:
        f1 >= LB_C - 1e-9
        f2 >= LB_E - 1e-9
    """

    def __init__(
        self,
        w: Union[List[float], np.ndarray, ProblemInstance],
        s: Optional[Union[List[float], np.ndarray]] = None,
        p: Optional[Union[List[float], np.ndarray]] = None,
        **kwargs,
    ):
        if isinstance(w, ProblemInstance):
            inst = w
            w_arr = np.asarray(inst.w, dtype=np.float64)
            s_arr = np.asarray(inst.s, dtype=np.float64)
            p_arr = np.asarray(inst.p, dtype=np.float64)
        else:
            w_arr = np.asarray(w, dtype=np.float64)
            s_arr = np.asarray(s, dtype=np.float64)
            p_arr = np.asarray(p, dtype=np.float64)

        self.w = w_arr
        self.s = s_arr
        self.p = p_arr
        self.n = len(self.w)
        self.m = len(self.s)

        if self.n == 0 or self.m == 0:
            raise ValueError("Tasks and machines must be non-empty.")
        if len(self.p) != self.m:
            raise ValueError(f"Mismatch: {self.m} speeds but {len(self.p)} powers.")

        # Analytical lower bounds
        total_w = float(np.sum(self.w))
        total_s = float(np.sum(self.s))
        self.e = self.p / self.s
        self.lb_makespan = float(max(total_w / total_s, np.max(self.w) / np.max(self.s)))
        self.lb_energy = float(total_w * np.min(self.e))

        super().__init__(
            n_var=self.n,
            n_obj=2,
            n_ieq_constr=0,
            xl=0,
            xu=self.m - 1,
            vtype=int,
            **kwargs,
        )

    def _evaluate(self, x, out, *args, **kwargs):
        """
        Evaluate candidate chromosome x.
        """
        x_int = np.clip(np.round(x).astype(np.int64), 0, self.m - 1)

        # Calculate machine completion times: times = bincount(x, weights=w) / s
        times = np.bincount(x_int, weights=self.w, minlength=self.m) / self.s

        # Objective 1: Makespan
        f1 = float(np.max(times))

        # Objective 2: Active Energy
        f2 = float(np.sum(self.p * times))

        # Enforce hard runtime assertion checks against lower bounds
        assert f1 >= self.lb_makespan - 1e-9, f"Makespan {f1} violates lower bound {self.lb_makespan}"
        assert f2 >= self.lb_energy - 1e-9, f"Energy {f2} violates lower bound {self.lb_energy}"

        out["F"] = [f1, f2]


class EndpointSeededSampling(Sampling):
    """Insert predefined endpoint schedules into an NSGA-II population."""

    def __init__(self, initial_solutions):
        super().__init__()
        self.initial_solutions = np.asarray(
            initial_solutions, dtype=np.int64
        )

        if self.initial_solutions.ndim != 2:
            raise ValueError("initial_solutions must be a 2D array.")

    def _do(self, problem, n_samples, **kwargs):
        if self.initial_solutions.shape[1] != problem.n_var:
            raise ValueError(
                "Initial solutions do not match the problem chromosome length."
            )

        seeds = self.initial_solutions

        if np.any(seeds < problem.xl) or np.any(seeds > problem.xu):
            raise ValueError("Initial solutions contain invalid machine indices.")

        # Keep only as many endpoint seeds as the population can hold.
        seeds = seeds[:n_samples].copy()
        remaining = n_samples - len(seeds)

        if remaining == 0:
            return seeds

        # Fill the remaining slots using pymoo's standard integer sampler.
        random_state = kwargs.get("random_state")
        random_samples = IntegerRandomSampling()._do(
            problem, remaining, random_state=random_state
        )

        return np.vstack((seeds, random_samples))


def run_nsga2(
    problem: SchedulingProblem,
    pop_size: int = 100,
    n_evals: int = 20000,
    mutation=None,
    seed: int = 42,
    crossover=None,
    initial_population: Optional[np.ndarray] = None,
) -> OptimizationResult:
    """
    Execute standard NSGA-II optimization engine on SchedulingProblem (Deliverable B5).

    Args:
        problem: SchedulingProblem instance.
        pop_size: Population size (default: 100).
        n_evals: Stopping criterion budget in function evaluations (default: 20000, B(n) = 200n).
        mutation: Optional custom pymoo mutation operator. Defaults to
            UniformIntegerMutation when None. Can be used with BM, EM, or
            other compatible mutation operators.
        seed: Random seed for reproducibility (default: 42).
        crossover: Crossover operator (defaults to PointCrossover(n_points=1)).

    Returns:
        OptimizationResult containing non-dominated objective vectors (res.F),
        schedules (res.X), and wall-clock runtime in seconds.
    """
    if crossover is None:
        crossover = PointCrossover(n_points=1)

    if mutation is None:
        mutation = UniformIntegerMutation(prob=1.0, prob_var=1.0 / problem.n)

    algorithm = NSGA2(
        pop_size=pop_size,
        sampling=(
            IntegerRandomSampling()
            if initial_population is None
            else EndpointSeededSampling(initial_population)
        ),
        crossover=crossover,
        mutation=mutation,
        eliminate_duplicates=True,
    )

    termination = get_termination("n_evals", n_evals)

    start_time = time.perf_counter()
    raw_res = minimize(
        problem,
        algorithm,
        termination,
        seed=seed,
        verbose=False,
    )
    runtime = time.perf_counter() - start_time

    F = np.asarray(raw_res.F, dtype=np.float64)
    if F.ndim == 1 and len(F) > 0:
        F = F[np.newaxis, :]
    elif len(F) == 0:
        F = np.empty((0, 2), dtype=np.float64)

    X = np.clip(np.round(raw_res.X).astype(np.int64), 0, problem.m - 1)
    if X.ndim == 1 and len(X) > 0:
        X = X[np.newaxis, :]
    elif len(X) == 0:
        X = np.empty((0, problem.n), dtype=np.int64)

    actual_eval_count = int(raw_res.algorithm.evaluator.n_eval)

    return OptimizationResult(
        F=F,
        X=X,
        runtime=runtime,
        raw_result=raw_res,
        eval_count=actual_eval_count,
    )
