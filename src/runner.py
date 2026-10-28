"""
Experiment runner and raw result persistence for multi-objective scheduling benchmarks.
"""

import json
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Union
import numpy as np

from src.baselines import evaluate_objectives, filter_pareto_front, solve_weighted_greedy_sweep
from src.instance_generator import ProblemInstance
from src.operators import (
    EnergyAwareMutation,
    UniformIntegerMutation,
    BottleneckAwareMutation
)
from src.optimizer import SchedulingProblem, run_nsga2
from src.seeding import build_endpoint_schedules

def save_run_result(
    filepath: Union[str, Path],
    non_dominated_F: np.ndarray,
    non_dominated_X: np.ndarray,
    wall_clock_time_sec: float,
    eval_count: int,
    metadata: Dict[str, Any],
) -> Path:
    """
    Save raw experiment run result to compressed .npz archive (Deliverable B8).
    """
    out_path = Path(filepath)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    F = np.asarray(non_dominated_F, dtype=np.float64)
    if F.ndim == 1 and len(F) > 0:
        F = F[np.newaxis, :]
    elif len(F) == 0:
        F = np.empty((0, 2), dtype=np.float64)

    X = np.asarray(non_dominated_X, dtype=np.int64)
    if X.ndim == 1 and len(X) > 0:
        X = X[np.newaxis, :]

    meta_str = json.dumps(metadata)

    np.savez_compressed(
        out_path,
        non_dominated_F=F,
        non_dominated_X=X,
        wall_clock_time_sec=float(wall_clock_time_sec),
        eval_count=int(eval_count),
        metadata=meta_str,
    )
    return out_path


def load_run_result(filepath: Union[str, Path]) -> Dict[str, Any]:
    """
    Load a raw experiment run result from a .npz archive.
    """
    path = Path(filepath)
    with np.load(path, allow_pickle=True) as data:
        meta_raw = data["metadata"]
        meta_str = str(meta_raw) if not isinstance(meta_raw, str) else meta_raw
        try:
            parsed_meta = json.loads(meta_str)
        except Exception:
            parsed_meta = {"raw": meta_str}

        return {
            "non_dominated_F": data["non_dominated_F"],
            "non_dominated_X": data["non_dominated_X"],
            "wall_clock_time_sec": float(data["wall_clock_time_sec"]),
            "eval_count": int(data["eval_count"]),
            "metadata": parsed_meta,
        }


def run_experiment_batch(
    instance: ProblemInstance,
    variants: List[str],
    seeds: List[int],
    output_dir: Union[str, Path] = "data/raw_runs",
    n_evals: Optional[int] = None,
    pop_size: int = 100,
) -> List[Path]:
    """
    Execute a batch of experiments across algorithm variants and random seeds (Deliverable B8).

    Supported variants:
        - "random": Uniform random sampling up to B(n) evaluations.
        - "nsga2_std": Standard NSGA-II with uniform random reset mutation.
        - "nsga2_em": NSGA-II integrated with EnergyAwareMutation.
        - "weighted_greedy": Deterministic weighted sweep.
        - "nsga2_bm": NSGA-II integrated with BottleneckAwareMutation.
        - "nsga2_seeded": Endpoint-seeded NSGA-II.
        - "nsga2_seeded_bm": Endpoint-seeded NSGA-II with BM.
        - "nsga2_seeded_em": Endpoint-seeded NSGA-II with EM.

    Enforces budget B(n) = 200 * n function evaluations unless overridden by n_evals.

    Outputs saved to:
        <output_dir>/<instance_id>/<variant>_seed<seed>.npz

    Returns:
        List[Path]: Paths to all written .npz result archives.
    """
    out_dir = Path(output_dir) / instance.instance_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # Budget calculation: exact budget B(n) = 200 * n evaluations
    eval_budget = n_evals if n_evals is not None else 200 * instance.n_tasks
    effective_pop = min(pop_size, eval_budget)

    generated_paths: List[Path] = []

    for variant in variants:
        if variant == "weighted_greedy":
            # Deterministic weighted sweep: run once per instance
            t0 = time.perf_counter()
            sweep_res = solve_weighted_greedy_sweep(
                w=instance.w,
                s=instance.s,
                p=instance.p,
                num_weights=21,
            )
            wall_clock = time.perf_counter() - t0
            eval_count = 21

            for seed in seeds:
                file_path = out_dir / f"{variant}_seed{seed}.npz"
                metadata = {
                    "variant": variant,
                    "instance_id": instance.instance_id,
                    "seed": int(seed),
                    "eval_count": eval_count,
                    "eval_count_type": "scalarization_candidates",
                    "requested_eval_budget": eval_budget,
                    "n_tasks": instance.n_tasks,
                    "m_machines": instance.m_machines,
                    "stochastic": False,
                    "deterministic_per_instance": True,
                    "statistical_unit": "instance",
                }
                save_run_result(
                    filepath=file_path,
                    non_dominated_F=sweep_res.objectives,
                    non_dominated_X=sweep_res.schedules,
                    wall_clock_time_sec=wall_clock,
                    eval_count=eval_count,
                    metadata=metadata,
                )
                generated_paths.append(file_path)

        elif variant == "random":
            for seed in seeds:
                rng = np.random.default_rng(seed)
                t0 = time.perf_counter()

                # Generate eval_budget random integer chromosomes
                X_all = rng.integers(0, instance.m_machines, size=(eval_budget, instance.n_tasks), dtype=np.int64)
                F_all = np.zeros((eval_budget, 2), dtype=np.float64)

                for idx in range(eval_budget):
                    F_all[idx] = evaluate_objectives(X_all[idx], instance.w, instance.s, instance.p)

                wall_clock = time.perf_counter() - t0

                # Filter non-dominated Pareto front
                pareto_X, pareto_F = filter_pareto_front(list(X_all), [tuple(f) for f in F_all])

                file_path = out_dir / f"{variant}_seed{seed}.npz"
                metadata = {
                    "variant": variant,
                    "instance_id": instance.instance_id,
                    "seed": int(seed),
                    "eval_count": eval_budget,
                    "eval_count_type": "objective_evaluations",
                    "requested_eval_budget": eval_budget,
                    "n_tasks": instance.n_tasks,
                    "m_machines": instance.m_machines,
                    "stochastic": True,
                    "deterministic_per_instance": False,
                    "statistical_unit": "instance_seed",
                }
                save_run_result(
                    filepath=file_path,
                    non_dominated_F=pareto_F,
                    non_dominated_X=pareto_X,
                    wall_clock_time_sec=wall_clock,
                    eval_count=eval_budget,
                    metadata=metadata,
                )
                generated_paths.append(file_path)

        elif variant == "nsga2_std":
            problem = SchedulingProblem(instance)
            mutation = UniformIntegerMutation(prob=1.0, prob_var=1.0 / instance.n_tasks)

            for seed in seeds:
                res = run_nsga2(
                    problem=problem,
                    pop_size=effective_pop,
                    n_evals=eval_budget,
                    mutation=mutation,
                    seed=seed,
                )
                file_path = out_dir / f"{variant}_seed{seed}.npz"
                metadata = {
                    "variant": variant,
                    "instance_id": instance.instance_id,
                    "seed": int(seed),
                    "eval_count": res.eval_count,
                    "requested_eval_budget": eval_budget,
                    "n_tasks": instance.n_tasks,
                    "m_machines": instance.m_machines,
                }
                save_run_result(
                    filepath=file_path,
                    non_dominated_F=res.F,
                    non_dominated_X=res.X,
                    wall_clock_time_sec=res.runtime,
                    eval_count=res.eval_count,
                    metadata=metadata,
                )
                generated_paths.append(file_path)

        elif variant == "nsga2_em":
            problem = SchedulingProblem(instance)
            mutation = EnergyAwareMutation(
                w=instance.w,
                s=instance.s,
                p=instance.p,
                prob=1.0 / instance.n_tasks,
            )

            for seed in seeds:
                res = run_nsga2(
                    problem=problem,
                    pop_size=effective_pop,
                    n_evals=eval_budget,
                    mutation=mutation,
                    seed=seed,
                )
                file_path = out_dir / f"{variant}_seed{seed}.npz"
                metadata = {
                    "variant": variant,
                    "instance_id": instance.instance_id,
                    "seed": int(seed),
                    "eval_count": res.eval_count,
                    "requested_eval_budget": eval_budget,
                    "n_tasks": instance.n_tasks,
                    "m_machines": instance.m_machines,
                }
                save_run_result(
                    filepath=file_path,
                    non_dominated_F=res.F,
                    non_dominated_X=res.X,
                    wall_clock_time_sec=res.runtime,
                    eval_count=res.eval_count,
                    metadata=metadata,
                )
                generated_paths.append(file_path)

        elif variant == "nsga2_bm":
            problem = SchedulingProblem(instance)
            mutation = BottleneckAwareMutation(
                w=instance.w,
                s=instance.s,
                prob=1.0,
            )

            for seed in seeds:
                res = run_nsga2(
                    problem=problem,
                    pop_size=effective_pop,
                    n_evals=eval_budget,
                    mutation=mutation,
                    seed=seed,
                )

                file_path = out_dir / f"{variant}_seed{seed}.npz"

                metadata = {
                    "variant": variant,
                    "instance_id": instance.instance_id,
                    "seed": int(seed),
                    "eval_count": res.eval_count,
                    "requested_eval_budget": eval_budget,
                    "n_tasks": instance.n_tasks,
                    "m_machines": instance.m_machines,
                }

                save_run_result(
                    filepath=file_path,
                    non_dominated_F=res.F,
                    non_dominated_X=res.X,
                    wall_clock_time_sec=res.runtime,
                    eval_count=res.eval_count,
                    metadata=metadata,
                )

                generated_paths.append(file_path)

        elif variant in (
            "nsga2_seeded",
            "nsga2_seeded_bm",
            "nsga2_seeded_em",
        ):
            problem = SchedulingProblem(instance)

            endpoint_seeds = build_endpoint_schedules(
                w=instance.w,
                s=instance.s,
                p=instance.p,
            )
            initial_population = np.asarray(
                endpoint_seeds, dtype=np.int64
            )

            for seed in seeds:
                if variant == "nsga2_seeded_bm":
                    mutation = BottleneckAwareMutation(
                        w=instance.w,
                        s=instance.s,
                        prob=1.0,
                    )
                elif variant == "nsga2_seeded_em":
                    mutation = EnergyAwareMutation(
                        w=instance.w,
                        s=instance.s,
                        p=instance.p,
                        prob=1.0 / instance.n_tasks,
                    )
                else:
                    mutation = UniformIntegerMutation(
                        prob=1.0,
                        prob_var=1.0 / instance.n_tasks,
                    )

                res = run_nsga2(
                    problem=problem,
                    pop_size=effective_pop,
                    n_evals=eval_budget,
                    mutation=mutation,
                    seed=seed,
                    initial_population=initial_population,
                )

                file_path = (
                    out_dir / f"{variant}_seed{seed}.npz"
                )
                metadata = {
                    "variant": variant,
                    "instance_id": instance.instance_id,
                    "seed": int(seed),
                    "eval_count": res.eval_count,
                    "requested_eval_budget": eval_budget,
                    "n_tasks": instance.n_tasks,
                    "m_machines": instance.m_machines,
                    "endpoint_seed_count": len(endpoint_seeds),
                    "endpoint_seeding": True,
                }

                save_run_result(
                    filepath=file_path,
                    non_dominated_F=res.F,
                    non_dominated_X=res.X,
                    wall_clock_time_sec=res.runtime,
                    eval_count=res.eval_count,
                    metadata=metadata,
                )
                generated_paths.append(file_path)

        else:
            raise ValueError(f"Unsupported algorithm variant: '{variant}'.")

    return generated_paths
