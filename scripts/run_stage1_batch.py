#!/usr/bin/env python3
"""
Production Stage 1 Batch Execution Harness (Deliverables B8 / Gate 3).

Orchestrates the frozen Stage 1 Core experimental evaluation matrix:
  - Problem size: n=200, m=8, B(n)=40,000 function evaluations, pop_size=100.
  - Exponents: alpha in [1.0, 2.0, 3.0].
  - Seeds: 30 reserved test seeds (201..230).
  - Variants: nsga2_std, nsga2_em, weighted_greedy.
  - Invariant enforcement: f1 >= LB_C - 1e-9, f2 >= LB_E - 1e-9.
  - CLI flag --smoke-test: executes on miniature dimensions (n=20, m=4, 2 seeds, 100 evals)
    for sub-second CI validation.
"""

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional

import numpy as np

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.instance_generator import InstanceGenerator, ProblemInstance
from src.runner import load_run_result, run_experiment_batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Execute Stage 1 Core experimental batch runs across algorithms and seeds."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="experiments/configs/stage1_core.json",
        help="Path to Stage 1 configuration JSON.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/raw_runs",
        help="Root directory for storing raw .npz experiment runs.",
    )
    parser.add_argument(
        "--instance-seed",
        type=int,
        default=42,
        help="Seed for deterministic problem instance generation.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Execute fast smoke test with miniature problem (n=20, m=4, 2 seeds, 100 evals).",
    )
    parser.add_argument(
        "--n-tasks",
        type=int,
        default=None,
        help="Override number of tasks n.",
    )
    parser.add_argument(
        "--m-machines",
        type=int,
        default=None,
        help="Override number of machines m.",
    )
    parser.add_argument(
        "--alphas",
        nargs="+",
        type=float,
        default=None,
        help="Override list of alpha values (e.g. 1.0 2.0 3.0).",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=None,
        help="Override list of seeds (e.g. 201 202).",
    )
    parser.add_argument(
        "--eval-budget",
        type=int,
        default=None,
        help="Override evaluation budget.",
    )
    parser.add_argument(
        "--pop-size",
        type=int,
        default=None,
        help="Override population size.",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        type=str,
        default=None,
        help="Override variants list (default: nsga2_std nsga2_em weighted_greedy).",
    )
    return parser.parse_args()


def load_config(config_path: Path) -> Dict[str, Any]:
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_stage1_batch(
    config_path: Optional[str] = "experiments/configs/stage1_core.json",
    output_dir: str = "data/raw_runs",
    instance_seed: int = 42,
    smoke_test: bool = False,
    n_tasks: Optional[int] = None,
    m_machines: Optional[int] = None,
    alphas: Optional[List[float]] = None,
    seeds: Optional[List[int]] = None,
    eval_budget: Optional[int] = None,
    pop_size: Optional[int] = None,
    variants: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Execute the Stage 1 batch runs across alpha values, algorithm variants, and seeds.
    """
    cfg_file = REPO_ROOT / (config_path or "experiments/configs/stage1_core.json")
    config = load_config(cfg_file) if cfg_file.exists() else {}

    # Target settings resolution
    if smoke_test:
        resolved_n = n_tasks if n_tasks is not None else 20
        resolved_m = m_machines if m_machines is not None else 4
        resolved_alphas = alphas if alphas is not None else [1.0, 2.0, 3.0]
        resolved_seeds = seeds if seeds is not None else [201, 202]
        resolved_budget = eval_budget if eval_budget is not None else 100
        resolved_pop_size = pop_size if pop_size is not None else 20
    else:
        resolved_n = n_tasks if n_tasks is not None else config.get("n_tasks", 200)
        resolved_m = m_machines if m_machines is not None else config.get("m_machines", 8)
        resolved_alphas = alphas if alphas is not None else config.get("alpha_values", [1.0, 2.0, 3.0])
        resolved_seeds = seeds if seeds is not None else config.get("final_test_seeds", list(range(201, 231)))
        resolved_budget = eval_budget if eval_budget is not None else config.get("eval_budget", 40000)
        resolved_pop_size = pop_size if pop_size is not None else config.get("pop_size", 100)

    # Production variants for comparison: nsga2_std, nsga2_em, weighted_greedy
    if variants is not None:
        resolved_variants = variants
    else:
        resolved_variants = ["nsga2_std", "nsga2_em", "weighted_greedy"]

    out_root = REPO_ROOT / output_dir
    out_root.mkdir(parents=True, exist_ok=True)
    inst_dir = REPO_ROOT / "data" / "instances"
    inst_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("STAGE 1 BATCH EXPERIMENTAL EXECUTION (Phase 5 / Deliverable B8)")
    print("=" * 80)
    print(f"Mode:             {'SMOKE TEST' if smoke_test else 'PRODUCTION RUN'}")
    print(f"Problem Scale:    n={resolved_n}, m={resolved_m}")
    print(f"Alpha Regimes:    {resolved_alphas}")
    print(f"Test Seeds ({len(resolved_seeds)}): {resolved_seeds[:5]}... (total {len(resolved_seeds)})")
    print(f"Variants:         {resolved_variants}")
    print(f"Eval Budget:      {resolved_budget} evals, Pop Size: {resolved_pop_size}")
    print(f"Output Directory: {out_root}")
    print("=" * 80)

    gen = InstanceGenerator()
    generated_instances: List[ProblemInstance] = []
    all_raw_files: List[Path] = []
    execution_stats: List[Dict[str, Any]] = []

    total_start_time = time.perf_counter()

    for alpha in resolved_alphas:
        print(f"\n>>> Processing Alpha Regime: alpha = {alpha} (Instance Seed: {instance_seed})")
        instance = gen.generate(
            n_tasks=resolved_n,
            m_machines=resolved_m,
            alpha=alpha,
            kappa=config.get("kappa", 1.0),
            seed=instance_seed,
        )
        instance.save(inst_dir)
        generated_instances.append(instance)

        print(f"    Instance ID:    {instance.instance_id}")
        print(f"    LB Makespan:    {instance.lower_bound_makespan:.6f}")
        print(f"    LB Energy:      {instance.lower_bound_energy:.6f}")
        print(f"    rho_s:          {instance.rho_s:.4f}, rho_e: {instance.rho_e:.4f}")

        # Execute batch for this instance
        batch_paths = run_experiment_batch(
            instance=instance,
            variants=resolved_variants,
            seeds=resolved_seeds,
            output_dir=out_root,
            n_evals=resolved_budget,
            pop_size=resolved_pop_size,
        )
        all_raw_files.extend(batch_paths)

        # Enforce lower bound invariants on all outputs
        tol = 1e-9
        for path in batch_paths:
            if not path.exists():
                raise FileNotFoundError(f"Missing output file: {path}")

            data = load_run_result(path)
            F = np.asarray(data["non_dominated_F"], dtype=np.float64)
            X = np.asarray(data["non_dominated_X"], dtype=np.int64)

            if len(F) == 0:
                raise ValueError(f"Run result {path.name} contains empty non-dominated front.")

            min_f1 = float(np.min(F[:, 0]))
            min_f2 = float(np.min(F[:, 1]))

            if min_f1 < instance.lower_bound_makespan - tol:
                raise AssertionError(
                    f"Invariant violation in {path.name}: min f1={min_f1} < LB_C={instance.lower_bound_makespan}"
                )
            if min_f2 < instance.lower_bound_energy - tol:
                raise AssertionError(
                    f"Invariant violation in {path.name}: min f2={min_f2} < LB_E={instance.lower_bound_energy}"
                )
            if np.any(X < 0) or np.any(X >= instance.m_machines):
                raise AssertionError(f"Invalid machine alleles in {path.name}: values outside [0, m-1].")

            execution_stats.append({
                "alpha": alpha,
                "instance_id": instance.instance_id,
                "file": path.name,
                "wall_clock": data["wall_clock_time_sec"],
                "eval_count": data["eval_count"],
                "front_size": F.shape[0],
                "min_f1": min_f1,
                "min_f2": min_f2,
            })

        print(f"    Completed {len(batch_paths)} runs for alpha={alpha}. Invariants verified.")

    total_duration = time.perf_counter() - total_start_time
    print("\n" + "=" * 80)
    print(f"Batch execution completed successfully in {total_duration:.2f}s!")
    print(f"Total raw .npz archives verified: {len(all_raw_files)}")
    print("=" * 80)

    return {
        "status": "PASS",
        "total_files": len(all_raw_files),
        "total_duration_sec": total_duration,
        "instances": [inst.instance_id for inst in generated_instances],
        "file_paths": [str(p) for p in all_raw_files],
    }


if __name__ == "__main__":
    cli_args = parse_args()
    run_stage1_batch(
        config_path=cli_args.config,
        output_dir=cli_args.output_dir,
        instance_seed=cli_args.instance_seed,
        smoke_test=cli_args.smoke_test,
        n_tasks=cli_args.n_tasks,
        m_machines=cli_args.m_machines,
        alphas=cli_args.alphas,
        seeds=cli_args.seeds,
        eval_budget=cli_args.eval_budget,
        pop_size=cli_args.pop_size,
        variants=cli_args.variants,
    )
