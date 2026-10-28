#!/usr/bin/env python3
"""
Pilot Benchmark Execution Pipeline (Deliverable B8 / Gate 3).

Loads the frozen Stage 1 Core experimental configuration, generates a pilot
instance (n=50, m=4, alpha=2.0, seed=101), executes one run per variant
(random, nsga2_std, nsga2_em, weighted_greedy), validates .npz persistence
in data/raw_runs/, and verifies the analytical lower bound invariants
(f1 >= LB_C, f2 >= LB_E).
"""

import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List

import numpy as np

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.instance_generator import InstanceGenerator, ProblemInstance
from src.runner import load_run_result, run_experiment_batch


def load_stage1_config(config_path: Path) -> Dict[str, Any]:
    """
    Load and validate Stage 1 Core configuration.
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    # Sanity checks on config fields
    required_keys = [
        "stage",
        "n_tasks",
        "m_machines",
        "alpha_values",
        "eval_budget",
        "pop_size",
        "development_seeds",
        "final_test_seeds",
        "variants",
    ]
    for key in required_keys:
        if key not in config:
            raise KeyError(f"Missing required key in config: '{key}'")

    return config


def run_pilot(
    config_file: str = "experiments/configs/stage1_core.json",
    pilot_n: int = 50,
    pilot_m: int = 4,
    pilot_alpha: float = 2.0,
    pilot_seed: int = 101,
    output_dir: str = "data/raw_runs",
) -> Dict[str, Any]:
    """
    Execute the pilot benchmark pipeline.
    """
    config_path = REPO_ROOT / config_file
    print("=" * 80)
    print("HYPERPARAMETER FREEZE & PILOT BENCHMARK PIPELINE (Gate 3)")
    print("=" * 80)
    print(f"Loading Stage 1 Core configuration from: {config_path}")
    config = load_stage1_config(config_path)

    print(f"Config loaded successfully: {config.get('description', 'Stage 1 Core')}")
    print(f"  Frozen pop_size:           {config['pop_size']}")
    print(f"  Frozen dev seeds:          {config['development_seeds']}")
    print(f"  Frozen final test seeds:   {len(config['final_test_seeds'])} seeds ({config['final_test_seeds'][0]}..{config['final_test_seeds'][-1]})")
    print(f"  Tracked variants:          {config['variants']}")

    # Validate pilot seed is in development seeds
    if pilot_seed not in config["development_seeds"]:
        raise ValueError(
            f"Pilot seed {pilot_seed} is not in frozen development seeds: {config['development_seeds']}"
        )

    # 1. Generate pilot instance (n=50, m=4, alpha=2.0, seed=101)
    print("\n" + "-" * 80)
    print(f"1. Generating pilot instance (n={pilot_n}, m={pilot_m}, alpha={pilot_alpha}, seed={pilot_seed})...")
    gen = InstanceGenerator()
    instance: ProblemInstance = gen.generate(
        n_tasks=pilot_n,
        m_machines=pilot_m,
        alpha=pilot_alpha,
        kappa=config.get("kappa", 1.0),
        seed=pilot_seed,
    )
    inst_save_dir = REPO_ROOT / "data" / "instances"
    saved_inst_path = instance.save(inst_save_dir)
    print(f"  Instance ID:               {instance.instance_id}")
    print(f"  Saved instance JSON to:    {saved_inst_path}")
    print(f"  Lower Bound Makespan (LB_C): {instance.lower_bound_makespan:.6f}")
    print(f"  Lower Bound Energy   (LB_E): {instance.lower_bound_energy:.6f}")
    print(f"  Speed Heterogeneity  (rho_s): {instance.rho_s:.4f}")
    print(f"  Power Heterogeneity  (rho_e): {instance.rho_e:.4f}")
    print(f"  Correlation r(s, e):         {instance.pearson_s_e:.4f}")

    # 2. Execute one run per variant
    variants: List[str] = config["variants"]
    pop_size: int = config["pop_size"]
    eval_budget: int = 200 * instance.n_tasks  # B(50) = 10,000 evaluations
    print("\n" + "-" * 80)
    print(f"2. Executing pilot runs across variants: {variants}")
    print(f"   Evaluation budget B(n) = {eval_budget}, Pop size = {pop_size}, Seed = {pilot_seed}")

    raw_runs_dir = REPO_ROOT / output_dir
    generated_paths = run_experiment_batch(
        instance=instance,
        variants=variants,
        seeds=[pilot_seed],
        output_dir=raw_runs_dir,
        n_evals=eval_budget,
        pop_size=pop_size,
    )

    # 3. Verify .npz persistence and lower bound invariants
    print("\n" + "-" * 80)
    print("3. Verifying output persistence and analytical lower bound invariants...")
    results_summary: List[Dict[str, Any]] = []

    for variant in variants:
        expected_file = raw_runs_dir / instance.instance_id / f"{variant}_seed{pilot_seed}.npz"
        if not expected_file.exists():
            raise FileNotFoundError(f"Expected output file not generated: {expected_file}")

        run_data = load_run_result(expected_file)
        F = np.asarray(run_data["non_dominated_F"], dtype=np.float64)
        X = np.asarray(run_data["non_dominated_X"], dtype=np.int64)
        wall_clock = run_data["wall_clock_time_sec"]
        eval_count = run_data["eval_count"]

        # Validate non-empty front
        if F.ndim != 2 or F.shape[0] == 0:
            raise ValueError(f"Variant '{variant}' produced empty non-dominated front.")

        # Invariant checks: f1 >= LB_C, f2 >= LB_E (allowing floating point epsilon)
        tol = 1e-9
        min_f1 = float(np.min(F[:, 0]))
        min_f2 = float(np.min(F[:, 1]))

        if min_f1 < instance.lower_bound_makespan - tol:
            raise AssertionError(
                f"Variant '{variant}' violated makespan lower bound: min f1={min_f1} < LB_C={instance.lower_bound_makespan}"
            )
        if min_f2 < instance.lower_bound_energy - tol:
            raise AssertionError(
                f"Variant '{variant}' violated energy lower bound: min f2={min_f2} < LB_E={instance.lower_bound_energy}"
            )

        # Validate schedule alleles
        if np.any(X < 0) or np.any(X >= instance.m_machines):
            raise AssertionError(f"Variant '{variant}' produced invalid machine assignments outside [0, m-1].")

        print(
            f"  [PASS] {variant:<16} | File: {expected_file.name:<25} | "
            f"Wall-Clock: {wall_clock:6.3f}s | Evals: {eval_count:5d} | "
            f"Front Size: {F.shape[0]:3d} | min f1: {min_f1:8.4f} (LB: {instance.lower_bound_makespan:.4f}) | "
            f"min f2: {min_f2:8.4f} (LB: {instance.lower_bound_energy:.4f})"
        )

        results_summary.append({
            "variant": variant,
            "seed": pilot_seed,
            "file": str(expected_file),
            "wall_clock": wall_clock,
            "eval_count": eval_count,
            "front_size": F.shape[0],
            "min_f1": min_f1,
            "min_f2": min_f2,
            "status": "PASS",
        })

    print("-" * 80)
    print("All pilot runs completed successfully and satisfied all invariant checks!")
    print("=" * 80)
    return {
        "instance_id": instance.instance_id,
        "results": results_summary,
        "lower_bounds": {
            "LB_C": instance.lower_bound_makespan,
            "LB_E": instance.lower_bound_energy,
        },
    }


if __name__ == "__main__":
    run_pilot()
