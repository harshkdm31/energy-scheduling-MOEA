
#!/usr/bin/env python3
"""
Stage 2 Scale-Generalization Batch Runner.

Runs the frozen Stage 2 experiment matrix for small and large
heterogeneous scheduling instances.
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.instance_generator import InstanceGenerator
from src.runner import load_run_result, run_experiment_batch


def parse_args():
    parser = argparse.ArgumentParser(
        description="Execute Stage 2 scale experiments."
    )
    parser.add_argument(
        "--config",
        default="experiments/configs/stage2_scale.json",
    )
    parser.add_argument(
        "--output-dir",
        default="data/raw_runs/stage2_scale",
    )
    parser.add_argument("--instance-seed", type=int, default=42)
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run a miniature validation batch instead of production."
    )
    return parser.parse_args()


def load_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def validate_result(path, instance, expected_budget):
    if not path.exists():
        raise FileNotFoundError(f"Missing archive: {path}")

    result = load_run_result(path)

    F = np.asarray(result["non_dominated_F"], dtype=np.float64)
    X = np.asarray(result["non_dominated_X"], dtype=np.int64)

    if F.ndim != 2 or F.shape[1] != 2 or len(F) == 0:
        raise ValueError(f"Invalid objective front: {path}")

    if X.ndim != 2 or X.shape[0] != F.shape[0]:
        raise ValueError(f"Invalid chromosome dimensions: {path}")

    if X.shape[1] != instance.n_tasks:
        raise ValueError(f"Incorrect chromosome length: {path}")

    if not np.all(np.isfinite(F)):
        raise ValueError(f"Non-finite objective values: {path}")

    if np.any(X < 0) or np.any(X >= instance.m_machines):
        raise AssertionError(f"Invalid machine assignment: {path}")

    tol = 1e-9

    if np.min(F[:, 0]) < instance.lower_bound_makespan - tol:
        raise AssertionError(f"Makespan lower-bound violation: {path}")

    if np.min(F[:, 1]) < instance.lower_bound_energy - tol:
        raise AssertionError(f"Energy lower-bound violation: {path}")

    metadata = result["metadata"]

    if metadata.get("instance_id") != instance.instance_id:
        raise AssertionError(f"Incorrect instance metadata: {path}")

    if metadata.get("n_tasks") != instance.n_tasks:
        raise AssertionError(f"Incorrect task-count metadata: {path}")

    if metadata.get("m_machines") != instance.m_machines:
        raise AssertionError(f"Incorrect machine-count metadata: {path}")

    if result["eval_count"] > expected_budget:
        raise AssertionError(f"Evaluation budget exceeded: {path}")

    if result["eval_count"] <= 0:
        raise AssertionError(f"Invalid evaluation count: {path}")

    return {
        "file": path.name,
        "front_size": len(F),
        "eval_count": result["eval_count"],
        "runtime": result["wall_clock_time_sec"],
    }


def run_stage2_batch(
    config_path="experiments/configs/stage2_scale.json",
    output_dir="data/raw_runs/stage2_scale",
    instance_seed=42,
    smoke_test=False,
):
    config_file = REPO_ROOT / config_path
    config = load_config(config_file)

    if smoke_test:
        scales = [{
            "name": "smoke",
            "n_tasks": 20,
            "m_machines": 4,
            "eval_budget": 100,
        }]
        seeds = [201, 202]
        variants = config["variants"]
        pop_size = 20
    else:
        scales = config["scales"]
        seeds = config["final_test_seeds"]
        variants = config["variants"]
        pop_size = config["pop_size"]

    output_root = REPO_ROOT / output_dir
    instance_dir = REPO_ROOT / "data" / "instances"

    # Prevent accidental overwriting of existing experiments.
    if output_root.exists():
        raise FileExistsError(
            f"Output directory already exists: {output_root}"
        )

    instance_dir.mkdir(parents=True, exist_ok=True)

    for scale in scales:
        expected_id = (
            f"inst_n{scale['n_tasks']}_m{scale['m_machines']}"
            f"_a{config['alpha']:g}_s{instance_seed}"
        )
        instance_file = instance_dir / f"{expected_id}.json"

        if instance_file.exists():
            raise FileExistsError(
                f"Instance already exists: {instance_file}"
            )

    output_root.mkdir(parents=True)

    generator = InstanceGenerator()
    all_archives = []
    total_start = time.perf_counter()

    print("=" * 72)
    print("STAGE 2 — SCALE-GENERALIZATION EXPERIMENT")
    print(f"Mode: {'SMOKE TEST' if smoke_test else 'PRODUCTION'}")
    print(f"Variants: {variants}")
    print(f"Seeds: {seeds}")
    print(f"Output: {output_root}")
    print("=" * 72)

    for scale in scales:
        name = scale["name"]
        n = scale["n_tasks"]
        m = scale["m_machines"]
        budget = scale["eval_budget"]

        if not smoke_test:
            expected_budget = config["eval_budget_multiplier"] * n
            if budget != expected_budget:
                raise ValueError(
                    f"Budget mismatch for {name}: "
                    f"{budget} != {expected_budget}"
                )

        print(f"\nProcessing scale: {name} ({n} tasks, {m} machines)")

        instance = generator.generate(
            n_tasks=n,
            m_machines=m,
            alpha=config["alpha"],
            kappa=config["kappa"],
            seed=instance_seed,
        )

        instance_path = instance.save(instance_dir)

        print(f"Instance: {instance.instance_id}")
        print(f"Instance saved: {instance_path}")
        print(f"LB makespan: {instance.lower_bound_makespan:.6f}")
        print(f"LB energy: {instance.lower_bound_energy:.6f}")
        print(f"Budget: {budget}")

        paths = run_experiment_batch(
            instance=instance,
            variants=variants,
            seeds=seeds,
            output_dir=output_root,
            n_evals=budget,
            pop_size=pop_size,
        )

        expected_count = len(variants) * len(seeds)

        if len(paths) != expected_count:
            raise AssertionError(
                f"{name}: expected {expected_count} archives, "
                f"received {len(paths)}"
            )

        for path in paths:
            stats = validate_result(path, instance, budget)
            all_archives.append(stats)

        print(f"PASS: {len(paths)} archives validated for {name}.")

    duration = time.perf_counter() - total_start

    print("\n" + "=" * 72)
    print("STAGE 2 EXECUTION COMPLETED")
    print(f"Total archives verified: {len(all_archives)}")
    print(f"Duration: {duration:.2f} seconds")
    print(f"Output directory: {output_root}")
    print("=" * 72)

    return {
        "status": "PASS",
        "total_files": len(all_archives),
        "duration_sec": duration,
        "output_dir": str(output_root),
    }


if __name__ == "__main__":
    args = parse_args()

    run_stage2_batch(
        config_path=args.config,
        output_dir=args.output_dir,
        instance_seed=args.instance_seed,
        smoke_test=args.smoke_test,
    )
