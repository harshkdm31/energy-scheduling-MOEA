
#!/usr/bin/env python3
"""
Stage 3 — Generalization and Robustness Batch Runner.

Runs experiments across:
- 2 workload scales
- 3 alpha values
- 20 independent problem instances per scale-alpha combination
- 20 algorithm seeds per instance

Supports instance-level resume. Completed instances are validated and
skipped. Incomplete or invalid instances are rerun.
"""

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.instance_generator import InstanceGenerator, load_instance
from src.runner import load_run_result, run_experiment_batch


def parse_args():
    parser = argparse.ArgumentParser(
        description="Execute Stage 3 generalization experiments."
    )
    parser.add_argument(
        "--config",
        default="experiments/configs/stage3_generalization.json",
    )
    parser.add_argument(
        "--output-dir",
        default="data/raw_runs/stage3_generalization",
    )
    parser.add_argument(
        "--instances-dir",
        default="data/instances",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run a miniature validation experiment.",
    )
    return parser.parse_args()


def load_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")

    with open(path, "r", encoding="utf-8-sig") as file:
        return json.load(file)


def validate_result(
    path: Path,
    instance,
    expected_budget: int,
    expected_variant: str,
    expected_seed: int,
):
    """Validate one saved experiment archive."""

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

    if not np.all(np.isfinite(X)):
        raise ValueError(f"Non-finite chromosome values: {path}")

    if np.any(X < 0) or np.any(X >= instance.m_machines):
        raise ValueError(f"Invalid machine assignment: {path}")

    tolerance = 1e-9

    if np.min(F[:, 0]) < instance.lower_bound_makespan - tolerance:
        raise ValueError(f"Makespan lower-bound violation: {path}")

    if np.min(F[:, 1]) < instance.lower_bound_energy - tolerance:
        raise ValueError(f"Energy lower-bound violation: {path}")

    metadata = result["metadata"]

    expected_metadata = {
        "instance_id": instance.instance_id,
        "n_tasks": instance.n_tasks,
        "m_machines": instance.m_machines,
        "variant": expected_variant,
        "seed": expected_seed,
        "requested_eval_budget": expected_budget,
    }

    for key, expected_value in expected_metadata.items():
        if metadata.get(key) != expected_value:
            raise ValueError(
                f"Metadata mismatch for '{key}' in {path}: "
                f"expected {expected_value}, got {metadata.get(key)}"
            )

    eval_count = int(result["eval_count"])

    if eval_count <= 0 or eval_count > expected_budget:
        raise ValueError(f"Invalid evaluation count: {path}")

    if metadata.get("eval_count") != eval_count:
        raise ValueError(f"Evaluation-count metadata mismatch: {path}")

    runtime = float(result["wall_clock_time_sec"])

    if not np.isfinite(runtime) or runtime < 0:
        raise ValueError(f"Invalid runtime: {path}")

    return {
        "file": path.name,
        "front_size": len(F),
        "eval_count": eval_count,
        "runtime": runtime,
    }


def validate_instance_archives(
    instance_output_dir: Path,
    instance,
    variants,
    seeds,
    budget,
):
    """
    Validate every expected archive for an instance.

    Returns archive statistics if the entire instance is complete.
    Returns None if any archive is missing or invalid.
    """

    expected_paths = [
        instance_output_dir / f"{variant}_seed{seed}.npz"
        for variant in variants
        for seed in seeds
    ]

    if not expected_paths:
        return None

    if not all(path.exists() for path in expected_paths):
        return None

    validated = []

    try:
        for variant in variants:
            for seed in seeds:
                path = instance_output_dir / f"{variant}_seed{seed}.npz"

                stats = validate_result(
                    path=path,
                    instance=instance,
                    expected_budget=budget,
                    expected_variant=variant,
                    expected_seed=seed,
                )
                validated.append(stats)

    except (ValueError, AssertionError, KeyError, OSError, EOFError) as exc:
        print(f"Existing instance archives failed validation: {exc}")
        return None

    return validated


def get_or_create_instance(
    generator,
    n_tasks,
    m_machines,
    alpha,
    kappa,
    seed,
    instances_dir,
):
    """Generate an instance or verify and reuse its existing JSON."""

    generated = generator.generate(
        n_tasks=n_tasks,
        m_machines=m_machines,
        alpha=alpha,
        kappa=kappa,
        seed=seed,
    )

    instance_path = instances_dir / f"{generated.instance_id}.json"

    if instance_path.exists():
        existing = load_instance(instance_path)

        if existing.to_dict() != generated.to_dict():
            raise ValueError(
                f"Existing instance does not match regenerated instance: "
                f"{instance_path}"
            )

        return existing, instance_path

    saved_path = generated.save(instances_dir)
    return generated, saved_path


def run_stage3_batch(
    config_path="experiments/configs/stage3_generalization.json",
    output_dir="data/raw_runs/stage3_generalization",
    instances_dir="data/instances",
    smoke_test=False,
):
    config_file = REPO_ROOT / config_path
    config = load_config(config_file)

    if smoke_test:
        scales = [
            {
                "name": "smoke",
                "n_tasks": 20,
                "m_machines": 4,
                "eval_budget": 100,
            }
        ]
        alpha_values = [2.0]
        instance_seeds = [42020]
        algorithm_seeds = [201, 202]
        variants = config["variants"]
        pop_size = 20
    else:
        scales = config["scales"]
        alpha_values = config["alpha_values"]
        instance_seeds = config["instance_seeds"]
        algorithm_seeds = config["final_test_seeds"]
        variants = config["variants"]
        pop_size = config["pop_size"]

    if not scales or not alpha_values or not instance_seeds:
        raise ValueError("Experiment configuration contains an empty matrix.")

    if not algorithm_seeds or not variants:
        raise ValueError("Algorithm seeds and variants must not be empty.")

    if len(set(instance_seeds)) != len(instance_seeds):
        raise ValueError("Instance seeds must be unique.")

    if len(set(algorithm_seeds)) != len(algorithm_seeds):
        raise ValueError("Algorithm seeds must be unique.")

    if len(set(variants)) != len(variants):
        raise ValueError("Algorithm variants must be unique.")

    output_root = REPO_ROOT / output_dir
    instance_root = REPO_ROOT / instances_dir

    output_root.mkdir(parents=True, exist_ok=True)
    instance_root.mkdir(parents=True, exist_ok=True)

    generator = InstanceGenerator()

    total_instances = (
        len(scales) * len(alpha_values) * len(instance_seeds)
    )

    expected_archives_per_instance = len(variants) * len(algorithm_seeds)

    total_expected_archives = (
        total_instances * expected_archives_per_instance
    )

    completed_instances = 0
    skipped_instances = 0
    rerun_instances = 0
    total_archives = 0

    total_start = time.perf_counter()

    print("=" * 76)
    print("STAGE 3 — GENERALIZATION AND ROBUSTNESS EXPERIMENT")
    print(f"Mode: {'SMOKE TEST' if smoke_test else 'PRODUCTION'}")
    print(f"Scales: {[scale['name'] for scale in scales]}")
    print(f"Alpha values: {alpha_values}")
    print(f"Independent instance seeds: {len(instance_seeds)}")
    print(f"Algorithm seeds per instance: {len(algorithm_seeds)}")
    print(f"Variants: {variants}")
    print(f"Expected independent instances: {total_instances}")
    print(f"Expected archive files: {total_expected_archives}")
    print(f"Output: {output_root}")
    print("=" * 76)

    for scale in scales:
        scale_name = scale["name"]
        n_tasks = scale["n_tasks"]
        m_machines = scale["m_machines"]
        budget = scale["eval_budget"]

        if not smoke_test:
            expected_budget = (
                config["eval_budget_multiplier"] * n_tasks
            )

            if budget != expected_budget:
                raise ValueError(
                    f"Budget mismatch for {scale_name}: "
                    f"{budget} != {expected_budget}"
                )

        for alpha in alpha_values:
            alpha_dir = output_root / scale_name / f"alpha_{alpha:g}"

            print("\n" + "-" * 76)
            print(
                f"Scale: {scale_name} | "
                f"Tasks: {n_tasks} | Machines: {m_machines} | "
                f"Alpha: {alpha}"
            )

            for instance_seed in instance_seeds:
                instance, instance_path = get_or_create_instance(
                    generator=generator,
                    n_tasks=n_tasks,
                    m_machines=m_machines,
                    alpha=alpha,
                    kappa=config["kappa"],
                    seed=instance_seed,
                    instances_dir=instance_root,
                )

                instance_output_dir = alpha_dir / instance.instance_id

                print(
                    f"\nInstance {completed_instances + skipped_instances + 1}"
                    f"/{total_instances}: {instance.instance_id}"
                )

                print(
                    f"Lower bounds — makespan: "
                    f"{instance.lower_bound_makespan:.6f}, "
                    f"energy: {instance.lower_bound_energy:.6f}"
                )

                print(f"Instance file: {instance_path}")

                existing_stats = validate_instance_archives(
                    instance_output_dir=instance_output_dir,
                    instance=instance,
                    variants=variants,
                    seeds=algorithm_seeds,
                    budget=budget,
                )

                if existing_stats is not None:
                    skipped_instances += 1
                    total_archives += len(existing_stats)

                    print(
                        f"SKIP: All {len(existing_stats)} archives "
                        f"already exist and passed validation."
                    )
                    continue

                if instance_output_dir.exists():
                    print(
                        "Incomplete or invalid instance detected. "
                        "Removing its output directory before rerunning."
                    )
                    shutil.rmtree(instance_output_dir)

                alpha_dir.mkdir(parents=True, exist_ok=True)

                print(
                    f"Running {len(variants)} variants × "
                    f"{len(algorithm_seeds)} seeds..."
                )

                paths = run_experiment_batch(
                    instance=instance,
                    variants=variants,
                    seeds=algorithm_seeds,
                    output_dir=alpha_dir,
                    n_evals=budget,
                    pop_size=pop_size,
                )

                if len(paths) != expected_archives_per_instance:
                    raise AssertionError(
                        f"{instance.instance_id}: expected "
                        f"{expected_archives_per_instance} archives, "
                        f"received {len(paths)}"
                    )

                validated_stats = []

                for variant in variants:
                    for seed in algorithm_seeds:
                        archive_path = (
                            instance_output_dir
                            / f"{variant}_seed{seed}.npz"
                        )

                        stats = validate_result(
                            path=archive_path,
                            instance=instance,
                            expected_budget=budget,
                            expected_variant=variant,
                            expected_seed=seed,
                        )

                        validated_stats.append(stats)

                completed_instances += 1
                rerun_instances += 1
                total_archives += len(validated_stats)

                print(
                    f"PASS: {len(validated_stats)} archives "
                    f"generated and validated."
                )

    duration = time.perf_counter() - total_start

    print("\n" + "=" * 76)
    print("STAGE 3 EXECUTION SUMMARY")
    print(f"Expected independent instances: {total_instances}")
    print(f"Newly completed instances: {completed_instances}")
    print(f"Previously completed instances skipped: {skipped_instances}")
    print(f"Instances rerun: {rerun_instances}")
    print(f"Archives validated or reused: {total_archives}")
    print(f"Expected archives: {total_expected_archives}")
    print(f"Duration: {duration:.2f} seconds")
    print(f"Output directory: {output_root}")

    if total_archives != total_expected_archives:
        raise AssertionError(
            f"Archive total mismatch: {total_archives} != "
            f"{total_expected_archives}"
        )

    print("STATUS: PASS")
    print("=" * 76)

    return {
        "status": "PASS",
        "total_instances": total_instances,
        "completed_instances": completed_instances,
        "skipped_instances": skipped_instances,
        "rerun_instances": rerun_instances,
        "total_files": total_archives,
        "duration_sec": duration,
        "output_dir": str(output_root),
    }


if __name__ == "__main__":
    args = parse_args()

    run_stage3_batch(
        config_path=args.config,
        output_dir=args.output_dir,
        instances_dir=args.instances_dir,
        smoke_test=args.smoke_test,
    )
