
#!/usr/bin/env python3
"""
Stage 3: Generalization and Robustness Statistical Analysis.

Experimental design:
- 2 scales: small and large.
- 3 alpha values: 1.0, 2.0, 3.0.
- 20 independent problem instances per scale-alpha condition.
- 20 algorithm seeds per instance.
- Algorithms: random, nsga2_std, nsga2_em, weighted_greedy.

The independent problem instance is the statistical experimental unit.
Algorithm seeds are nested repetitions within each instance and are NOT
treated as independent problem instances.

For each instance:
1. Compute one common hypervolume reference point across all algorithms
   and seeds.
2. Compute HV for each archive.
3. Aggregate stochastic algorithm runs using their median HV.
4. Count Weighted Greedy once per instance after verifying its
   seed-labelled archives are identical.

For each scale-alpha condition:
- Compare EM against standard NSGA-II, random, and Weighted Greedy.
- Use paired two-tailed Wilcoxon signed-rank tests across instances.
- Apply Holm-Bonferroni correction across all 18 hypotheses.
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.runner import load_run_result
from src.statistics import (
    calculate_paired_rank_biserial,
    holm_bonferroni_correction,
    paired_wilcoxon_test,
    summarize_metric,
)


EXPECTED_VARIANTS = [
    "random",
    "nsga2_std",
    "nsga2_em",
    "weighted_greedy",
]

STOCHASTIC_VARIANTS = [
    "random",
    "nsga2_std",
    "nsga2_em",
]

COMPARISONS = [
    ("nsga2_em", "nsga2_std", "EM vs Standard NSGA-II"),
    ("nsga2_em", "random", "EM vs Random"),
    ("nsga2_em", "weighted_greedy", "EM vs Weighted Greedy"),
]


def compute_hypervolume_2d(
    front: np.ndarray,
    ref_point: np.ndarray,
) -> float:
    """Compute exact 2D hypervolume for minimization objectives."""

    F = np.asarray(front, dtype=np.float64)
    r = np.asarray(ref_point, dtype=np.float64)

    if r.shape != (2,):
        raise ValueError("Reference point must contain exactly two values.")

    if F.ndim == 1 and F.size == 2:
        F = F.reshape(1, 2)

    if F.ndim != 2 or F.shape[1] != 2:
        raise ValueError("Objective front must have shape (n, 2).")

    if len(F) == 0:
        return 0.0

    if not np.all(np.isfinite(F)):
        raise ValueError("Objective front contains non-finite values.")

    valid = (F[:, 0] < r[0]) & (F[:, 1] < r[1])
    points = F[valid]

    if len(points) == 0:
        return 0.0

    points = points[np.argsort(points[:, 0])]

    pareto = [points[0]]

    for point in points[1:]:
        if point[1] < pareto[-1][1]:
            pareto.append(point)

    points = np.asarray(pareto)
    hv = (r[0] - points[-1, 0]) * (r[1] - points[-1, 1])

    for i in range(len(points) - 1):
        hv += (
            (points[i + 1, 0] - points[i, 0])
            * (r[1] - points[i, 1])
        )

    return float(hv)


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load and validate the Stage 3 experiment configuration."""

    with open(config_path, "r", encoding="utf-8-sig") as file:
        config = json.load(file)

    required = [
        "scales",
        "alpha_values",
        "instance_seeds",
        "final_test_seeds",
        "variants",
    ]

    for key in required:
        if key not in config:
            raise ValueError(f"Missing configuration field: {key}")

    if len(config["instance_seeds"]) != 20:
        raise ValueError("Stage 3 requires exactly 20 instance seeds.")

    if len(config["final_test_seeds"]) != 20:
        raise ValueError("Stage 3 requires exactly 20 algorithm seeds.")

    if set(config["variants"]) != set(EXPECTED_VARIANTS):
        raise ValueError("Unexpected Stage 3 algorithm variant list.")

    return config


def load_instance_metadata(
    instance_id: str,
    instances_dir: Path,
) -> Dict[str, Any]:
    """Load metadata for a generated problem instance."""

    path = instances_dir / f"{instance_id}.json"

    if not path.exists():
        raise FileNotFoundError(
            f"Missing instance metadata: {path}"
        )

    with open(path, "r", encoding="utf-8-sig") as file:
        metadata = json.load(file)

    required = [
        "instance_id",
        "n_tasks",
        "m_machines",
        "alpha",
        "seed",
    ]

    for key in required:
        if key not in metadata:
            raise ValueError(
                f"Missing metadata field '{key}' in {path}"
            )

    return metadata


def discover_archives(
    raw_dir: Path,
) -> Dict[str, Dict[str, Dict[int, Path]]]:
    """
    Discover archives recursively.

    Supports the Stage 3 hierarchy:
    scale/alpha_<value>/<instance_id>/<variant>_seed<seed>.npz

    Returns:
        {instance_id: {variant: {algorithm_seed: archive_path}}}
    """

    organized = defaultdict(lambda: defaultdict(dict))

    if not raw_dir.exists():
        raise FileNotFoundError(
            f"Raw experiment directory does not exist: {raw_dir}"
        )

    pattern = re.compile(r"^(.+)_seed(\d+)\.npz$")

    for path in sorted(raw_dir.rglob("*.npz")):
        match = pattern.match(path.name)

        if not match:
            continue

        variant = match.group(1)
        seed = int(match.group(2))
        instance_id = path.parent.name

        if variant not in EXPECTED_VARIANTS:
            continue

        if seed in organized[instance_id][variant]:
            raise ValueError(
                f"Duplicate archive for {instance_id}, "
                f"{variant}, seed {seed}"
            )

        organized[instance_id][variant][seed] = path

    return {
        instance_id: {
            variant: dict(seed_paths)
            for variant, seed_paths in variants.items()
        }
        for instance_id, variants in organized.items()
    }


def validate_instance_archives(
    instance_id: str,
    variants: Dict[str, Dict[int, Path]],
    expected_seeds: List[int],
) -> None:
    """Require complete algorithm and seed coverage for an instance."""

    for variant in EXPECTED_VARIANTS:
        if variant not in variants:
            raise ValueError(
                f"{instance_id}: missing variant {variant}"
            )

        observed = set(variants[variant])
        expected = set(expected_seeds)

        if observed != expected:
            raise ValueError(
                f"{instance_id}: incomplete seeds for {variant}. "
                f"Expected {sorted(expected)}, got {sorted(observed)}"
            )


def analyze_instance(
    instance_id: str,
    variants: Dict[str, Dict[int, Path]],
) -> Dict[str, Any]:
    """
    Compute per-run HV and per-instance algorithm summaries.

    One common reference point is used across all variants and seeds
    belonging to this instance.
    """

    loaded = {}
    all_fronts = []

    for variant, seed_paths in variants.items():
        loaded[variant] = {}

        for seed, path in seed_paths.items():
            data = load_run_result(path)

            front = np.asarray(
                data["non_dominated_F"],
                dtype=np.float64,
            )

            if front.ndim != 2 or front.shape[1] != 2:
                raise ValueError(
                    f"Invalid objective front in {path}"
                )

            if len(front) == 0 or not np.all(np.isfinite(front)):
                raise ValueError(
                    f"Empty or non-finite objective front in {path}"
                )

            loaded[variant][seed] = front
            all_fronts.append(front)

    stacked = np.vstack(all_fronts)

    max_f1 = float(np.max(stacked[:, 0]))
    max_f2 = float(np.max(stacked[:, 1]))

    reference_point = np.array(
        [1.1 * max_f1, 1.1 * max_f2],
        dtype=np.float64,
    )

    run_hv = {}

    for variant, seed_fronts in loaded.items():
        run_hv[variant] = {
            seed: compute_hypervolume_2d(front, reference_point)
            for seed, front in seed_fronts.items()
        }

    # Verify deterministic Weighted Greedy archives are duplicates.
    greedy_values = np.asarray(
        list(run_hv["weighted_greedy"].values()),
        dtype=np.float64,
    )

    if not np.allclose(
        greedy_values,
        greedy_values[0],
        rtol=1e-10,
        atol=1e-12,
    ):
        raise ValueError(
            f"Weighted Greedy archives disagree for {instance_id}."
        )

    # One observation per independent instance and algorithm.
    instance_values = {}

    for variant in STOCHASTIC_VARIANTS:
        values = np.asarray(
            list(run_hv[variant].values()),
            dtype=np.float64,
        )

        instance_values[variant] = float(np.median(values))

    instance_values["weighted_greedy"] = float(greedy_values[0])

    within_instance_iqr = {}

    for variant in STOCHASTIC_VARIANTS:
        values = np.asarray(
            list(run_hv[variant].values()),
            dtype=np.float64,
        )

        within_instance_iqr[variant] = float(
            np.percentile(values, 75)
            - np.percentile(values, 25)
        )

    return {
        "instance_id": instance_id,
        "reference_point": reference_point.tolist(),
        "run_hv": run_hv,
        "instance_values": instance_values,
        "within_instance_iqr": within_instance_iqr,
    }


def run_stage3_analysis(
    raw_dir: str = "data/raw_runs/stage3_generalization",
    instances_dir: str = "data/instances",
    config_path: str = "experiments/configs/stage3_generalization.json",
    output_report: str = "experiments/stage3_hypothesis_results.md",
    alpha_significance: float = 0.05,
) -> Dict[str, Any]:
    """Run the complete Stage 3 instance-level statistical analysis."""

    raw_path = REPO_ROOT / raw_dir
    instances_path = REPO_ROOT / instances_dir
    config_file = REPO_ROOT / config_path
    report_path = REPO_ROOT / output_report

    config = load_config(config_file)

    expected_algorithm_seeds = sorted(
        int(seed) for seed in config["final_test_seeds"]
    )
    expected_instance_seeds = set(
        int(seed) for seed in config["instance_seeds"]
    )

    archives = discover_archives(raw_path)

    if not archives:
        raise RuntimeError(
            f"No Stage 3 archives found in {raw_path}"
        )

    print("=" * 80)
    print("STAGE 3: INSTANCE-LEVEL GENERALIZATION ANALYSIS")
    print("=" * 80)
    print(f"Discovered instances: {len(archives)}")

    # Map each instance to its configured scale and alpha condition.
    scale_lookup = {}

    for scale in config["scales"]:
        scale_lookup[
            (scale["n_tasks"], scale["m_machines"])
        ] = scale["name"]

    instance_results = {}
    condition_instances = defaultdict(list)

    for instance_id, variant_archives in sorted(archives.items()):
        metadata = load_instance_metadata(
            instance_id,
            instances_path,
        )

        dimensions = (
            metadata["n_tasks"],
            metadata["m_machines"],
        )

        if dimensions not in scale_lookup:
            raise ValueError(
                f"Unexpected dimensions for {instance_id}: {dimensions}"
            )

        scale_name = scale_lookup[dimensions]
        alpha_value = float(metadata["alpha"])
        instance_seed = int(metadata["seed"])

        if instance_seed not in expected_instance_seeds:
            raise ValueError(
                f"Unexpected instance seed {instance_seed}: {instance_id}"
            )

        if alpha_value not in [
            float(value) for value in config["alpha_values"]
        ]:
            raise ValueError(
                f"Unexpected alpha {alpha_value}: {instance_id}"
            )

        validate_instance_archives(
            instance_id,
            variant_archives,
            expected_algorithm_seeds,
        )

        result = analyze_instance(
            instance_id,
            variant_archives,
        )

        result["metadata"] = metadata
        result["scale"] = scale_name
        result["alpha"] = alpha_value
        result["instance_seed"] = instance_seed

        instance_results[instance_id] = result

        condition_instances[
            (scale_name, alpha_value)
        ].append(result)

    # Enforce the full 2 x 3 x 20 independent-instance design.
    expected_conditions = [
        (scale["name"], float(alpha))
        for scale in config["scales"]
        for alpha in config["alpha_values"]
    ]

    for condition in expected_conditions:
        results = condition_instances[condition]

        observed_seeds = [
            result["instance_seed"] for result in results
        ]

        if len(results) != 20:
            raise ValueError(
                f"{condition}: expected 20 instances, "
                f"found {len(results)}"
            )

        if len(set(observed_seeds)) != 20:
            raise ValueError(
                f"{condition}: duplicate instance seeds found."
            )

        if set(observed_seeds) != expected_instance_seeds:
            raise ValueError(
                f"{condition}: instance seed coverage is incomplete."
            )

    print("All 120 independent instances validated.")

    # Build the 18 hypothesis comparisons.
    hypothesis_records = []

    for scale_name, alpha_value in expected_conditions:
        results = sorted(
            condition_instances[(scale_name, alpha_value)],
            key=lambda item: item["instance_seed"],
        )

        for variant_a, variant_b, label in COMPARISONS:
            values_a = [
                result["instance_values"][variant_a]
                for result in results
            ]

            values_b = [
                result["instance_values"][variant_b]
                for result in results
            ]

            statistic, p_value = paired_wilcoxon_test(
                values_a,
                values_b,
                method="approx",
            )

            effect = calculate_paired_rank_biserial(
                values_a,
                values_b,
            )

            differences = np.asarray(values_a) - np.asarray(values_b)

            wins = int(np.sum(differences > 0))
            losses = int(np.sum(differences < 0))
            ties = int(np.sum(differences == 0))

            hypothesis_records.append({
                "scale": scale_name,
                "alpha": alpha_value,
                "comparison": label,
                "variant_a": variant_a,
                "variant_b": variant_b,
                "n_instances": len(results),
                "summary_a": summarize_metric(values_a),
                "summary_b": summarize_metric(values_b),
                "wilcoxon_w": float(statistic),
                "p_value": float(p_value),
                "rank_biserial": float(effect),
                "wins": wins,
                "losses": losses,
                "ties": ties,
            })

    if len(hypothesis_records) != 18:
        raise RuntimeError(
            f"Expected 18 hypotheses, got {len(hypothesis_records)}"
        )

    # Holm correction across all 18 comparisons.
    adjusted = holm_bonferroni_correction(
        [record["p_value"] for record in hypothesis_records],
        alpha=alpha_significance,
    )

    for record, (adjusted_p, significant) in zip(
        hypothesis_records,
        adjusted,
    ):
        record["holm_adjusted_p"] = float(adjusted_p)
        record["significant"] = bool(significant)

    # Descriptive algorithm summaries across independent instances.
    descriptive_records = []

    for scale_name, alpha_value in expected_conditions:
        results = condition_instances[(scale_name, alpha_value)]

        for variant in EXPECTED_VARIANTS:
            instance_values = [
                result["instance_values"][variant]
                for result in results
            ]

            within_iqrs = [
                result["within_instance_iqr"][variant]
                for result in results
                if variant in result["within_instance_iqr"]
            ]

            descriptive_records.append({
                "scale": scale_name,
                "alpha": alpha_value,
                "variant": variant,
                "summary": summarize_metric(instance_values),
                "median_within_instance_iqr": (
                    float(np.median(within_iqrs))
                    if within_iqrs else None
                ),
            })

    # Generate report.
    lines = [
        "# Stage 3: Generalization and Robustness Results",
        "",
        "## Experimental Design",
        "",
        "- **Scales**: Small (50 tasks, 4 machines) and large "
        "(500 tasks, 16 machines).",
        "- **Alpha values**: 1.0, 2.0, and 3.0.",
        "- **Independent problem instances**: 20 per scale-alpha condition.",
        "- **Algorithm seeds**: 20 per independent instance.",
        "- **Total independent instances**: 120.",
        "- **Stochastic algorithm runs**: 7,200.",
        "- **Algorithms**: Random, standard NSGA-II, energy-aware NSGA-II, "
        "and Weighted Greedy.",
        "",
        "## Statistical Methodology",
        "",
        "1. Compute a common two-objective hypervolume reference point "
        "for all algorithms and seeds within each instance:",
        "   `r = (1.1 * max(f1), 1.1 * max(f2))`.",
        "2. Calculate hypervolume separately for every run.",
        "3. Aggregate the 20 stochastic runs using the median HV for each "
        "algorithm on each instance.",
        "4. Count Weighted Greedy once per instance after verifying that "
        "its seed-labelled archives are identical.",
        "5. Compare algorithms across the 20 independent instances using "
        "paired two-tailed Wilcoxon signed-rank tests.",
        "6. Apply Holm-Bonferroni correction across all 18 hypotheses.",
        "",
        "**Experimental unit:** The independent problem instance. "
        "Algorithm seeds are nested repetitions and are not treated as "
        "independent instances.",
        "",
        "**Effect size:** Paired rank-biserial correlation. Positive "
        "values indicate higher HV for the first algorithm in the comparison.",
        "",
        "**Wins / losses / ties:** Counts across the 20 paired "
        "independent instances.",
        "",
        "**Within-instance variability:** Median of the per-instance "
        "interquartile ranges across stochastic algorithm seeds. "
        "Not applicable to deterministic Weighted Greedy.",
        "",
        "**Reference-point limitation:** The common reference point is "
        "derived from the observed fronts for each instance after the "
        "runs. It is common across algorithms within an instance, but "
        "is data-dependent rather than externally fixed.",
        "",
        "**Energy model:** Energy is modeled by the experiment's objective "
        "function; it is not a direct measurement of physical energy use.",
        "",
        "## Descriptive Results",
        "",
        "Values summarize the 20 independent instance-level observations. "
        "For stochastic algorithms, each observation is the median HV "
        "across 20 seeds on that instance.",
        "",
        "| Scale | Alpha | Algorithm | Median instance HV (IQR) | "
        "Median within-instance seed IQR |",
        "|---|---:|---|---:|---:|",
    ]

    for record in descriptive_records:
        summary = record["summary"]
        median_iqr = (
            f"{record['median_within_instance_iqr']:.2f}"
            if record["median_within_instance_iqr"] is not None
            else "N/A (deterministic)"
        )

        lines.append(
            f"| {record['scale']} | {record['alpha']:.1f} "
            f"| {record['variant']} "
            f"| {summary['median']:.2f} ({summary['iqr']:.2f}) "
            f"| {median_iqr} |"
        )

    lines.extend([
        "",
        "## Hypothesis Testing Results",
        "",
        "| Scale | Alpha | Comparison | n instances | "
        "Median A (IQR) | Median B (IQR) | W | p-value | "
        "Holm-adjusted p | Rank-biserial | Wins / Losses / Ties | "
        "Significant? |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ])

    for record in hypothesis_records:
        a = record["summary_a"]
        b = record["summary_b"]

        p_value = f"{record['p_value']:.4e}"
        adjusted_p = f"{record['holm_adjusted_p']:.4e}"
        significance = "YES" if record["significant"] else "NO"

        lines.append(
            f"| {record['scale']} "
            f"| {record['alpha']:.1f} "
            f"| {record['comparison']} "
            f"| {record['n_instances']} "
            f"| {a['median']:.2f} ({a['iqr']:.2f}) "
            f"| {b['median']:.2f} ({b['iqr']:.2f}) "
            f"| {record['wilcoxon_w']:.1f} "
            f"| {p_value} "
            f"| {adjusted_p} "
            f"| {record['rank_biserial']:+.3f} "
            f"| {record['wins']} / {record['losses']} / "
            f"{record['ties']} "
            f"| {significance} |"
        )

    lines.extend([
        "",
        "## Interpretation Notes",
        "",
        "- Statistical conclusions are based on independent problem "
        "instances, not on algorithm seeds treated as independent samples.",
        "- Results apply to the tested workload generator, machine "
        "heterogeneity model, scales, and parameter settings.",
        "- A statistically significant result does not by itself establish "
        "practical importance or universal algorithm superiority.",
        "- The hypervolume reference point is data-dependent and should "
        "be considered when interpreting comparisons across instances.",
        "- Energy values represent the modeled objective, not measured "
        "hardware energy consumption.",
        "",
    ])

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")

    report = "\n".join(lines)

    print("=" * 80)
    print(report)
    print("=" * 80)
    print(f"Report saved to: {report_path}")
    print(f"Hypotheses tested: {len(hypothesis_records)}")

    return {
        "report_file": str(report_path),
        "instances_analyzed": len(instance_results),
        "conditions_analyzed": len(expected_conditions),
        "hypotheses_tested": len(hypothesis_records),
        "instance_results": instance_results,
        "hypothesis_records": hypothesis_records,
        "descriptive_records": descriptive_records,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze Stage 3 multi-instance experiment results."
    )

    parser.add_argument(
        "--raw-dir",
        default="data/raw_runs/stage3_generalization",
    )

    parser.add_argument(
        "--instances-dir",
        default="data/instances",
    )

    parser.add_argument(
        "--config",
        default="experiments/configs/stage3_generalization.json",
    )

    parser.add_argument(
        "--output",
        default="experiments/stage3_hypothesis_results.md",
    )

    parser.add_argument(
        "--alpha-sig",
        type=float,
        default=0.05,
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    run_stage3_analysis(
        raw_dir=args.raw_dir,
        instances_dir=args.instances_dir,
        config_path=args.config,
        output_report=args.output,
        alpha_significance=args.alpha_sig,
    )
