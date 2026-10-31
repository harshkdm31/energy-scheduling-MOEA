#!/usr/bin/env python3
"""
Automated Statistical Hypothesis Testing Engine (Deliverables B9 / Gate 3).

Loads raw experiment runs (.npz archives) from data/raw_runs/, computes the
2D Hypervolume (HV) indicator using analytical reference points
r = (1.1 * max f1, 1.1 * max f2), and executes formal non-parametric hypothesis
testing via:
  1. Paired two-tailed Wilcoxon signed-rank tests (nsga2_em vs nsga2_std, nsga2_em vs weighted_greedy).
  2. Holm-Bonferroni step-down family-wise error rate correction (alpha = 0.05).
  3. Cliff's Delta non-parametric effect sizes.
  4. Median and IQR distribution summaries.
Outputs a structured Markdown report to experiments/hypothesis_results.md.
"""

import argparse
import json
from pathlib import Path
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.runner import load_run_result
from src.statistics import (
    calculate_paired_rank_biserial,
    holm_bonferroni_correction,
    one_sample_wilcoxon_test,
    paired_wilcoxon_test,
    summarize_metric,
)


def compute_hypervolume_2d(front: np.ndarray, ref_point: np.ndarray) -> float:
    """
    Compute exact 2D hypervolume for a set of non-dominated objective points
    with respect to a reference point r = (r0, r1).

    Args:
        front: 2D array of shape (k, 2) where objectives are (makespan, energy).
        ref_point: 1D array of shape (2,) representing the upper bounding reference point.

    Returns:
        float: Hypervolume measure dominated by the front and bounded by ref_point.
    """
    F = np.asarray(front, dtype=np.float64)
    r = np.asarray(ref_point, dtype=np.float64)

    if F.ndim == 1 and len(F) == 2:
        F = F[np.newaxis, :]
    elif len(F) == 0:
        return 0.0

    # Filter points strictly dominating reference point
    valid_mask = (F[:, 0] < r[0]) & (F[:, 1] < r[1])
    F_valid = F[valid_mask]
    if len(F_valid) == 0:
        return 0.0

    # Sort ascending by f1 (makespan)
    sort_idx = np.argsort(F_valid[:, 0])
    F_sorted = F_valid[sort_idx]

    # Filter to non-dominated points: f2 must strictly decrease as f1 increases
    pareto_pts: List[np.ndarray] = [F_sorted[0]]
    for pt in F_sorted[1:]:
        if pt[1] < pareto_pts[-1][1]:
            pareto_pts.append(pt)

    pts = np.array(pareto_pts)
    k = len(pts)

    # Calculate exact 2D hypervolume using disjoint vertical slices
    hv = (r[0] - pts[-1, 0]) * (r[1] - pts[-1, 1])
    for i in range(k - 1):
        hv += (pts[i + 1, 0] - pts[i, 0]) * (r[1] - pts[i, 1])

    return float(hv)


def discover_instance_runs(
    raw_dir: Path,
    target_instance_ids: Optional[List[str]] = None,
    instances_dir: Optional[Path] = None,
    n_tasks_filter: Optional[int] = None,
    m_machines_filter: Optional[int] = None,
) -> Dict[str, Dict[str, Dict[int, Path]]]:
    """
    Scan raw_runs directory and organize runs by instance_id, variant, and seed.

    Returns:
        dict: {instance_id: {variant: {seed: file_path}}}
    """
    organized: Dict[str, Dict[str, Dict[int, Path]]] = {}

    if not raw_dir.exists():
        return organized

    for inst_folder in sorted(raw_dir.iterdir()):
        if not inst_folder.is_dir():
            continue
        instance_id = inst_folder.name
        if target_instance_ids and instance_id not in target_instance_ids:
            continue

        if instances_dir and (n_tasks_filter is not None or m_machines_filter is not None):
            meta = load_instance_metadata(instance_id, instances_dir)
            if n_tasks_filter is not None and meta.get("n_tasks") != n_tasks_filter:
                continue
            if m_machines_filter is not None and meta.get("m_machines") != m_machines_filter:
                continue

        npz_files = list(inst_folder.glob("*.npz"))
        if not npz_files:
            continue

        organized[instance_id] = {}
        for fpath in npz_files:
            # Match pattern: <variant>_seed<seed>.npz
            match = re.match(r"^(.+)_seed(\d+)\.npz$", fpath.name)
            if not match:
                continue
            variant = match.group(1)
            seed = int(match.group(2))

            if variant not in organized[instance_id]:
                organized[instance_id][variant] = {}
            organized[instance_id][variant][seed] = fpath

    return organized


def load_instance_metadata(instance_id: str, instances_dir: Path) -> Dict[str, Any]:
    """
    Load problem instance JSON metadata to extract alpha, dimensions, and lower bounds.
    """
    json_path = instances_dir / f"{instance_id}.json"
    if json_path.exists():
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)

    # Fallback to regex extraction from instance_id
    # e.g., inst_n200_m8_a2_s42 or inst_n20_m4_a1.5_s42
    match = re.match(r"^inst_n(\d+)_m(\d+)_a([0-9.]+)_s(\d+)$", instance_id)
    if match:
        return {
            "instance_id": instance_id,
            "n_tasks": int(match.group(1)),
            "m_machines": int(match.group(2)),
            "alpha": float(match.group(3)),
            "seed": int(match.group(4)),
        }
    return {"instance_id": instance_id, "alpha": 0.0}


def run_hypothesis_analysis(
    raw_dir: str = "data/raw_runs",
    instances_dir: str = "data/instances",
    output_report: str = "experiments/hypothesis_results.md",
    target_instances: Optional[List[str]] = None,
    alpha_significance: float = 0.05,
    n_tasks_filter: Optional[int] = None,
    m_machines_filter: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Execute end-to-end automated hypothesis testing and generate markdown summary table.
    """
    raw_path = REPO_ROOT / raw_dir
    inst_path = REPO_ROOT / instances_dir
    report_path = REPO_ROOT / output_report
    report_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("AUTOMATED STATISTICAL HYPOTHESIS TESTING ENGINE (Gate 3)")
    print("=" * 80)
    print(f"Scanning raw run archives from: {raw_path}")

    instance_runs = discover_instance_runs(
        raw_dir=raw_path,
        target_instance_ids=target_instances,
        instances_dir=inst_path,
        n_tasks_filter=n_tasks_filter,
        m_machines_filter=m_machines_filter,
    )
    if not instance_runs:
        raise RuntimeError(f"No experiment run data found in {raw_path}.")

    print(f"Found {len(instance_runs)} instance dataset(s): {list(instance_runs.keys())}")

    # Step 1: For each instance, determine reference point r = (1.1 * max f1, 1.1 * max f2)
    # and compute Hypervolume for all runs
    hv_by_instance: Dict[str, Dict[str, Dict[int, float]]] = {}
    instance_meta: Dict[str, Dict[str, Any]] = {}

    for inst_id, variants_dict in instance_runs.items():
        meta = load_instance_metadata(inst_id, inst_path)
        instance_meta[inst_id] = meta

        # Find maximum observed f1 and f2 across all variants and runs on this instance
        all_points: List[np.ndarray] = []
        for variant, seeds_dict in variants_dict.items():
            for seed, file_path in seeds_dict.items():
                data = load_run_result(file_path)
                F = np.asarray(data["non_dominated_F"], dtype=np.float64)
                if len(F) > 0:
                    all_points.append(F)

        if not all_points:
            continue

        stacked_F = np.vstack(all_points)
        max_f1 = float(np.max(stacked_F[:, 0]))
        max_f2 = float(np.max(stacked_F[:, 1]))

        # Reference point r = (1.1 * max f1, 1.1 * max f2)
        ref_point = np.array([1.1 * max_f1, 1.1 * max_f2], dtype=np.float64)

        hv_by_instance[inst_id] = {}
        for variant, seeds_dict in variants_dict.items():
            hv_by_instance[inst_id][variant] = {}
            for seed, file_path in seeds_dict.items():
                data = load_run_result(file_path)
                F = np.asarray(data["non_dominated_F"], dtype=np.float64)
                hv_val = compute_hypervolume_2d(F, ref_point)
                hv_by_instance[inst_id][variant][seed] = hv_val


    # Step 2: Formulate hypothesis comparisons for each instance / alpha.
    #
    # EM vs Standard: paired stochastic runs, matched by seed.
    # EM vs Random: paired stochastic runs, matched by seed.
    # EM vs Weighted Greedy: one-sample test against one fixed deterministic HV.
    comparisons = [
        ("nsga2_em", "nsga2_std", "paired",
         "HV: nsga2_em vs nsga2_std"),
        ("nsga2_em", "random", "paired",
         "HV: nsga2_em vs random"),
        ("nsga2_em", "weighted_greedy", "fixed_baseline",
         "HV: nsga2_em vs weighted_greedy"),
    ]

    hypothesis_records: List[Dict[str, Any]] = []

    sorted_instances = sorted(
        hv_by_instance.keys(),
        key=lambda i: instance_meta[i].get("alpha", 0.0),
    )

    for inst_id in sorted_instances:
        meta = instance_meta[inst_id]
        alpha_val = meta.get("alpha", "N/A")
        variants_dict = hv_by_instance[inst_id]

        for var_a, var_b, comparison_type, comp_label in comparisons:
            if var_a not in variants_dict or var_b not in variants_dict:
                continue

            seeds_a = variants_dict[var_a]
            seeds_b = variants_dict[var_b]

            if not seeds_a or not seeds_b:
                continue

            if comparison_type == "paired":
                common_seeds = sorted(
                    set(seeds_a.keys()).intersection(seeds_b.keys())
                )

                if len(common_seeds) == 0:
                    continue

                data_a = [seeds_a[s] for s in common_seeds]
                data_b = [seeds_b[s] for s in common_seeds]

                w_stat, p_val = paired_wilcoxon_test(data_a, data_b)

                effect_size = calculate_paired_rank_biserial(
                    data_a, data_b
                )

                summary_a = summarize_metric(data_a)
                summary_b = summarize_metric(data_b)

                n_samples = len(common_seeds)
                test_name = "Paired Wilcoxon signed-rank"
                effect_name = "Paired rank-biserial"

            else:
                # Weighted Greedy is deterministic per instance.
                # Its seed-labelled archives are duplicate representations
                # of one fixed baseline, not independent observations.
                greedy_values = np.asarray(
                    list(seeds_b.values()), dtype=np.float64
                )

                baseline_hv = float(greedy_values[0])

                if not np.allclose(
                    greedy_values,
                    baseline_hv,
                    rtol=1e-10,
                    atol=1e-12,
                ):
                    raise RuntimeError(
                        f"Weighted Greedy archives disagree for instance "
                        f"{inst_id}: {greedy_values.tolist()}"
                    )

                # Each stochastic EM run contributes one difference
                # against the same fixed deterministic baseline.
                common_seeds = sorted(seeds_a.keys())
                data_a = [seeds_a[s] for s in common_seeds]

                w_stat, p_val = one_sample_wilcoxon_test(
                    data_a,
                    reference=baseline_hv,
                )

                effect_size = calculate_paired_rank_biserial(
                    data_a,
                    [baseline_hv] * len(data_a),
                )

                summary_a = summarize_metric(data_a)
                summary_b = summarize_metric([baseline_hv])

                n_samples = len(data_a)
                test_name = "One-sample Wilcoxon signed-rank"
                effect_name = "One-sample rank-biserial"

            hypothesis_records.append({
                "metric_comp": comp_label,
                "instance_id": inst_id,
                "alpha": alpha_val,
                "comparison_type": comparison_type,
                "test_name": test_name,
                "effect_name": effect_name,
                "n_samples": n_samples,
                "summary_a": summary_a,
                "summary_b": summary_b,
                "wilcoxon_w": w_stat,
                "p_value": p_val,
                "rank_biserial": effect_size,
            })

    # Step 3: Apply Holm-Bonferroni step-down correction across the hypothesis family
    raw_p_values = [rec["p_value"] for rec in hypothesis_records]
    adjusted_results = holm_bonferroni_correction(raw_p_values, alpha=alpha_significance)

    for rec, (adj_p, is_sig) in zip(hypothesis_records, adjusted_results):
        rec["holm_adj_p"] = adj_p
        rec["is_significant"] = is_sig


    # Step 4: Generate Markdown Table Report
    md_lines: List[str] = [
        "# Stage 1 Hypothesis Testing Results",
        "",
        "## Statistical Methodology",
        "- **Metric**: 2D Hypervolume indicator ($HV$), evaluated using a common reference point for all variants and runs of each instance: $r = (1.1 \\times \\max f_1, 1.1 \\times \\max f_2)$.",
        "- **EM vs Standard and EM vs Random**: Paired two-tailed Wilcoxon signed-rank tests using matching stochastic run seeds.",
        "- **EM vs Weighted Greedy**: Two-tailed one-sample Wilcoxon signed-rank test on EM run differences from the single deterministic Weighted Greedy baseline.",
        "- **Effect Size**: Matched-pairs rank-biserial correlation. Positive values indicate higher HV for NSGA-II EM.",
        "- **Multiple Comparisons**: Holm-Bonferroni step-down correction across all hypotheses included in this analysis, at family-wise error rate $\\alpha = 0.05$.",
        "- **Summary**: Median and interquartile range (IQR).",
        "- **Experimental unit**: Stochastic algorithm runs on a fixed problem instance. Run seeds are not independent problem instances.",
        "",
        "## Summary Results Table",
        "",
        "| Metric / Comparison | alpha | Test | n | Median A (IQR) | Median B (IQR) | Wilcoxon W | p-value | Holm-Adj p | Rank-biserial | Significant? |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]

    for rec in hypothesis_records:
        med_a_str = (
            f"{rec['summary_a']['median']:.2f} "
            f"({rec['summary_a']['iqr']:.2f})"
        )
        med_b_str = (
            f"{rec['summary_b']['median']:.2f} "
            f"({rec['summary_b']['iqr']:.2f})"
        )
        w_val_str = f"{rec['wilcoxon_w']:.1f}"
        p_val_str = (
            f"{rec['p_value']:.4e}"
            if rec["p_value"] < 0.0001
            else f"{rec['p_value']:.4f}"
        )
        adj_p_str = (
            f"{rec['holm_adj_p']:.4e}"
            if rec["holm_adj_p"] < 0.0001
            else f"{rec['holm_adj_p']:.4f}"
        )
        effect_str = f"{rec['rank_biserial']:+.3f}"
        sig_str = "**YES**" if rec["is_significant"] else "NO"

        alpha_str = (
            f"{rec['alpha']}"
            if isinstance(rec["alpha"], str)
            else f"{rec['alpha']:.1f}"
        )

        row = (
            f"| {rec['metric_comp']} | {alpha_str} | "
            f"{rec['test_name']} | {rec['n_samples']} | "
            f"{med_a_str} | {med_b_str} | {w_val_str} | "
            f"{p_val_str} | {adj_p_str} | {effect_str} | {sig_str} |"
        )
        md_lines.append(row)

    md_lines.append("")
    report_content = "\n".join(md_lines)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)

    print("\n" + report_content)
    print("=" * 80)
    print(f"Results report successfully saved to: {report_path}")
    print("=" * 80)

    return {
        "report_file": str(report_path),
        "hypotheses_tested": len(hypothesis_records),
        "records": hypothesis_records,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run automated statistical hypothesis testing on batch experiment outputs."
    )
    parser.add_argument(
        "--raw-dir",
        type=str,
        default="data/raw_runs",
        help="Directory containing raw .npz experiment runs.",
    )
    parser.add_argument(
        "--instances-dir",
        type=str,
        default="data/instances",
        help="Directory containing problem instance JSON files.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="experiments/hypothesis_results.md",
        help="Output markdown report filepath.",
    )
    parser.add_argument(
        "--instances",
        nargs="+",
        type=str,
        default=None,
        help="Optional list of specific instance_ids to analyze.",
    )
    parser.add_argument(
        "--n-tasks",
        type=int,
        default=None,
        help="Optional filter for number of tasks n.",
    )
    parser.add_argument(
        "--m-machines",
        type=int,
        default=None,
        help="Optional filter for number of machines m.",
    )
    parser.add_argument(
        "--alpha-sig",
        type=float,
        default=0.05,
        help="Target family-wise error rate alpha for Holm-Bonferroni correction.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    cli_args = parse_args()
    run_hypothesis_analysis(
        raw_dir=cli_args.raw_dir,
        instances_dir=cli_args.instances_dir,
        output_report=cli_args.output,
        target_instances=cli_args.instances,
        alpha_significance=cli_args.alpha_sig,
        n_tasks_filter=cli_args.n_tasks,
        m_machines_filter=cli_args.m_machines,
    )
