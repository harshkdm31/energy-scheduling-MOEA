"""
Empirical visualization suite for multi-objective energy scheduling benchmarks.
(Deliverable B9 / Phase 6).

Provides publication-ready plotting utilities:
  1. plot_pareto_fronts: Non-dominated Pareto front comparisons in (f1, f2) space
     with analytical lower bound guidelines (LB_C, LB_E).
  2. plot_metric_boxplots: Distribution boxplots comparing performance indicators
     (e.g., Hypervolume) across random test seeds with Median and IQR annotations.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import matplotlib
matplotlib.use("Agg")  # Enforce headless rendering
import matplotlib.pyplot as plt
import numpy as np


# Cohesive, publication-grade color and marker palette
PALETTE = {
    "nsga2_std": {"color": "#1f77b4", "marker": "o", "label": "NSGA-II (Std)", "alpha": 0.85},
    "nsga2_em": {"color": "#2ca02c", "marker": "s", "label": "NSGA-II (Energy-Aware Mutation)", "alpha": 0.9},
    "weighted_greedy": {"color": "#d62728", "marker": "^", "label": "Weighted-Greedy Sweep", "alpha": 0.95},
    "random": {"color": "#7f7f7f", "marker": "x", "label": "Random Search", "alpha": 0.6},
}

DEFAULT_STYLE = {"color": "#9467bd", "marker": "d", "label": "Other Variant", "alpha": 0.8}


def _extract_objective_array(val: Any) -> np.ndarray:
    """
    Extract a 2D float64 numpy array of shape (N, 2) from diverse input representations.
    """
    if isinstance(val, dict):
        if "non_dominated_F" in val:
            arr = np.asarray(val["non_dominated_F"], dtype=np.float64)
        elif "objectives" in val:
            arr = np.asarray(val["objectives"], dtype=np.float64)
        elif "F" in val:
            arr = np.asarray(val["F"], dtype=np.float64)
        else:
            arr = np.asarray(list(val.values()), dtype=np.float64)
    else:
        arr = np.asarray(val, dtype=np.float64)

    if arr.ndim == 1 and len(arr) == 2:
        arr = arr[np.newaxis, :]
    elif arr.ndim == 1 and len(arr) == 0:
        arr = np.empty((0, 2), dtype=np.float64)

    return arr


def plot_pareto_fronts(
    archive_dict: Dict[str, Any],
    instance: Any,
    save_path: Union[str, Path],
    title: Optional[str] = None,
    dpi: int = 300,
) -> Path:
    """
    Plot non-dominated Pareto solutions for algorithms in makespan (f1) vs. energy (f2) space
    with explicit analytical lower-bound reference guidelines (Deliverable B9).

    Args:
        archive_dict: Dict mapping variant names to objective arrays or result dicts.
        instance: ProblemInstance object or dict containing lower bounds and instance metadata.
        save_path: Destination path for the saved figure.
        title: Optional plot title override.
        dpi: Resolution of saved image (default: 300 DPI).

    Returns:
        Path: Absolute path of saved figure.
    """
    out_file = Path(save_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    # Extract instance metadata and lower bounds
    lb_c = getattr(instance, "lower_bound_makespan", None)
    lb_e = getattr(instance, "lower_bound_energy", None)
    instance_id = getattr(instance, "instance_id", None)
    alpha = getattr(instance, "alpha", None)

    if isinstance(instance, dict):
        lb_c = lb_c if lb_c is not None else instance.get("lower_bound_makespan", 0.0)
        lb_e = lb_e if lb_e is not None else instance.get("lower_bound_energy", 0.0)
        instance_id = instance_id if instance_id is not None else instance.get("instance_id", "Instance")
        alpha = alpha if alpha is not None else instance.get("alpha", 2.0)

    fig, ax = plt.subplots(figsize=(8, 6), dpi=dpi)

    all_f1: List[float] = []
    all_f2: List[float] = []

    # Plot each algorithm variant
    for variant_name, data in archive_dict.items():
        F = _extract_objective_array(data)
        if len(F) == 0:
            continue

        all_f1.extend(F[:, 0].tolist())
        all_f2.extend(F[:, 1].tolist())

        style = PALETTE.get(variant_name, DEFAULT_STYLE)
        label = style["label"]

        # Sort front for line connectivity
        sort_idx = np.argsort(F[:, 0])
        F_sorted = F[sort_idx]

        # Scatter plot non-dominated points
        ax.scatter(
            F_sorted[:, 0],
            F_sorted[:, 1],
            c=style["color"],
            marker=style["marker"],
            s=45,
            alpha=style["alpha"],
            label=label,
            edgecolors="k",
            linewidths=0.5,
            zorder=4,
        )

        # Subtle connecting line
        ax.plot(
            F_sorted[:, 0],
            F_sorted[:, 1],
            color=style["color"],
            linestyle="-",
            linewidth=1.2,
            alpha=0.5,
            zorder=3,
        )

    # Plot analytical lower bound reference lines
    if lb_c is not None and lb_c > 0:
        ax.axvline(
            x=lb_c,
            color="#e377c2",
            linestyle="--",
            linewidth=1.5,
            alpha=0.85,
            label=f"Analytical $LB_C$ ({lb_c:.2f})",
            zorder=2,
        )
    if lb_e is not None and lb_e > 0:
        ax.axhline(
            y=lb_e,
            color="#bcbd22",
            linestyle="--",
            linewidth=1.5,
            alpha=0.85,
            label=f"Analytical $LB_E$ ({lb_e:.2f})",
            zorder=2,
        )

    # Set titles and clean axes
    plot_title = (
        title
        if title is not None
        else f"Pareto Frontier Comparison ($\\alpha = {alpha}$, {instance_id})"
    )
    ax.set_title(plot_title, fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Makespan $C_{\\max}$ ($f_1$)", fontsize=11, labelpad=8)
    ax.set_ylabel("Total Energy Consumption $E$ ($f_2$)", fontsize=11, labelpad=8)

    ax.grid(True, linestyle="--", alpha=0.5, zorder=1)
    ax.legend(loc="upper right", framealpha=0.92, fontsize=9)

    plt.tight_layout()
    plt.savefig(out_file, dpi=dpi)
    plt.close(fig)

    return out_file


def plot_metric_boxplots(
    metrics_by_variant: Dict[str, Union[List[float], np.ndarray]],
    metric_name: str,
    save_path: Union[str, Path],
    title: Optional[str] = None,
    dpi: int = 300,
) -> Path:
    """
    Generate publication-ready boxplots comparing metric distributions across variants,
    annotating Median and Interquartile Range (IQR) (Deliverable B9).

    Args:
        metrics_by_variant: Dict mapping variant names to distributions (e.g. Hypervolume values).
        metric_name: Name of metric (e.g. "Hypervolume (HV)").
        save_path: Destination path for the saved figure.
        title: Optional plot title override.
        dpi: Resolution of saved image (default: 300 DPI).

    Returns:
        Path: Absolute path of saved figure.
    """
    out_file = Path(save_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    variant_names = list(metrics_by_variant.keys())
    data_lists = [
        np.asarray(metrics_by_variant[v], dtype=np.float64).ravel()
        for v in variant_names
    ]

    fig, ax = plt.subplots(figsize=(8, 6), dpi=dpi)

    # Display boxplots
    bp = ax.boxplot(
        data_lists,
        patch_artist=True,
        tick_labels=[PALETTE.get(v, DEFAULT_STYLE)["label"] for v in variant_names],
        widths=0.45,
        medianprops=dict(color="#d62728", linewidth=2.0),
        boxprops=dict(facecolor="#c6dbef", color="#1f77b4", linewidth=1.2),
        whiskerprops=dict(color="#1f77b4", linewidth=1.2),
        capprops=dict(color="#1f77b4", linewidth=1.2),
        flierprops=dict(marker="o", color="#d62728", alpha=0.6, markersize=5),
    )

    # Color boxes according to PALETTE
    for patch, v in zip(bp["boxes"], variant_names):
        c = PALETTE.get(v, DEFAULT_STYLE)["color"]
        patch.set_facecolor(c)
        patch.set_alpha(0.35)

    # Annotate Median and IQR values
    for idx, (v, arr) in enumerate(zip(variant_names, data_lists), start=1):
        if len(arr) == 0:
            continue
        med = float(np.median(arr))
        q25 = float(np.percentile(arr, 25))
        q75 = float(np.percentile(arr, 75))
        iqr = q75 - q25

        # Overlay text annotation
        ax.text(
            idx,
            med,
            f" Med: {med:.2f}\n IQR: {iqr:.2f}",
            horizontalalignment="left",
            verticalalignment="center",
            fontsize=8,
            fontweight="semibold",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8, edgecolor="gray"),
        )

    plot_title = (
        title
        if title is not None
        else f"Empirical Distribution Comparison: {metric_name}"
    )
    ax.set_title(plot_title, fontsize=12, fontweight="bold", pad=12)
    ax.set_ylabel(metric_name, fontsize=11, labelpad=8)
    ax.grid(True, linestyle="--", alpha=0.5, axis="y")

    plt.xticks(rotation=15, ha="right", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_file, dpi=dpi)
    plt.close(fig)

    return out_file
