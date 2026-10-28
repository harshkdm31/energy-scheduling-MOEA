"""
Unit and integration tests for the empirical visualization suite and report generator.
(Phase 6 / Deliverable B9 / Gate 3).
"""

from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest

import matplotlib
matplotlib.use("Agg")

from scripts.generate_report import generate_empirical_report
from src.visualizer import plot_metric_boxplots, plot_pareto_fronts


@pytest.fixture
def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def test_plot_pareto_fronts_mock_data(tmp_path: Path):
    """
    Test plot_pareto_fronts renders high-DPI PNG with analytical lower bound lines.
    """
    save_file = tmp_path / "test_pareto.png"

    archive_dict = {
        "nsga2_std": np.array([[120.0, 9500.0], [130.0, 9100.0], [150.0, 8900.0]]),
        "nsga2_em": np.array([[118.0, 9200.0], [125.0, 8800.0], [145.0, 8600.0]]),
        "weighted_greedy": np.array([[115.0, 9800.0], [135.0, 8700.0], [160.0, 8500.0]]),
    }

    mock_instance = {
        "instance_id": "mock_n50_m4_a2_s42",
        "lower_bound_makespan": 110.0,
        "lower_bound_energy": 8500.0,
        "alpha": 2.0,
    }

    out_path = plot_pareto_fronts(
        archive_dict=archive_dict,
        instance=mock_instance,
        save_path=save_file,
        dpi=150,
    )

    assert out_path.exists()
    assert out_path.stat().st_size > 5000  # Valid non-empty PNG file


def test_plot_metric_boxplots_mock_data(tmp_path: Path):
    """
    Test plot_metric_boxplots renders distribution boxplots with Median + IQR annotations.
    """
    save_file = tmp_path / "test_boxplots.png"

    rng = np.random.default_rng(42)
    metrics_by_variant = {
        "nsga2_std": rng.normal(loc=100.0, scale=10.0, size=30).tolist(),
        "nsga2_em": rng.normal(loc=115.0, scale=8.0, size=30).tolist(),
        "weighted_greedy": rng.normal(loc=90.0, scale=12.0, size=30).tolist(),
    }

    out_path = plot_metric_boxplots(
        metrics_by_variant=metrics_by_variant,
        metric_name="Hypervolume (HV)",
        save_path=save_file,
        dpi=150,
    )

    assert out_path.exists()
    assert out_path.stat().st_size > 5000


def test_visualizer_empty_and_single_point_graceful_handling(tmp_path: Path):
    """
    Test visualizer handles empty or single-point fronts gracefully without crashing.
    """
    save_file = tmp_path / "test_edge.png"

    archive_dict = {
        "nsga2_std": np.empty((0, 2)),
        "nsga2_em": np.array([[150.0, 5000.0]]),
    }

    mock_instance = {
        "instance_id": "edge_instance",
        "lower_bound_makespan": 100.0,
        "lower_bound_energy": 4500.0,
        "alpha": 1.0,
    }

    out_path = plot_pareto_fronts(
        archive_dict=archive_dict,
        instance=mock_instance,
        save_path=save_file,
        dpi=100,
    )

    assert out_path.exists()
    assert out_path.stat().st_size > 2000


def test_generate_empirical_report_execution(tmp_path: Path, repo_root: Path):
    """
    Test end-to-end report synthesis pipeline.
    """
    rep_file = tmp_path / "summary_report.md"
    fig_dir = tmp_path / "figures"

    res_path = generate_empirical_report(
        raw_dir="data/raw_runs",
        instances_dir="data/instances",
        hypothesis_results_path="experiments/hypothesis_results.md",
        figures_dir=str(fig_dir),
        report_output_path=str(rep_file),
    )

    assert res_path.exists()
    content = res_path.read_text(encoding="utf-8")
    assert "# Final Empirical Summary Report" in content
    assert "Experiment Scope & Hyperparameter Freeze" in content
    assert "Invariant Verification & Evaluator Bounding Rigor" in content
    assert "Algorithmic Takeaways & Power Scaling Dynamics" in content


def test_cli_generate_report_subprocess(repo_root: Path):
    """
    Test scripts/generate_report.py via CLI invocation.
    """
    cmd = [sys.executable, str(repo_root / "scripts" / "generate_report.py")]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(repo_root))
    assert proc.returncode == 0, f"generate_report.py failed:\n{proc.stderr}"
    assert "Final empirical summary report compiled to" in proc.stdout
