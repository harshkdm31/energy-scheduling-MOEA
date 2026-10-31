"""
Integration test suite for Stage 1 batch execution and automated hypothesis testing pipeline.
(Phase 5 / Deliverables B8, B9 / Gate 3).
"""

from pathlib import Path
import re
import subprocess
import sys
import numpy as np
import pytest

from scripts.analyze_hypotheses import compute_hypervolume_2d, run_hypothesis_analysis
from scripts.run_stage1_batch import run_stage1_batch
from src.runner import load_run_result


@pytest.fixture
def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def test_smoke_batch_execution_direct_function(tmp_path: Path):
    """
    Execute run_stage1_batch in smoke-test mode directly and verify output integrity.
    """
    raw_dir = tmp_path / "raw_runs"
    res = run_stage1_batch(
        output_dir=str(raw_dir),
        smoke_test=True,
        n_tasks=20,
        m_machines=4,
        alphas=[1.0, 2.0],
        seeds=[201, 202],
        eval_budget=50,
        pop_size=10,
        variants=["random", "nsga2_std", "nsga2_em", "weighted_greedy"],
    )

    assert res["status"] == "PASS"
    assert res["total_files"] == 16  # 2 alphas * (2 seeds * 4 variants) = 16

    for path_str in res["file_paths"]:
        path = Path(path_str)
        assert path.exists(), f"File {path} does not exist"

        data = load_run_result(path)
        F = np.asarray(data["non_dominated_F"], dtype=np.float64)
        X = np.asarray(data["non_dominated_X"], dtype=np.int64)

        assert F.ndim == 2
        assert F.shape[0] > 0
        assert F.shape[1] == 2
        assert X.ndim == 2
        assert X.shape[0] == F.shape[0]
        assert X.shape[1] == 20
        assert np.all(X >= 0)
        assert np.all(X < 4)


def test_cli_smoke_test_subprocess(repo_root: Path, tmp_path: Path):
    """
    Execute scripts/run_stage1_batch.py via CLI with --smoke-test flag.
    """
    raw_dir = tmp_path / "raw_runs_cli"
    cmd = [
        sys.executable,
        str(repo_root / "scripts" / "run_stage1_batch.py"),
        "--smoke-test",
        "--output-dir",
        str(raw_dir),
        "--alphas",
        "1.0",
        "2.0",
        "--seeds",
        "201",
        "202",
        "--eval-budget",
        "50",
        "--pop-size",
        "10",
    ]

    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(repo_root))
    assert proc.returncode == 0, f"run_stage1_batch.py failed:\n{proc.stderr}"
    assert "Batch execution completed successfully" in proc.stdout

    # Verify archives generated
    npz_files = list(raw_dir.rglob("*.npz"))
    assert len(npz_files) == 16


def test_automated_hypothesis_analysis_pipeline(repo_root: Path, tmp_path: Path):
    """
    Execute run_hypothesis_analysis on smoke-test outputs and verify Markdown table generation.
    """
    raw_dir = tmp_path / "raw_runs"
    report_file = tmp_path / "hypothesis_results.md"

    # 1. Generate smoke test data
    run_stage1_batch(
        output_dir=str(raw_dir),
        smoke_test=True,
        n_tasks=20,
        m_machines=4,
        alphas=[1.0, 2.0, 3.0],
        seeds=[201, 202],
        eval_budget=50,
        pop_size=10,
        variants=["random", "nsga2_std", "nsga2_em", "weighted_greedy"],
    )

    # 2. Run hypothesis analysis
    analysis_res = run_hypothesis_analysis(
        raw_dir=str(raw_dir),
        instances_dir="data/instances",
        output_report=str(report_file),
        n_tasks_filter=20,
        m_machines_filter=4,
        alpha_significance=0.05,
    )

    assert analysis_res["hypotheses_tested"] == 9  # 3 alphas * 3 comparisons
    assert report_file.exists()

    content = report_file.read_text(encoding="utf-8")
    assert "# Stage 1 Hypothesis Testing Results" in content
    assert "| Metric / Comparison | alpha | Test | n | Median A (IQR) | Median B (IQR) | Wilcoxon W | p-value | Holm-Adj p | Rank-biserial | Significant? |" in content

    # Check that each comparison row exists and has numeric entries
    for alpha in [1.0, 2.0, 3.0]:
        assert f"HV: nsga2_em vs nsga2_std | {alpha:.1f}" in content
        assert f"HV: nsga2_em vs weighted_greedy | {alpha:.1f}" in content

    for rec in analysis_res["records"]:
        assert rec["wilcoxon_w"] >= 0.0
        assert 0.0 <= rec["p_value"] <= 1.0
        assert 0.0 <= rec["holm_adj_p"] <= 1.0
        assert -1.0 <= rec["rank_biserial"] <= 1.0
        assert "median" in rec["summary_a"]
        assert "median" in rec["summary_b"]


def test_cli_hypothesis_analysis_subprocess(repo_root: Path, tmp_path: Path):
    """
    Execute scripts/analyze_hypotheses.py via CLI using independently
    generated smoke-test data.
    """
    raw_dir = tmp_path / "raw_runs_cli"
    report_file = tmp_path / "cli_hypothesis_results.md"

    # 1. Generate isolated input data for the CLI subprocess.
    run_stage1_batch(
        output_dir=str(raw_dir),
        smoke_test=True,
        n_tasks=20,
        m_machines=4,
        alphas=[1.0, 2.0, 3.0],
        seeds=[201, 202],
        eval_budget=50,
        pop_size=10,
        variants=["random", "nsga2_std", "nsga2_em", "weighted_greedy"],
    )

    # 2. Execute the analysis script against the generated data.
    cmd = [
        sys.executable,
        str(repo_root / "scripts" / "analyze_hypotheses.py"),
        "--raw-dir",
        str(raw_dir),
        "--instances-dir",
        "data/instances",
        "--output",
        str(report_file),
        "--n-tasks",
        "20",
        "--m-machines",
        "4",
    ]

    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(repo_root),
    )

    assert proc.returncode == 0, (
        f"analyze_hypotheses.py failed:\n"
        f"STDOUT:\n{proc.stdout}\n"
        f"STDERR:\n{proc.stderr}"
    )

    assert report_file.exists()

    content = report_file.read_text(encoding="utf-8")
    assert "# Stage 1 Hypothesis Testing Results" in content
    assert "HV: nsga2_em vs nsga2_std" in content
    assert "HV: nsga2_em vs weighted_greedy" in content

def test_hypervolume_2d_analytical_calculation():
    """
    Verify exact 2D hypervolume calculation with known geometric geometries.
    """
    ref = np.array([10.0, 10.0])

    # Single point (2, 3) -> box (10-2) * (10-3) = 8 * 7 = 56
    p1 = np.array([[2.0, 3.0]])
    assert pytest.approx(compute_hypervolume_2d(p1, ref), 1e-9) == 56.0

    # Two non-dominated points (2, 5) and (5, 2)
    # Area = (5-2)*(10-5) + (10-2)*(10-2) ?
    # Vertical slices: (x2 - x1)*(r1 - y1) + (r0 - x2)*(r1 - y2)
    # = (5 - 2)*(10 - 5) + (10 - 5)*(10 - 2) = 3 * 5 + 5 * 8 = 15 + 40 = 55
    p2 = np.array([[2.0, 5.0], [5.0, 2.0]])
    assert pytest.approx(compute_hypervolume_2d(p2, ref), 1e-9) == 55.0

    # Empty front returns 0.0
    assert compute_hypervolume_2d(np.empty((0, 2)), ref) == 0.0

    # Point outside reference point is excluded
    p_out = np.array([[12.0, 12.0]])
    assert compute_hypervolume_2d(p_out, ref) == 0.0
