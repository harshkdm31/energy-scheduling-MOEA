"""
Unit tests for ProblemInstance data contract and InstanceGenerator.
"""

import json
from pathlib import Path
import pytest

from src.instance_generator import (
    InstanceGenerator,
    ProblemInstance,
    load_instance,
    save,
)


def test_lengths_match_dimensions():
    """Assert lengths of w, s, and p match n_tasks and m_machines."""
    generator = InstanceGenerator()
    test_cases = [
        (10, 3),
        (50, 5),
        (100, 10),
    ]

    for n_tasks, m_machines in test_cases:
        instance = generator.generate(n_tasks=n_tasks, m_machines=m_machines, seed=42)
        assert len(instance.w) == n_tasks, f"Expected {n_tasks} tasks, got {len(instance.w)}"
        assert len(instance.s) == m_machines, f"Expected {m_machines} speeds, got {len(instance.s)}"
        assert len(instance.p) == m_machines, f"Expected {m_machines} powers, got {len(instance.p)}"
        assert len(instance.e) == m_machines, f"Expected {m_machines} energy costs, got {len(instance.e)}"


def test_strictly_positive_values_and_bounds():
    """Assert all values in w, s, and p are strictly positive and within defined sampling bounds."""
    generator = InstanceGenerator()
    instance = generator.generate(n_tasks=200, m_machines=20, alpha=2.0, kappa=1.0, seed=123)

    # Workloads: w_i in [1.0, 100.0]
    assert all(w_i > 0 for w_i in instance.w), "All workloads must be strictly positive"
    assert all(1.0 <= w_i <= 100.0 for w_i in instance.w), "Workloads should fall in [1.0, 100.0]"

    # Speeds: s_j in [1.0, 10.0]
    assert all(s_j > 0 for s_j in instance.s), "All speeds must be strictly positive"
    assert all(1.0 <= s_j <= 10.0 for s_j in instance.s), "Speeds should fall in [1.0, 10.0]"

    # Powers: p_j > 0
    assert all(p_j > 0 for p_j in instance.p), "All powers must be strictly positive"

    # Energy costs: e_j = p_j / s_j > 0
    assert all(e_j > 0 for e_j in instance.e), "All energy efficiencies must be strictly positive"


def test_descriptor_metrics_heterogeneity_and_correlation():
    """Assert calculated descriptor metrics rho_s >= 1.0 and rho_e >= 1.0, and pearson_s_e in [-1, 1]."""
    generator = InstanceGenerator()

    for seed in [1, 42, 999]:
        instance = generator.generate(n_tasks=40, m_machines=8, alpha=2.0, kappa=1.0, seed=seed)

        # Check rho_s >= 1.0
        assert instance.rho_s >= 1.0, f"rho_s must be >= 1.0, got {instance.rho_s}"
        expected_rho_s = max(instance.s) / min(instance.s)
        assert pytest.approx(instance.rho_s, rel=1e-6) == expected_rho_s

        # Check rho_e >= 1.0
        assert instance.rho_e >= 1.0, f"rho_e must be >= 1.0, got {instance.rho_e}"
        expected_rho_e = max(instance.e) / min(instance.e)
        assert pytest.approx(instance.rho_e, rel=1e-6) == expected_rho_e

        # Check pearson_s_e in [-1.0, 1.0]
        assert -1.0 <= instance.pearson_s_e <= 1.0, (
            f"Pearson correlation must be in [-1, 1], got {instance.pearson_s_e}"
        )


def test_json_serialization_and_deserialization_integrity(tmp_path: Path):
    """Test JSON serialization, file dump, and roundtrip deserialization integrity."""
    generator = InstanceGenerator()
    orig = generator.generate(n_tasks=25, m_machines=5, alpha=3.0, kappa=1.0, seed=777)

    # 1. String JSON roundtrip
    json_str = orig.to_json()
    from_json_inst = ProblemInstance.from_json(json_str)
    assert from_json_inst.instance_id == orig.instance_id
    assert from_json_inst.n_tasks == orig.n_tasks
    assert from_json_inst.m_machines == orig.m_machines
    assert from_json_inst.alpha == orig.alpha
    assert from_json_inst.kappa == orig.kappa
    assert from_json_inst.seed == orig.seed
    assert from_json_inst.w == orig.w
    assert from_json_inst.s == orig.s
    assert from_json_inst.p == orig.p
    assert pytest.approx(from_json_inst.rho_s) == orig.rho_s
    assert pytest.approx(from_json_inst.rho_e) == orig.rho_e
    assert pytest.approx(from_json_inst.pearson_s_e) == orig.pearson_s_e

    # 2. File save and load roundtrip
    saved_path = save(orig, directory=tmp_path)
    assert saved_path.exists()
    assert saved_path.name == f"{orig.instance_id}.json"

    loaded_inst = load_instance(saved_path)
    assert loaded_inst.instance_id == orig.instance_id
    assert loaded_inst.to_dict() == orig.to_dict()

    # 3. Instance method save and class method load
    saved_path2 = orig.save(directory=tmp_path)
    assert saved_path2 == saved_path
    loaded_inst2 = ProblemInstance.load(saved_path2)
    assert loaded_inst2.to_dict() == orig.to_dict()


def test_reproducibility_with_seed():
    """Assert instances generated with the exact same seed are identical."""
    generator = InstanceGenerator()
    inst1 = generator.generate(n_tasks=30, m_machines=6, seed=2026)
    inst2 = generator.generate(n_tasks=30, m_machines=6, seed=2026)
    inst3 = generator.generate(n_tasks=30, m_machines=6, seed=2027)

    assert inst1.w == inst2.w
    assert inst1.s == inst2.s
    assert inst1.p == inst2.p
    assert inst1.to_dict() == inst2.to_dict()

    # Different seeds must produce different workloads
    assert inst1.w != inst3.w


def test_power_model_formulation_alpha_variants():
    """Verify active power formula P_j = kappa * s_j^alpha * epsilon_j across alpha in {1, 2, 3}."""
    generator = InstanceGenerator()
    for alpha in [1.0, 2.0, 3.0]:
        inst = generator.generate(n_tasks=10, m_machines=5, alpha=alpha, kappa=1.0, seed=42)
        assert inst.alpha == alpha
        assert inst.kappa == 1.0
        # Check that epsilon = P_j / (kappa * s_j^alpha) is within [0.8, 1.2]
        for s_j, p_j in zip(inst.s, inst.p):
            eps_j = p_j / (inst.kappa * (s_j ** alpha))
            assert 0.8 <= eps_j <= 1.2, f"Noise epsilon {eps_j} out of [0.8, 1.2] bound"


def test_input_validation():
    """Assert invalid inputs raise ValueError."""
    generator = InstanceGenerator()
    with pytest.raises(ValueError, match="n_tasks must be positive"):
        generator.generate(n_tasks=0, m_machines=5)

    with pytest.raises(ValueError, match="m_machines must be positive"):
        generator.generate(n_tasks=10, m_machines=-1)

    with pytest.raises(ValueError, match="kappa must be positive"):
        generator.generate(n_tasks=10, m_machines=5, kappa=0.0)
