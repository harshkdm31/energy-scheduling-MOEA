"""
Instance generator and ProblemInstance data contract for multi-objective energy-aware scheduling.
"""

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import scipy.stats


@dataclass
class ProblemInstance:
    """
    Data contract representing a problem instance for energy-aware task scheduling.

    Attributes:
        instance_id: Unique identifier for the instance.
        n_tasks: Number of tasks (n).
        m_machines: Number of heterogeneous machines (m).
        alpha: Power model exponent parameter.
        kappa: Power model scaling constant (default: 1.0).
        seed: Random seed used to generate this instance.
        w: Task workload vector of length n, where w_i > 0.
        s: Machine processing speed vector of length m, where s_j > 0.
        p: Machine active power rating vector of length m, where p_j > 0.
        rho_s: Speed heterogeneity ratio (s_max / s_min >= 1.0).
        rho_e: Energy efficiency heterogeneity ratio (e_max / e_min >= 1.0),
               where e_j = p_j / s_j.
        pearson_s_e: Pearson correlation r(s, e) between machine speeds and energy costs.
    """

    instance_id: str
    n_tasks: int
    m_machines: int
    alpha: float
    kappa: float
    seed: int
    w: List[float]
    s: List[float]
    p: List[float]
    rho_s: float
    rho_e: float
    pearson_s_e: float

    @property
    def e(self) -> List[float]:
        """Energy cost per unit of workload on each machine: e_j = p_j / s_j."""
        return [p_j / s_j for p_j, s_j in zip(self.p, self.s)]

    @property
    def lower_bound_energy(self) -> float:
        """Analytical lower bound on energy: LB_E = sum(w) * min(e)."""
        return sum(self.w) * min(self.e)

    @property
    def lower_bound_makespan(self) -> float:
        """
        Analytical lower bound on makespan:
        LB_C = max(sum(w) / sum(s), max(w) / max(s)).
        """
        return max(sum(self.w) / sum(self.s), max(self.w) / max(self.s))

    def to_dict(self) -> Dict[str, Any]:
        """Convert instance to dictionary format."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProblemInstance":
        """Instantiate ProblemInstance from a dictionary."""
        return cls(
            instance_id=str(data["instance_id"]),
            n_tasks=int(data["n_tasks"]),
            m_machines=int(data["m_machines"]),
            alpha=float(data["alpha"]),
            kappa=float(data["kappa"]),
            seed=int(data["seed"]),
            w=[float(x) for x in data["w"]],
            s=[float(x) for x in data["s"]],
            p=[float(x) for x in data["p"]],
            rho_s=float(data["rho_s"]),
            rho_e=float(data["rho_e"]),
            pearson_s_e=float(data["pearson_s_e"]),
        )

    def to_json(self, indent: int = 2) -> str:
        """Serialize instance to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_json(cls, json_str: str) -> "ProblemInstance":
        """Deserialize ProblemInstance from JSON string."""
        return cls.from_dict(json.loads(json_str))

    def save(self, directory: Union[str, Path] = "data/instances") -> Path:
        """Save this instance as a JSON file in the specified directory."""
        return save(self, directory)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "ProblemInstance":
        """Load a ProblemInstance from a JSON file."""
        return load_instance(filepath)


class InstanceGenerator:
    """
    Generator for multi-objective energy-aware scheduling problem instances.

    Sampling specification:
    - Workloads: w_i ~ Uniform(1.0, 100.0)
    - Speeds: s_j ~ LogUniform(1.0, 10.0)
    - Power ratings: P_j = kappa * (s_j ** alpha) * epsilon_j
      where kappa = 1.0 and epsilon_j ~ Uniform(0.8, 1.2)
    - Descriptors:
      * rho_s = s_max / s_min
      * rho_e = e_max / e_min, where e_j = P_j / s_j
      * pearson_s_e = Pearson correlation r(s, e)
    """

    def __init__(self, default_seed: Optional[int] = None):
        self.default_seed = default_seed

    def generate(
        self,
        n_tasks: int,
        m_machines: int,
        alpha: float = 2.0,
        kappa: float = 1.0,
        seed: Optional[int] = None,
        instance_id: Optional[str] = None,
    ) -> ProblemInstance:
        """
        Generate a synthetic problem instance with rigorous parameter validation
        and cleanly calculated descriptor metrics.
        """
        if n_tasks <= 0:
            raise ValueError(f"n_tasks must be positive, got {n_tasks}")
        if m_machines <= 0:
            raise ValueError(f"m_machines must be positive, got {m_machines}")
        if kappa <= 0:
            raise ValueError(f"kappa must be positive, got {kappa}")

        resolved_seed = seed if seed is not None else self.default_seed
        if resolved_seed is None:
            resolved_seed = int(np.random.randint(0, 2**31 - 1))

        rng = np.random.default_rng(resolved_seed)

        # 1. Sample task workloads: w_i ~ Uniform(1.0, 100.0)
        w_arr = rng.uniform(1.0, 100.0, size=n_tasks)

        # 2. Sample machine speeds: s_j ~ LogUniform(1.0, 10.0)
        log_s = rng.uniform(np.log(1.0), np.log(10.0), size=m_machines)
        s_arr = np.exp(log_s)

        # 3. Compute active powers: P_j = kappa * s_j^alpha * epsilon_j
        #    where epsilon_j ~ Uniform(0.8, 1.2)
        epsilon = rng.uniform(0.8, 1.2, size=m_machines)
        p_arr = kappa * (s_arr ** alpha) * epsilon

        # 4. Energy per unit workload: e_j = P_j / s_j
        e_arr = p_arr / s_arr

        # 5. Descriptors:
        # rho_s = s_max / s_min
        rho_s = float(np.max(s_arr) / np.min(s_arr))

        # rho_e = e_max / e_min
        rho_e = float(np.max(e_arr) / np.min(e_arr))

        # Pearson correlation r(s, e)
        if m_machines >= 2 and np.std(s_arr) > 1e-12 and np.std(e_arr) > 1e-12:
            r_val, _ = scipy.stats.pearsonr(s_arr, e_arr)
            pearson_s_e = float(r_val)
        else:
            pearson_s_e = 0.0

        if instance_id is None:
            formatted_alpha = f"{alpha:g}"
            instance_id = f"inst_n{n_tasks}_m{m_machines}_a{formatted_alpha}_s{resolved_seed}"

        return ProblemInstance(
            instance_id=instance_id,
            n_tasks=n_tasks,
            m_machines=m_machines,
            alpha=float(alpha),
            kappa=float(kappa),
            seed=resolved_seed,
            w=[float(x) for x in w_arr],
            s=[float(x) for x in s_arr],
            p=[float(x) for x in p_arr],
            rho_s=rho_s,
            rho_e=rho_e,
            pearson_s_e=pearson_s_e,
        )

    @staticmethod
    def save(instance: ProblemInstance, directory: Union[str, Path] = "data/instances") -> Path:
        """Dump formatted JSON instance to directory."""
        return save(instance, directory)

    @staticmethod
    def load(filepath: Union[str, Path]) -> ProblemInstance:
        """Load instance from JSON file."""
        return load_instance(filepath)


def save(instance: ProblemInstance, directory: Union[str, Path] = "data/instances") -> Path:
    """
    Dump formatted JSON instance to directory.
    Default directory: data/instances/
    """
    dir_path = Path(directory)
    dir_path.mkdir(parents=True, exist_ok=True)
    file_path = dir_path / f"{instance.instance_id}.json"
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(instance.to_dict(), f, indent=2)
    return file_path


def load_instance(filepath: Union[str, Path]) -> ProblemInstance:
    """Load a ProblemInstance from a JSON file."""
    path = Path(filepath)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return ProblemInstance.from_dict(data)
