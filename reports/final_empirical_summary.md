# Final Empirical Summary Report: Energy-Aware MOEA Scheduling Benchmarks

**Lead Authors**: Person B (Mohammad Ayaan / [`moonylit`](https://github.com/moonylit)) & Person A (Harsh Kadam / [`harshkdm31`](https://github.com/harshkdm31))  
**Repository**: [`harshkdm31/energy-scheduling-MOEA`](https://github.com/harshkdm31/energy-scheduling-MOEA)  
**Deliverable**: Final Research Synthesis & Empirical Visualizations (Phase 6 / Gate 3)  
**Date**: 2026-10-28  

---

## Executive Summary
This report synthesizes the empirical performance, statistical significance, and architectural trade-offs of the Multi-Objective Evolutionary Algorithm (MOEA) framework for scheduling compute workloads onto heterogeneous computing systems under non-linear energy models.

We evaluated three primary algorithmic variants:
1. **`nsga2_std`**: Standard NSGA-II multi-objective genetic algorithm with uniform random reset mutation.
2. **`nsga2_em`**: NSGA-II optimizer integrated with domain-aware **Energy-Aware Mutation (`EnergyAwareMutation`)**, which stochastically biases reassignment toward more energy-efficient machines.
3. **`weighted_greedy`**: Deterministic weighted LPT-based greedy heuristic sweep across 21 scalarization weights $\lambda \in [0, 1]$.

---

## 1. Experiment Scope & Hyperparameter Freeze (DEC-011, DEC-012)
To avoid data snooping and ensure unbiased evaluation, the experimental protocol enforces strict parameter locking:
- **Evaluation Budget Contract**: Locked at $B(n) = 200n$ function evaluations across all stochastic search algorithms ($B(200) = 40,000$, $B(50) = 10,000$, $B(500) = 100,000$).
- **Population Size**: Fixed at $N_{\text{pop}} = 100$.
- **Seed Isolation Contract**:
  - **Development Seeds** (`[101, 102, 103]`): Strictly reserved for pilot validation, hyperparameter tuning, and pipeline sanity checking.
  - **Final Test Seeds** (`[201, 202, ..., 230]`): 30 independent test seeds reserved exclusively for final evaluation and statistical hypothesis testing.
- **Power Model**:
  $$P_j = \kappa s_j^\alpha \epsilon_j, \quad \kappa = 1.0, \quad \epsilon_j \sim \text{Uniform}(0.8, 1.2)$$
  evaluated across sub-linear/linear ($\alpha = 1.0$), quadratic ($\alpha = 2.0$), and cubic ($\alpha = 3.0$) CMOS scaling regimes.

---

## 2. Invariant Verification & Evaluator Bounding Rigor (DEC-010)
All schedules evaluated during both pilot and batch executions were checked against analytical lower bounds:
$$LB_C = \max\left(\frac{\sum_{i=1}^n w_i}{\sum_{j=1}^m s_j}, \frac{\max_i w_i}{\max_j s_j}\right)$$
$$LB_E = \sum_{i=1}^n w_i \times \min_{j \in [m]} \left(\frac{P_j}{s_j}\right)$$

### Verification Guarantees
1. **Zero Violations**: Over $100,000+$ candidate schedule evaluations across all runs, zero lower bound violations ($f_1 < LB_C$ or $f_2 < LB_E$) were observed.
2. **Exhaustive Combinatorial Attainment**: Brute-force evaluation of all $3^8 = 6,561$ schedules rigorously proved that the minimum possible energy consumption matches $LB_E$ within floating-point tolerance ($|\min f_2 - LB_E| < 1.14 \times 10^{-13}$).
3. **True Pareto Optimality of Extreme Anchor**: The deterministic energy-greedy schedule ($\lambda = 0$) strictly belongs to the true Pareto-optimal set $\mathcal{P}^*$.

---

## 3. Empirical Ablation & Statistical Hypothesis Testing (DEC-014)

### Statistical Methodology
- **Performance Metric**: 2D Hypervolume indicator ($HV$) evaluated with analytical reference point $r = (1.1 \times \max f_1, 1.1 \times \max f_2)$.
- **Hypothesis Family**: Paired two-tailed Wilcoxon signed-rank tests across matching seeds comparing `nsga2_em` against `nsga2_std` and `weighted_greedy`.
- **Multiple Comparisons**: Holm-Bonferroni step-down correction at family-wise error rate $\alpha = 0.05$.
- **Effect Size & Dispersion**: Cliff's Delta non-parametric effect size $d \in [-1.0, 1.0]$ and Median (IQR).

### Hypothesis Testing Results Table
| Metric / Comparison | alpha | Test | n | Median A (IQR) | Median B (IQR) | Wilcoxon W | p-value | Holm-Adj p | Rank-biserial | Significant? |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| HV: nsga2_em vs nsga2_std | 1.0 | Paired Wilcoxon signed-rank | 30 | 5074464.82 (57462.61) | 4763886.81 (76488.80) | 0.0 | 1.8626e-09 | 1.6764e-08 | +1.000 | **YES** |
| HV: nsga2_em vs random | 1.0 | Paired Wilcoxon signed-rank | 30 | 5074464.82 (57462.61) | 2624251.38 (46276.23) | 0.0 | 1.8626e-09 | 1.6764e-08 | +1.000 | **YES** |
| HV: nsga2_em vs weighted_greedy | 1.0 | One-sample Wilcoxon signed-rank | 30 | 5074464.82 (57462.61) | 4195607.47 (0.00) | 0.0 | 1.8626e-09 | 1.6764e-08 | +1.000 | **YES** |
| HV: nsga2_em vs nsga2_std | 2.0 | Paired Wilcoxon signed-rank | 30 | 468035449.37 (3832008.11) | 435608878.29 (7928286.85) | 0.0 | 1.8626e-09 | 1.6764e-08 | +1.000 | **YES** |
| HV: nsga2_em vs random | 2.0 | Paired Wilcoxon signed-rank | 30 | 468035449.37 (3832008.11) | 336657310.82 (4172902.75) | 0.0 | 1.8626e-09 | 1.6764e-08 | +1.000 | **YES** |
| HV: nsga2_em vs weighted_greedy | 2.0 | One-sample Wilcoxon signed-rank | 30 | 468035449.37 (3832008.11) | 127589408.35 (0.00) | 0.0 | 1.8626e-09 | 1.6764e-08 | +1.000 | **YES** |
| HV: nsga2_em vs nsga2_std | 3.0 | Paired Wilcoxon signed-rank | 30 | 4151763996.09 (24993464.71) | 4025297860.93 (51362602.80) | 0.0 | 1.8626e-09 | 1.6764e-08 | +1.000 | **YES** |
| HV: nsga2_em vs random | 3.0 | Paired Wilcoxon signed-rank | 30 | 4151763996.09 (24993464.71) | 3212191677.10 (49495646.53) | 0.0 | 1.8626e-09 | 1.6764e-08 | +1.000 | **YES** |
| HV: nsga2_em vs weighted_greedy | 3.0 | One-sample Wilcoxon signed-rank | 30 | 4151763996.09 (24993464.71) | 3420428505.33 (0.00) | 0.0 | 1.8626e-09 | 1.6764e-08 | +1.000 | **YES** |

---

## 4. Visualizations: Pareto Frontiers Across Power Regimes

### Linear Regime ($\alpha = 1.0$)
![Pareto Frontier Comparison (alpha=1.0)](figures/pareto_comparison_alpha1.0.png)

### Quadratic Regime ($\alpha = 2.0$)
![Pareto Frontier Comparison (alpha=2.0)](figures/pareto_comparison_alpha2.0.png)

### Cubic Regime ($\alpha = 3.0$)
![Pareto Frontier Comparison (alpha=3.0)](figures/pareto_comparison_alpha3.0.png)

---

## 5. Algorithmic Takeaways & Power Scaling Dynamics

### 1. Behavior Under Linear Scaling ($\alpha = 1.0$)
When $\alpha = 1.0$, machine active power scales proportionally with processing speed ($P_j \propto s_j$). Consequently, the unit energy cost $e_j = P_j / s_j$ varies only according to hardware noise $\epsilon_j \sim [0.8, 1.2]$, yielding low energy heterogeneity ($\rho_e \approx 1.35$). Under this mild trade-off, standard genetic exploration (`nsga2_std`) and Energy-Aware Mutation (`nsga2_em`) perform similarly.

### 2. Superiority Under Super-Linear Scaling ($\alpha = 2.0, 3.0$)
As CMOS scaling shifts to quadratic ($\alpha = 2.0$) and cubic ($\alpha = 3.0$) regimes, energy heterogeneity explodes:
$$\rho_e = \frac{\max_j (P_j / s_j)}{\min_j (P_j / s_j)} \approx \rho_s^{\alpha - 1}$$
Reassigning a workload from a slow, highly efficient machine to a fast, power-hungry machine incurs massive energy penalties.

Under these conditions:
- **`nsga2_std`** suffers from random mutation drift, frequently assigning tasks to high-power machines and degrading energy convergence.
- **`nsga2_em`** actively samples from $\mathcal{M}_{\text{better}}(i) = \{k \mid e_k < e_{x_i}\}$, exerting targeted evolutionary pressure toward lower-energy allocations while preserving makespan scheduling efficiency.
- **Result**: `nsga2_em` achieves consistently lower minimum energy frontiers and superior Hypervolume coverage compared to standard NSGA-II.

---

## 6. Artifact Inventory & Reproducibility
- **Configurations**: [`experiments/configs/stage1_core.json`](../experiments/configs/stage1_core.json), [`experiments/configs/stage2_scale.json`](../experiments/configs/stage2_scale.json)
- **Batch Orchestrator**: [`scripts/run_stage1_batch.py`](../scripts/run_stage1_batch.py)
- **Statistical Engine**: [`scripts/analyze_hypotheses.py`](../scripts/analyze_hypotheses.py)
- **Visualizer Engine**: [`src/visualizer.py`](../src/visualizer.py)
- **Decision Registry**: [`decisions.md`](../decisions.md) (DEC-001 through DEC-016)
- **Experiment Log**: [`experiment_log.md`](../experiment_log.md) (RUN-001 through RUN-008)
