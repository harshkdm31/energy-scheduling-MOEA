# Stage 3: Generalization and Robustness Experimental Protocol

## 1. Overview

Stage 3 evaluates the generalization and robustness of the proposed energy-aware multi-objective evolutionary algorithm across independently generated heterogeneous scheduling instances.

Unlike Stages 1 and 2, which evaluated algorithms using fixed problem instances, Stage 3 introduces multiple independent instances for every combination of workload scale and machine energy-model parameter.

The primary objective is to determine whether the performance observed in earlier experiments generalizes across different problem instances rather than depending on a particular randomly generated workload.

The experiment compares four scheduling approaches:

- Random baseline
- Standard NSGA-II
- Energy-aware NSGA-II (EM)
- Weighted Greedy baseline

## 2. Research Objectives

The experiment addresses the following objectives:

1. Evaluate algorithm performance across independently generated scheduling instances.
2. Investigate the effect of workload scale on optimization performance.
3. Investigate the effect of the machine energy-model parameter alpha.
4. Compare the proposed energy-aware NSGA-II against standard NSGA-II and baseline approaches.
5. Quantify variability across algorithm seeds and independent instances.
6. Determine whether observed performance differences remain statistically significant after multiple-hypothesis correction.

## 3. Experimental Design

### 3.1 Workload scales

Two workload scales are considered.

| Parameter | Small | Large |
|---|---:|---:|
| Number of tasks | 50 | 500 |
| Number of machines | 4 | 16 |
| Evaluation budget | 10,000 | 100,000 |
| Population size | 100 | 100 |

The evaluation budget is scaled according to the workload size.

### 3.2 Energy-model parameter

Three alpha values are investigated:

- alpha = 1.0
- alpha = 2.0
- alpha = 3.0

The machine power model is:

P_j = kappa * s_j^alpha * epsilon_j

Where:

- P_j represents the modeled power of machine j.
- s_j represents the processing speed of machine j.
- alpha controls the relationship between processing speed and power.
- kappa is the scaling coefficient.
- epsilon_j is a random machine-specific factor sampled from U[0.8, 1.2].

The scaling coefficient is fixed at:

kappa = 1.0

### 3.3 Independent problem instances

Twenty independent problem instances are generated for every scale-alpha combination.

Instance seeds:

1001–1020

The complete experimental design contains:

2 workload scales x 3 alpha values x 20 instances = 120 independent instances.

Each instance is generated deterministically using its corresponding instance seed.

### 3.4 Algorithm seeds

Each independent problem instance is evaluated using 20 algorithm seeds:

201–220

The three stochastic algorithms are:

- Random
- Standard NSGA-II
- Energy-aware NSGA-II

Therefore, the total number of stochastic algorithm executions is:

120 instances x 20 seeds x 3 algorithms = 7,200 runs.

Weighted Greedy is deterministic under a fixed problem instance and configuration. Its result is therefore counted once per instance. Seed-labelled duplicate archives are retained by the existing runner convention and must be verified as identical during analysis.

The experiment consequently contains 120 unique Weighted Greedy results.

## 4. Algorithms Under Evaluation

### 4.1 Random baseline

The random baseline generates scheduling assignments without using the proposed evolutionary optimization strategy.

It provides a reference for evaluating the benefit of structured optimization.

### 4.2 Standard NSGA-II

Standard NSGA-II is used as the evolutionary baseline.

It provides a comparison against the proposed energy-aware modification while retaining the multi-objective optimization framework.

### 4.3 Energy-aware NSGA-II (EM)

The proposed energy-aware NSGA-II incorporates the energy-aware modification being investigated.

Its performance is compared against standard NSGA-II to evaluate the contribution of the proposed approach.

### 4.4 Weighted Greedy

Weighted Greedy is included as a deterministic baseline.

Its results are evaluated once per independent instance rather than treating duplicate seed-labelled archives as independent observations.

## 5. Objective Functions

The scheduling problem considers two minimization objectives: makespan and modeled active energy.

### 5.1 Makespan

The completion time of machine j is:

C_j(X) = sum(w_i / s_j), for all tasks i assigned to machine j.

The makespan is:

f1(X) = max(C_j(X))

### 5.2 Modeled active energy

The modeled active energy objective is:

f2(X) = sum(P_j * C_j(X))

This represents the modeled active energy associated with machine execution.

It is an optimization objective derived from the experimental energy model and is not a direct measurement of physical hardware energy consumption.

## 6. Hypervolume Evaluation

Hypervolume (HV) is used to evaluate the quality of the non-dominated solution archives.

Since both objectives are minimized, a larger hypervolume indicates a better approximation of the objective-space region dominated by the solution archive relative to the reference point.

### 6.1 Common reference point

For each independent instance, one common reference point is calculated using all available algorithm fronts and seeds:

r = (1.1 * max(f1), 1.1 * max(f2))

The same reference point is applied to every algorithm and seed belonging to that instance.

This ensures that algorithms are compared using a consistent reference point within an instance.

### 6.2 Reference-point limitation

The reference point is derived from the observed solution fronts after the experiment.

Consequently, it is data-dependent rather than externally fixed.

This limitation must be considered when interpreting hypervolume comparisons, particularly across different independent instances.

### 6.3 Instance-level aggregation

For each stochastic algorithm:

1. Calculate HV independently for every algorithm seed.
2. Obtain 20 HV observations for the instance.
3. Calculate the median HV across those 20 observations.
4. Use that median as the algorithm's instance-level observation.

Weighted Greedy contributes one HV observation per instance after duplicate archive verification.

## 7. Statistical Analysis

### 7.1 Experimental unit

The independent problem instance is the primary statistical experimental unit.

Algorithm seeds are nested repetitions within each instance. They are not treated as independent problem instances.

Therefore, statistical comparisons are performed using 20 paired instance-level observations for each scale-alpha condition.

### 7.2 Hypotheses

Three comparisons are performed within each scale-alpha condition:

1. Energy-aware NSGA-II versus Standard NSGA-II.
2. Energy-aware NSGA-II versus Random.
3. Energy-aware NSGA-II versus Weighted Greedy.

There are six scale-alpha conditions:

2 scales x 3 alpha values = 6 conditions.

The total number of statistical hypotheses is:

6 conditions x 3 comparisons = 18 hypotheses.

### 7.3 Paired Wilcoxon signed-rank test

A two-tailed paired Wilcoxon signed-rank test is used to compare the instance-level hypervolume observations.

For Stage 3, the test explicitly uses the approximate method with tie-adjusted normal approximation.

The paired observations correspond to the same independently generated problem instances.

The significance level is:

alpha = 0.05

### 7.4 Multiple-hypothesis correction

Holm-Bonferroni correction is applied across all 18 hypothesis tests.

This controls the family-wise error rate when evaluating multiple comparisons.

Both raw p-values and Holm-adjusted p-values are reported.

### 7.5 Effect size

Paired rank-biserial correlation is calculated to quantify the direction and magnitude of paired differences.

A positive value indicates higher hypervolume for the first algorithm in the comparison.

### 7.6 Additional descriptive statistics

The analysis reports:

- Median instance-level hypervolume.
- Interquartile range (IQR).
- Minimum and maximum instance-level hypervolume.
- Number of independent instances.
- Median within-instance seed IQR for stochastic algorithms.
- Number of wins, losses, and ties across paired instances.

Within-instance IQR measures variability across algorithm seeds for a given instance.

It is not applicable to deterministic Weighted Greedy.

## 8. Reproducibility

The experiment uses fixed instance seeds and algorithm seeds.

The Stage 3 configuration is maintained in:

`experiments/configs/stage3_generalization.json`

The resumable execution script is:

`scripts/run_stage3_batch.py`

The statistical analysis script is:

`scripts/analyze_stage3.py`

The configuration defines:

- Workload scales.
- Alpha values.
- Instance seeds.
- Algorithm seeds.
- Algorithm variants.
- Population size.
- Evaluation budget parameters.
- Energy-model coefficient.

The runner validates generated archives and supports resuming completed instances.

An instance is skipped when its expected archives are already present and pass validation.

Incomplete or invalid instance outputs are regenerated.

## 9. Execution Procedure

Run the complete Stage 3 experiment from the repository root.

### Step 1: Validate the configuration and runner

```powershell
python -m py_compile scripts/run_stage3_batch.py
```

### Step 2: Execute a smoke test

```powershell
python scripts/run_stage3_batch.py --smoke-test --output-dir data/raw_runs/stage3_smoke
```

### Step 3: Execute the production experiment

```powershell
python scripts/run_stage3_batch.py
```

The production experiment is computationally expensive. A pilot run should be used to estimate execution time and resource requirements before committing to the full experiment.

The runner supports resuming validated completed instances.

### Step 4: Run the statistical analysis

After all 120 independent instances have been completed and validated:

```powershell
python scripts/analyze_stage3.py
```

The generated report is saved to:

`experiments/stage3_hypothesis_results.md`

The analysis must not be interpreted as complete until all 120 instances and their expected archives have passed validation.

## 10. Expected Experimental Outputs

The experiment is expected to generate:

| Output | Expected count |
|---|---:|
| Independent problem instances | 120 |
| Random algorithm runs | 2,400 |
| Standard NSGA-II runs | 2,400 |
| Energy-aware NSGA-II runs | 2,400 |
| Unique Weighted Greedy results | 120 |
| Stochastic algorithm runs | 7,200 |
| Archive files under existing runner convention | 9,600 |
| Statistical hypotheses | 18 |

Weighted Greedy's 120 unique results are stored using the existing seed-labelled archive convention, producing duplicate archive files for the remaining seed labels.

## 11. Limitations

The following limitations must be considered when interpreting the results:

1. The conclusions apply to the workload generator and machine heterogeneity model used in the experiment.
2. Only two workload scales and three alpha values are investigated.
3. Algorithm seeds represent repeated stochastic executions within an instance and do not increase the number of independent problem instances.
4. The hypervolume reference point is derived from observed fronts and is data-dependent.
5. Energy is modeled using the defined objective function and is not measured directly from physical hardware.
6. Statistical significance does not automatically establish practical importance.
7. Results should not be interpreted as proof of universal superiority across all heterogeneous scheduling environments.

## 12. Conclusion

Stage 3 is designed to provide a more robust evaluation of the proposed energy-aware multi-objective evolutionary algorithm.

By introducing 120 independently generated problem instances, repeated stochastic executions, instance-level statistical comparisons, and multiple-hypothesis correction, the experiment addresses the limitations of evaluating algorithms on a small number of fixed problem instances.

The resulting analysis will provide evidence regarding the generalization of the proposed approach across the tested workload scales and energy-model settings.