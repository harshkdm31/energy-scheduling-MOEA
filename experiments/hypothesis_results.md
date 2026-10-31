# Stage 1 Hypothesis Testing Results

## Statistical Methodology
- **Metric**: 2D Hypervolume indicator ($HV$), evaluated using a common reference point for all variants and runs of each instance: $r = (1.1 \times \max f_1, 1.1 \times \max f_2)$.
- **EM vs Standard and EM vs Random**: Paired two-tailed Wilcoxon signed-rank tests using matching stochastic run seeds.
- **EM vs Weighted Greedy**: Two-tailed one-sample Wilcoxon signed-rank test on EM run differences from the single deterministic Weighted Greedy baseline.
- **Effect Size**: Matched-pairs rank-biserial correlation. Positive values indicate higher HV for NSGA-II EM.
- **Multiple Comparisons**: Holm-Bonferroni step-down correction across all hypotheses included in this analysis, at family-wise error rate $\alpha = 0.05$.
- **Summary**: Median and interquartile range (IQR).
- **Experimental unit**: Stochastic algorithm runs on a fixed problem instance. Run seeds are not independent problem instances.

## Summary Results Table

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
