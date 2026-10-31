# Stage 2 Hypothesis Testing Results

## Statistical Methodology
- **Metric**: 2D Hypervolume indicator ($HV$), evaluated using a common reference point for all variants and runs of each instance: $r = (1.1 \times \max f_1, 1.1 \times \max f_2)$.
- **EM vs Standard and EM vs Random**: Paired two-tailed Wilcoxon signed-rank tests using matching stochastic run seeds.
- **EM vs Weighted Greedy**: Two-tailed one-sample Wilcoxon signed-rank test on EM run differences from the single deterministic Weighted Greedy baseline.
- **Effect Size**: Matched-pairs rank-biserial correlation. Positive values indicate higher HV for NSGA-II EM.
- **Multiple Comparisons**: Holm-Bonferroni step-down correction across all hypotheses included in this analysis, at family-wise error rate $\alpha = 0.05$.
- **Summary**: Median and interquartile range (IQR).
- **Experimental unit**: Stochastic algorithm runs on a fixed problem instance. Run seeds are not independent problem instances.

## Summary Results Table

| Instance (tasks ? machines) | Metric / Comparison | alpha | Test | n | Median A (IQR) | Median B (IQR) | Wilcoxon W | p-value | Holm-Adj p | Rank-biserial | Significant? |
|---:|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| 500 x 16 | HV: nsga2_em vs nsga2_std | 2.0 | Paired Wilcoxon signed-rank | 30 | 2130987217.00 (25858765.00) | 1965408887.24 (38038328.27) | 0.0 | 1.8626e-09 | 1.1176e-08 | +1.000 | **YES** |
| 500 x 16 | HV: nsga2_em vs random | 2.0 | Paired Wilcoxon signed-rank | 30 | 2130987217.00 (25858765.00) | 1345453152.53 (22425512.64) | 0.0 | 1.8626e-09 | 1.1176e-08 | +1.000 | **YES** |
| 500 x 16 | HV: nsga2_em vs weighted_greedy | 2.0 | One-sample Wilcoxon signed-rank | 30 | 2130987217.00 (25858765.00) | 499612271.94 (0.00) | 0.0 | 1.8626e-09 | 1.1176e-08 | +1.000 | **YES** |
| 50 x 4 | HV: nsga2_em vs nsga2_std | 2.0 | Paired Wilcoxon signed-rank | 30 | 25681266.02 (110447.89) | 25004741.58 (523671.32) | 0.0 | 1.8626e-09 | 1.1176e-08 | +1.000 | **YES** |
| 50 x 4 | HV: nsga2_em vs random | 2.0 | Paired Wilcoxon signed-rank | 30 | 25681266.02 (110447.89) | 20696199.87 (491032.38) | 0.0 | 1.8626e-09 | 1.1176e-08 | +1.000 | **YES** |
| 50 x 4 | HV: nsga2_em vs weighted_greedy | 2.0 | One-sample Wilcoxon signed-rank | 30 | 25681266.02 (110447.89) | 12050432.42 (0.00) | 0.0 | 1.8626e-09 | 1.1176e-08 | +1.000 | **YES** |
