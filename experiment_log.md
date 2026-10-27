# Experiment Log

| Run ID | Timestamp | Commit | Instance ID | Algorithm Variant | Seed | Budget / Evals | Wall-Clock (s) | Status / Notes |
|---|---|---|---|---|---|---|---|---|
| RUN-001 | 2026-10-27T21:46:00+05:30 | 2e05a37 | N/A (Unit Tests) | Instance Generator Validation Suite | 42, 123, 777, 999, 2026 | 7 Test Functions (pytest) | 13.77s | PASS: Validated dimensions, strict positivity, descriptor metrics (rho_s, rho_e >= 1.0), power scaling across alpha, and JSON roundtrip serialization |
| RUN-002 | 2026-10-27T21:55:00+05:30 | b1b3392 | N/A (Unit Tests) | Deterministic Baselines & Weighted Sweep (MCT, Energy-Greedy, Weighted-Sweep) | 42, 100, 101, 123, 999, 2026 | 7 Test Functions (pytest), 21 weights per sweep | 4.18s | PASS: Validated MCT dimension/alleles, exact LB_E attainment in energy-greedy, Pareto non-domination in weighted-sweep, and deterministic execution |
| RUN-003 | 2026-10-27T22:07:00+05:30 | a99b0a2 | inst_n12_m3_a2_s42 | NSGA-II Optimizer Engine & EnergyAwareMutation Validation | 42, 123 | 4 Test Functions (pytest), 100 evals/mini-run | 4.48s | PASS: Validated EnergyAwareMutation allele validity, strictly lower energy descent, fallback rule, and NSGA-II execution satisfying makespan/energy lower-bound invariants (f1 >= LB_C, f2 >= LB_E) |
