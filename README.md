
# Energy Scheduling MOEA

Research project on multi-objective energy-efficient scheduling of independent tasks on heterogeneous machines.

## Research objectives

We investigate the trade-off between:
- **Makespan:** completion time of the last machine.
- **Modeled active energy:** energy estimated from task workloads, machine speeds, and active power.

NSGA-II is the base optimizer. We evaluate endpoint seeding, bottleneck-aware mutation (BM), and energy-aware mutation (EM) against baseline algorithms.

## Research questions

1. Do BM and EM improve Pareto-front quality beyond standard NSGA-II and endpoint seeding?
2. Do their effects depend on machine speed and energy-efficiency characteristics?
3. When is NSGA-II competitive with a weighted-greedy approach?

## Repository structure

- `src/` — model, objectives, algorithms, and operators
- `tests/` — correctness and regression tests
- `configs/` — experiment settings
- `data/instances/` — generated benchmark instances
- `results/` — local experiment outputs
- `notebooks/` — exploratory analysis
- `docs/literature/` — literature review
- `paper/` — manuscript drafts

## Setup

Use Python 3.11 or another version compatible with the pinned dependencies.

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Research integrity

- Keep development instances separate from final-test instances.
- Save experiment configurations and random seeds.
- Do not claim novelty or superiority until supported by the literature review and results.
- Record decisions and experiments in `DECISIONS.md` and `EXPERIMENT_LOG.md`.
