# FeatureSelection-XAI-SPOPlus-MSE

Seminar paper code (TU Dresden, WS 2026/27): How does the training loss, prediction loss (MSE) vs. decision loss (SPO+), change which features an XAI method reports as important in predict-then-optimize?

The pre-study trains a linear model with MSE and with SPO+ and compares permutation feature importance (ΔMSE, ΔMSE_adj, Δ normalized regret) on two problems:

- **Shortest path** on a k×k grid (LP, minimization)
- **Knapsack**, 2-dimensional 0-1, 32 items, fixed weights, uncertain item values (binary IP, maximization)

Each problem is run with the PyEPO standard generator and with an own generator with planted effects (critical, distractor and noise features).

## Structure

```
src/prestudy.py                 generators, training, PFI, plot/analysis helpers (problem as parameter)
notebooks/prestudy/
  shortest_path_no_planted_effects.ipynb
  shortest_path_including_planted_effects.ipynb
  knapsack_no_planted_effects.ipynb
  knapsack_including_planted_effects.ipynb
results/prestudy/               CSV outputs of the notebooks (same names)
docs/prestudy/                  analysis notes
```

## Setup

Requires [uv](https://docs.astral.sh/uv/) and a Gurobi license (the pip `gurobipy` ships a size-limited license, which is enough for these problem sizes).

```bash
uv sync
uv run jupyter lab
```

The notebooks load the CSVs from `results/prestudy/` if present; delete a CSV to rerun the experiment. `prestudy.run(..., n_jobs=...)` runs the models in parallel processes (default: all cores but two); results do not depend on `n_jobs`. Rough runtimes with 12 processes: shortest path a few minutes per notebook, knapsack 15–20 min per notebook.

## License

MIT, see [LICENSE](LICENSE).
