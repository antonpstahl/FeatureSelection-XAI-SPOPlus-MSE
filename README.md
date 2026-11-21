# FeatureSelection-XAI-SPOPlus-MSE

Seminar paper code (TU Dresden, WS 2026/27): How does the training loss, prediction loss (MSE) vs. decision loss (SPO+), change which features an XAI method reports as important in predict-then-optimize?

The pre-study trains a linear model with MSE and with SPO+ on shortest-path data in a k×k grid and compares permutation feature importance (ΔMSE, ΔMSE_adj, Δ normalized regret).

## Structure

```
src/prestudy.py        data generators (PyEPO, planted), training, PFI, plot/analysis helpers
notebooks/
  prestudy_no_planted_effects.ipynb         PyEPO standard generator, 5×5 grid
  prestudy_including_planted_effects.ipynb  own generator with planted effects, 5×5 and 10×10
results/               CSV outputs of the notebooks
docs/                  analysis notes
```

## Setup

Requires [uv](https://docs.astral.sh/uv/) and a Gurobi license (the pip `gurobipy` ships a size-limited license, which is enough for these grid sizes).

```bash
uv sync
uv run jupyter lab
```

The notebooks load the CSVs from `results/` if present; delete a CSV to rerun the experiment (no planted effects ≈ 10 min, planted effects ≈ 40 min).

## License

MIT, see [LICENSE](LICENSE).
