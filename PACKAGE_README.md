# CORA-FL implementation package

This package contains the implementation, configurations, tests, measured run artifacts, traces, derived tables/plot data, and verification scripts corresponding to the final manuscript. The LaTeX manuscript itself is distributed separately.

Useful checks:

```bash
python -m pytest -q
python scripts/verify_confirmatory.py
```

The revised plotting script is in `paper_assets/make_paper_figures.py`. It reads the existing CSV files under `results/derived/plot_data` and changes presentation only.
