# Bundled datasets

This folder vendors small strange-attractor trajectories for quick experiments.

- `manifest.json` — registry of bundled datasets
- `strange_attractors/hr_I1.00.csv` — Hindmarsh–Rose trajectory (I=1.00)
- `strange_attractors/hr_I1.20.csv` — Hindmarsh–Rose trajectory (I=1.20)

APIs/UI
- `/datasets` (API) lists available bundles.
- `/datasets/preview` (API) returns a preview of a dataset (first N rows).
- UI → “Datasets” page lets you load the list and preview samples and scatter plots.
