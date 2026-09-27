# Week 5 — Comprehensive Data Science Project

## Project
End-to-end Iris classification and strategic insight project integrating Weeks 1–4.

## What is included
- `docs/` — final submission report
- `data/` — raw-quality simulation, cleaned dataset, model/statistical outputs
- `visualizations/` — report figures
- `src/` — reproducible Python scripts
- `outputs/` — project summary JSON

## Run
1. Create a virtual environment.
2. Install dependencies:
   `pip install -r requirements.txt`
3. Run:
   `python src/run_pipeline.py`

## Workflow
Data quality → EDA → statistical testing → ML modeling → validation → interpretation → strategic recommendations.

## Dataset
Iris dataset packaged with scikit-learn, originally introduced by Ronald A. Fisher (1936). The report also identifies the UCI Machine Learning Repository as a public reference.

## Reproducibility
Random seed: 42. Train/test split: 80/20, stratified. Cross-validation: 5-fold stratified.
