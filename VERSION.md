# Reproducibility package aligned with the Telecommunication Systems submission

This repository refresh is based directly on the archived **v11 confirmatory natural-test repository**.

The experimental source files and archived result tables are preserved from that package. The refresh updates documentation and metadata so that the repository matches the current manuscript title and accurately documents the paths expected by the existing code.

The confirmatory D3 package includes:

- session-level train/validation/test separation before sampling;
- train-only negative sampling;
- natural-prevalence validation and untouched test evaluation;
- validation-based threshold and candidate-pipeline selection;
- event-level metrics;
- average precision, ROC-AUC, MCC, specificity, Brier score, and false-positive counts;
- five repeated seeds: `1, 7, 21, 42, 100`.
