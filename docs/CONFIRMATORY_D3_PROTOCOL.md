# Confirmatory D3 natural-prevalence protocol

This experiment was added after reviewer-style evaluation identified two sources of optimism in the exploratory D3 benchmark:

1. negative sampling was applied before the train-test split; and
2. the best classifier was identified from test F1.

The confirmatory runner addresses those issues as follows:

- session groups are split before sampling;
- negative sampling is restricted to the training partition;
- validation and test retain their natural event prevalence;
- each candidate threshold is selected on validation data;
- the final candidate pipeline is also selected on validation data;
- the untouched test partition is evaluated once;
- test features are materialized in chunks to avoid multi-gigabyte arrays.

## Candidate pipelines

- raw windows + Random Forest;
- temporal engineering + Random Forest;
- concatenated LSTM/GRU/TCN/Transformer features + Random Forest.

## Repeated seeds

`1, 7, 21, 42, 100`

## Main validation-selected result

Across the five untouched natural-prevalence test partitions:

- positive prevalence: 0.0142 +/- 0.0004;
- precision: 0.3988 +/- 0.0354;
- recall: 0.6587 +/- 0.0150;
- F1: 0.4959 +/- 0.0262;
- average precision: 0.4231 +/- 0.0357;
- MCC: 0.5033 +/- 0.0221;
- event-level recall: 0.8811 +/- 0.0160;
- false positives per 1,000 negative windows: 14.46 +/- 2.23.

Raw-window Random Forest was selected on validation in four seeds and temporal-engineering Random Forest in one seed. No candidate pair showed a statistically significant F1 difference with only five paired runs.

## Interpretation

The exploratory sampled D3 benchmark reported F1 near 0.92. The natural-prevalence confirmatory result is near 0.50. This difference is expected and demonstrates why sampling protocol and class prevalence must be reported with performance metrics.
