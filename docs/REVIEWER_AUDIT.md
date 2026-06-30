# Reviewer-Style Internal Audit for Manuscript v3

Title selected: **Enhanced 5G Handover Prediction over Real-Network RSRP Traces Using Lightweight Temporal Neural Representations**

## Pass 1 - Structural review
Initial score: 7.0/10.
Issues found: title needed stronger contribution language; algorithm descriptions were too short; evaluation metrics were not formally defined.
Fixes applied: title changed; LSTM/GRU/TCN algorithm sections added; classifier rationale added; metric equations added.

## Pass 2 - Methodological review
Intermediate score: 8.1/10.
Issues found: results could be overclaimed because TCN only slightly improves F1 over LSTM; GRU did not outperform baseline and needed honest discussion.
Fixes applied: discussion now states that TCN is competitive/slightly better, not a decisive replacement; GRU is positioned as a lightweight alternative tested for scientific completeness.

## Pass 3 - Reviewer risk review
Final internal score: 8.6/10 for a conference-style or short journal submission draft.
Remaining risks: sample size is limited; raw drive-test reprocessing is not fully repeated; temporal deployment split is not yet included; end-to-end handover classification is future work.
Why the draft is stronger now: claims are proportional to evidence, formulas are included, algorithms are justified, imbalanced evaluation is explained, and limitations are explicit.

Recommendation: submit only after selecting a target venue and adapting formatting, word limits, and author affiliation requirements.
