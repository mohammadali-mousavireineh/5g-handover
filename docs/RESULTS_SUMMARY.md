# Results Summary

The default experiment compares LSTM-, GRU-, and TCN-generated predicted-RSRP feature windows using the same downstream classifier search protocol.

## Best held-out test results

| Feature model | Classifier | Accuracy | Balanced accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| TCN | SVM_RBF | 0.6810 | 0.6540 | 0.3289 | 0.6098 | 0.4274 | 0.6938 |
| LSTM | SVM_RBF | 0.6762 | 0.6510 | 0.3247 | 0.6098 | 0.4237 | 0.7050 |
| GRU | SVM_RBF | 0.6714 | 0.6296 | 0.3108 | 0.5610 | 0.4000 | 0.7004 |

## Interpretation

- TCN achieved the highest F1-score in the default held-out protocol.
- LSTM achieved the highest ROC-AUC but a slightly lower F1-score than TCN.
- Accuracy alone is not sufficient because handover samples are the minority class.
- F1-score, recall, balanced accuracy, ROC-AUC, and confusion matrices should be interpreted together.
