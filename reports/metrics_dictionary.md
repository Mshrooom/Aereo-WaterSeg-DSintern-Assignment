# Metrics dictionary

- **IoU/Jaccard:** overlap divided by union; primary region metric.
- **Dice/F1:** twice overlap divided by total predicted and true area.
- **Precision:** fraction of predicted water pixels that are labelled water.
- **Recall/Sensitivity:** fraction of labelled water pixels recovered.
- **Specificity:** fraction of background correctly rejected.
- **Pixel accuracy:** total correctly classified pixels; can be misleading with class imbalance.
- **Balanced accuracy:** average of recall and specificity.
- **MCC:** correlation-like metric using every confusion-matrix cell.
- **Cohen's kappa:** agreement beyond chance.
- **Boundary F1:** boundary matching within a pixel tolerance.
- **Boundary IoU:** direct overlap of extracted boundaries.
- **HD95:** robust 95th-percentile symmetric boundary distance.
- **ASSD:** average symmetric surface distance.
- **ECE:** difference between predicted probability and empirical water frequency across bins.
- **Brier score:** mean squared probability error.
- **AUROC/AUPRC:** histogram-approximated pixel-ranking metrics.
- **Latency:** measured model-forward time per image on the evaluation hardware.
- **Bootstrap CI:** image-level uncertainty interval around macro IoU, Dice, and boundary F1.
