import numpy
from sklearn.metrics import (
    roc_auc_score,
    precision_recall_curve,
    auc,
    roc_curve
)

class OODMetricsCalculator:
    """
    A class to compute OOD detection metrics given anomaly scores and ground truth labels.

    Convention:
        - y_true: 1D array-like of ground truth labels, where 0 = in-distribution, 1 = OOD.
        - y_score: 1D array-like of anomaly scores. Higher scores indicate a higher likelihood of being OOD.
    """

    def compute(self, y_true, y_score):
        """
        Compute and return the OOD metrics.

        Args:
            y_true (array-like): True labels (0 for in-distribution, 1 for OOD).
            y_score (array-like): Anomaly scores (higher means more likely OOD).

        Returns:
            dict: A dictionary containing the following metrics:
                - "AUROC": Area under the ROC curve.
                - "AUPR-IN": Area under the precision-recall curve for in-distribution (treating in-distribution as positive).
                - "AUPR-OUT": Area under the precision-recall curve for OOD (treating OOD as positive).
                - "FPR95TPR": False positive rate when the true positive rate is approximately 95%.
        """
        y_true = numpy.asarray(y_true)
        y_score = numpy.asarray(y_score)

        # 1. Compute AUROC (for OOD detection, with OOD as positive (label=1))
        auroc = roc_auc_score(y_true, y_score)

        # 2. Compute AUPR-OUT (with OOD as positive)
        precision_out, recall_out, _ = precision_recall_curve(y_true, (-1)*y_score, pos_label=1)
        aupr_out = auc(recall_out, precision_out)

        # 3. Compute AUPR-IN (with in-distribution as positive)
        # Invert scores and labels: in-distribution becomes positive.
        precision_in, recall_in, _ = precision_recall_curve(1 - y_true, 1 - y_score, pos_label=1)
        aupr_in = auc(recall_in, precision_in)

        # 4. Compute FPR at 95% TPR
        fpr, tpr, _ = roc_curve(y_true, y_score)
        target_tpr = 0.95
        # Find the index where TPR is closest to 95%
        idx = numpy.argmin(numpy.abs(tpr - target_tpr))
        fpr95 = fpr[idx]

        return {
            "AUROC": auroc,
            "AUPR-IN": aupr_in,
            "AUPR-OUT": aupr_out,
            "FPR95TPR": fpr95,
        }
