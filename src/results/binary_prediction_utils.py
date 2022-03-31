"""
Utility function file for binary prediction tasks.

Protected attributes (_X) take different inputs, main functions (Y) require inputs (y_true, y_score).
"""


import numpy as np


def _get_cm_values(has_label_true: np.ndarray, has_label_pred: np.ndarray) -> dict:
    """
    Get dictionary of TP, FN, FP, TN values given a binary label indicator and a vector of binary predictions.

    Params:
    - has_label_true: np.ndarray of shape (N, ) with {0, 1} entries.
    - has_label_pred: np.ndarray of shape (N, ) with {0, 1} entries.

    Returns:
    - Dictionary of True/False Positives/Negatives
    """

    # Both equal to 1 for true positives
    tp = ((has_label_true == 1) & (has_label_pred == 1)).sum()

    # True=1, False=0 for false negatives
    fn = ((has_label_true == 1) & (has_label_pred == 0)).sum()

    # True=0, False=1 for false positives
    fp = ((has_label_true == 0) & (has_label_pred == 1)).sum()

    # True=0, False=0 for true negatives
    tn = ((has_label_true == 0) & (has_label_pred == 0)).sum()

    # Return dictionary
    return {"tp": tp, "fn": fn, "fp": fp, "tn": tn}


def _custom_cm_over_threshold(y_true: np.ndarray, y_score: np.ndarray, num: int = 1e-6) -> dict:
    """
    Compute True/False Positive/Negatives of multi-class predictions y_true, y_score with a commonly varying threshold.

    Params:
    - y_true: np.ndarray of shape (N, num_outcs) with one-hot encoded true label encodings.
    - y_score: np.ndarray of shape (N, num_outcs) with predicted probability outcome assignments.
    - num: steps to consider a sliding scale over

    Returns:
        Tuple (threshold, TP, FN, FP, TN) of T/F P/N values for a common threshold list.
    """

    # Compute varying thresholds
    _min, _max = np.min(y_score), np.max(y_score)
    thresholds = np.linspace(start=_min, stop=_max, num=num, endpoint=True)

    # Initialise output variables
    shape = (num, y_true.shape[-1])
    tp, fn, fp, tn = np.zeros(shape), np.zeros(shape), np.zeros(shape), np.zeros(shape)

    # Iterate over thresholds
    for thresh_id, eps in enumerate(thresholds):

        # Convert scores to binary
        y_pred_thresh = (y_score >= eps).astype(int)

        # Iterate over outcomes
        for outc_id in range(y_true.shape[-1]):
            # Get conf matrix values
            conf_matrix = _get_cm_values(y_true[:, outc_id], y_pred_thresh[:, outc_id])

            # Update scores
            tp[thresh_id, outc_id] = conf_matrix["tp"]
            fn[thresh_id, outc_id] = conf_matrix["fn"]
            fp[thresh_id, outc_id] = conf_matrix["fp"]
            tn[thresh_id, outc_id] = conf_matrix["tn"]

    return {"thresholds": thresholds, "tp": tp, "fn": fn, "fp": fp, "tn": tn}


def custom_auc(y_true: np.ndarray, y_score: np.ndarray) -> np.ndarray:
    """
    Compute Custom Area-under-the-receiver-curve on multi-class setting. The score is computed as regular AUROC, except
    that the thresholds per each class are not allowed to vary independently.

    For each threshold, labels with value higher than epsilon are flagged up, consequently AUROC per each class is
    computed. Area under the curve is consequently computed as regular.

    Params:
    - y_true: np.ndarray of shape (N, num_outcs) with one-hot encoded true label encodings.
    - y_score: np.ndarray of shape (N, num_outcs) with predicted probability outcome assignments.

    Returns:
        area: np.array of dimension (y_score.shape[-1], ) with corresponding area estimations.
    """
    metrics_dic = _custom_cm_over_threshold(y_true=y_true, y_score=y_score)
    tp, fn, fp, tn = metrics_dic["tp"], metrics_dic["fn"], metrics_dic["fp"], metrics_dic["tn"]

    # Compute Sensitivity and Specificity
    tpr = tp / (tp + fn)
    fpr = fp / (fp + tn)

    # Ensure fpr always non-increasing
    assert np.all(np.diff(fpr[1:, :], axis=0) <= 0, axis=None)

    # Compute area under curve with trapezoidal rule - multiply by -1 as fpr is decreasing
    area = - np.trapz(y=tpr, x=fpr, axis=0)

    return area


def custom_prc(y_true: np.ndarray, y_score: np.ndarray) -> np.ndarray:
    """
    Compute custom area-under-the-precision-recall curve on multi-class setting. The score is computed as regular AUPRC,
     except thresholds per each class are not allowed to vary independently, and apply to each class simultaneously.

    Params:
    - y_true: np.ndarray of shape (N, num_outcs) with one-hot encoded true label encodings.
    - y_score: np.ndarray of shape (N, num_outcs) with predicted probability outcome assignments.

    Returns:
        area: np.array of dimension (y_score.shape[-1], ) with corresponding area estimations.

    """
    _, tp, fn, fp, tn = _custom_cm_over_threshold(y_true=y_true, y_score=y_score)

    # Compute Precision and recall
    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=tp + fp == 0)
    recall = tp / (tp + fn)

    # Ensure always decreasing
    assert np.all(np.diff(recall, axis=0) <= 0, axis=None)

    # Compute Area under curve - need to multiply by -1 as recall is decreasing (and not increasing)
    area = - np.trapz(y=precision, x=recall, axis=0)

    return area
