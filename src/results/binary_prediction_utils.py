"""
Utility function file for binary prediction tasks.

Protected attributes (_X) have variable input, main functions (Y) have inputs (y_true, y_score).
"""


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap


def _get_single_cm_values(has_label_true: np.ndarray, has_label_pred: np.ndarray) -> dict:
    """
    Get dictionary of TP, FN, FP, TN values given a binary label indicator and binary predictions.

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


def _get_cm_values(y_true, y_score, num: int = 10000, mode: str = "custom") -> dict:
    """
    Compute True/False Positive/Negatives of multi-class predictions y_true, y_score.

    Params:
    - y_true, y_score: np.ndarray of shape (N, num_outcs)
    - num: int describing number of steps to compute intervals for. (default = 1e6)
    - mode: str, compute custom version of multi-class averaging or not. (Default="custom")

    Returns:
        dict (threshold, TP, FN, FP, TN) of T/F P/N values for a common threshold list.
    """
    # Initialise output variables
    shape, num_outcs = (int(num), y_true.shape[-1]), y_true.shape[-1]
    tp, fn, fp, tn = np.zeros(shape), np.zeros(shape), np.zeros(shape), np.zeros(shape)

    if mode=="custom":
        # Compute thresholds per each class separately
        _min, _max = np.min(y_score, axis=0), np.max(y_score, axis=0)
        thresholds = np.linspace(start=_min, stop=_max, num=num)

    else:
        # Compute thresholds commonly to all classes
        _min, _max = np.min(y_score), np.max(y_score)
        thresholds = np.repeat(np.linspace(start=_min, stop=_max, num=num).reshape(-1, 1), 
            repeats=num_outcs, axis=-1)

    # Iterate over outcomes and thresholds
    for outc_id in range(num_outcs):
        for thresh_id, eps in enumerate(thresholds[:,outc_id]):    

            # Convert scores to binary
            y_pred_thresh = (y_score >= eps).astype(int)

            # Get conf matrix values
            conf_matrix = _get_single_cm_values(y_true[:, outc_id], y_pred_thresh[:, outc_id])

            # Update scores
            tp[thresh_id, outc_id] = conf_matrix["tp"]
            fn[thresh_id, outc_id] = conf_matrix["fn"]
            fp[thresh_id, outc_id] = conf_matrix["fp"]
            tn[thresh_id, outc_id] = conf_matrix["tn"]

    return {"thresholds": thresholds, "tp": tp, "fn": fn, "fp": fp, "tn": tn}


def _compute_bin_metrics(y_true: np.ndarray, y_score: np.ndarray, **kwargs) -> dict:
    """
    Compute True Positive Rate (TPR), False Positive Rate (FPR), recall and precision given
    multi-class true labels and predicted scores.

    Returns:
    - Dict with corresponding scores, and list of thresholds. shape (num_threshs, num_outcs)
    """
    threshs, tp, fn, fp, tn = _get_cm_values(y_true=y_true, y_score=y_score, **kwargs).values()

    # Compute sensitivity (TPR) and 1 - specificity (FPR)
    tpr = tp / (tp + fn)                                   # tpr = tp / true_positive
    fpr = fp / (fp + tn)                                   # fpr = fp / true_negative

    # Compute recall and precision
    recall = tp / (tp + fn)                                 # recall = tpr
    precision = tp / (tp + fp)                              # precision = tp / predicted positive

    # Need to be careful when tp+fp==0, precision is 1 if there are NO False negatives (follows gerbil package implementation)
    cond = tp+fp==0
    precision[cond] = 1

    return {"thresholds": threshs, "tpr": tpr, "fpr": fpr, "recall": recall, "precision": precision}


def custom_auc_auprc(y_true: np.ndarray, y_score: np.ndarray, mode: str = "custom", **kwargs) -> np.ndarray:
    """
    Compute Custom Area-under-the-receiver-curve and precision-recall curve on multi-class setting. The score is computed as regular AUROC, except
    that the thresholds per each class are not allowed to vary independently.

    For each threshold, labels with value higher than epsilon are flagged up, consequently AUROC per each class is
    computed. Area under the curve is consequently computed as regular.

    Params:
    - y_true: np.ndarray of shape (N, num_outcs) with one-hot encoded true label encodings.
    - y_score: np.ndarray of shape (N, num_outcs) with predicted probability outcome assignments.
    - mode: str, indicates whether to make custom AUROC computation or use standard AUROC. (default="custom")

    Returns:
        area: np.array of dimension (y_score.shape[-1], ) with corresponding area estimations.
    """
    metrics_dic = _compute_bin_metrics(y_true=y_true, y_score=y_score, mode=mode, **kwargs)

    # Unpack tpr and fpr
    tpr, fpr = metrics_dic["tpr"], metrics_dic["fpr"]
    assert np.all(np.diff(fpr,axis=0) <=0)            # Check FPR decreases as threshold increases 

    # Compute area under curve using trapezium rule
    auroc = np.trapz(y=tpr[::-1, :], x=fpr[::-1, :], axis=0)

    # Do the same for recall and precision
    precision, recall = metrics_dic["precision"], metrics_dic["recall"]
    auprc = np.trapz(y=precision[::-1, :], x=recall[::-1, :], axis=0)

    return {"AUROC": auroc, "AUPRC": auprc}


def plot_auc_auprc(y_true: np.ndarray, y_score: np.ndarray, mode: str = "custom", outc_names: list = None, **kwargs):
    """
    Make plots for Receiver-Operating-curves and Precision-Recall curves.
    
    Params:
    - y_true: np.ndarray of shape (N, num_outcs) with one-hot label encodings.
    - y_score: np.ndarray of shape (N, num_outcs) with predicted label assignments.
    - mode: str, indicates whether to proceed with custom AUROC computation or use standard.
    - outc_names: List or None, if names are provided, add these to plots.
    - kwargs: any other parameters (number of steps for curve estimation.
    """
    colors = get_cmap("tab10").colors

    # Manually
    if outc_names is None:
        num_outcs = y_true.shape[-1]
        outc_names = list(range(1, 1 + num_outcs))

    # Load metrics
    _, tpr, fpr, recall, precision = _compute_bin_metrics(y_true, y_score, mode=mode, **kwargs).values()
    auroc, auprc = custom_auc_auprc(y_true, y_score, mode=mode, **kwargs).values()

    # Initialise plots
    fig, ax = plt.subplots(nrows=1, ncols=2, sharex="all", sharey="all")
    outc_id = 0

    for outc_id, outc in enumerate(outc_names):
        ax[0].plot(fpr[::-1, outc_id], tpr[::-1, outc_id], linestyle="--", color=colors[outc_id], label=f"{outc} - auroc {auroc[outc_id]:.3f}")
        ax[1].plot(recall[::-1, outc_id], precision[::-1, outc_id], linestyle="--", color=colors[outc_id], label=f"{outc} - auprc {auprc[outc_id]:.3f}")

    # Add baseline 0.5 to AUROC plot
    baseline = np.linspace(0, 1, num=1000)
    ax[0].plot(baseline, baseline, label="random", linestyle="-", color=colors[1 + outc_id])

    # Add labels
    ax[0].set_xlabel("FPR")
    ax[0].set_ylabel("TPR")
    ax[1].set_xlabel("Recall")
    ax[1].set_ylabel("Precision")

    ax[0].set_title("ROC Curve")
    ax[1].set_title("PRC Curve")
    ax[0].legend()
    ax[1].legend()

    # Set suptitle and clearlayout
    fig.suptitle(f"{mode} Curves")
    plt.tight_layout()

    return fig, ax
