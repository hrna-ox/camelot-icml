#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 21 10:48:57 2021

@author: henrique.aguiar@ds.ccrg.kadooriecentre.org
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from typing import Union, List

from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, confusion_matrix
from sklearn.metrics import roc_auc_score, f1_score, recall_score, precision_score, average_precision_score
from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay
from sklearn.metrics.cluster import contingency_matrix

# List of ADMISSIBLE RESULTS SAVED KEYS
ADMISSIBLE_RESULT_KEYS = ["y_pred", "outc_pred", "y_true", "pis_pred", "clus_pred", "clus_phenotypes"]


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
    tp = ((has_label_true==1) & (has_label_pred==1)).sum()

    # True=1, False=0 for false negatives
    fn = ((has_label_true==1) & (has_label_pred==0)).sum()

    # True=0, False=1 for false positives
    fp = ((has_label_true==0) & (has_label_pred==1)).sum()

    # True=0, False=0 for true negatives
    tn = ((has_label_true==0) & (has_label_pred==0)).sum()

    # Return dictionary
    return {"tp": tp, "fn": fn, "fp": fp, "tn": tn}


def _custom_cm_over_threshold(y_true: np.ndarray, y_score: np.ndarray) -> tuple:
    """
    Compute True/False Positive/Negatives of multi-class predictions y_true, y_score with a commonly variying threshold.

    Params:
    - y_true: np.ndarray of shape (N, num_outcs) with one-hot encoded true label encodings.
    - y_score: np.ndarray of shape (N, num_outcs) with predicted probability outcome assignments.

    Returns:
        Tuple (threshold, TP, FN, FP, TN) of T/F P/N values for a common threshold list.
    """
    # Compute varying thresholds
    _min, _max = np.min(y_score), np.max(y_score)
    thresholds = np.linspace(start=_min, stop=_max, num=10000, endpoint=True)

    # Initialise output variables
    shape = (10000, y_true.shape[-1])
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

    return thresholds, tp, fn, fp, tn


def custom_auc(y_true: np.ndarray, y_score: np.ndarray) -> np.ndarray:
    """
    Compute Custom Area-under-the-receiver-curve on multi-class setting. The score is computed as regular AUROC, except that the thresholds per each class
    are not allowed to vary independently. 

    For each threshold, labels with value higher than epsilon are flagged up, consequently AUROC per each class is computed. Area under the curve is consequently
    computed as regular.

    Params:
    - y_true: np.ndarray of shape (N, num_outcs) with one-hot encoded true label encodings.
    - y_score: np.ndarray of shape (N, num_outcs) with predicted probability outcome assignments.

    Returns:
        area: np.array of dimension (y_score.shape[-1], ) with corresponding area eestimations.
    """
    _ , tp, fn, fp, tn = _custom_cm_over_threshold(y_true=y_true, y_score=y_score)

    # Compute Sensitivity and Specificity
    tpr = tp / (tp + fn)
    fpr = fp / (fp + tn)

    # Ensure fpr always non-increasing
    assert np.all(np.diff(fpr[1:, :],axis=0) <= 0, axis=None)

    # Compute area under curve with trapezoidal rule - multiply by -1 as fpr is decreasing
    area = - np.trapz(y=tpr, x=fpr, axis=0)

    return area
    

def custom_prc(y_true: np.ndarray, y_score: np.ndarray) -> np.ndarray:
    """
    Compute custom area-under-the-precision-recall curve on multi-class setting. The score is computed as regular AUPRC, except thresholds per each class
    are not allowed to vary independently, and apply to each class simultaneously.
    
    Params:
    - y_true: np.ndarray of shape (N, num_outcs) with one-hot encoded true label encodings.
    - y_score: np.ndarray of shape (N, num_outcs) with predicted probability outcome assignments.

    Returns:
        area: np.array of dimension (y_score.shape[-1], ) with corresponding area eestimations.
    
    """
    _ , tp, fn, fp, tn = _custom_cm_over_threshold(y_true=y_true, y_score=y_score)

    # Compute Precision and recall
    precision = np.divide(tp ,tp + fp, out=np.zeros_like(tp), where=tp+fp==0)
    recall = tp / (tp + fn)

    # Ensure always decreasing
    assert np.all(np.diff(recall, axis=0) <= 0, axis=None)

    # Compute Area under curve - need to multiply by -1 as recall is decreasing (and not increasing)
    area = - np.trapz(y=precision, x=recall, axis=0)

    return area


def get_clus_outc_numbers(y_true: np.ndarray, clus_pred: np.ndarray):
    """
    Compute contingency matrix: entry (i,j) denotes the number of patients with true sample i and predicted clus j.

    Params:
    - y_true: array-like of true outcome one-hot assignments.
    - clus_pred: array-like of cluster label assignments.

    Returns:
        - cont_matrix: numpy ndarray of shape (num_outcs, num_clus) where entry (i,j) denotes the number of patients
        with outcome i and cluster j.
    """

    # Convert to categorical
    labels_true = np.argmax(y_true, axis=1)
    labels_pred = clus_pred

    # Compute contingency matrix
    cont_matrix = contingency_matrix(labels_true, labels_pred)

    return cont_matrix


def _get_outputs_dic(model_name: Union[List, None], run_num: Union[List, None]):
    """
    Get results dictionary given results_dictionary
    """

    # Save paths where data is saved.
    save_fd = f"results/{model_name}/run{run_num}/"
    output_dic = {}

    # Load results given admissible keys
    for key in ADMISSIBLE_RESULT_KEYS:

        # Try to load object and save to dictionary
        try:
            load_obj = pd.read_csv(save_fd + key + ".csv", index_col=0, header=0)
            output_dic[key] = load_obj

            print(f"Object {key} successfully loaded.")

        except FileNotFoundError:
            print(f"Object {key} not loaded.")
            pass

    return output_dic


def _convert_to_one_hot_from_probs(array_pred: Union[np.ndarray, pd.DataFrame]):
    """
    Convert array of predicted class/cluster probability assignments to one-hot encoding of the most common class/clus.

    Params: - array_pred: array-like of shape (N, K), where K is the number of target classes, with probability class
    assignments.

    Returns:
    - Output: array-like of shape (N, K) with one-hot encoded most likely class assignments.
    """

    # Convert to array if necessary
    if isinstance(array_pred, pd.DataFrame):
        array_pred = array_pred.values

    # Compute dimensionality
    if len(array_pred.shape) == 2:
        _, K = array_pred.shape

        # Convert to categorical
        class_pred = np.eye(K)[np.argmax(array_pred, axis=1)]

    else:
        # Array_pred already categorical
        K = array_pred.size

        # Convert to categorical
        class_pred = np.eye(K)[array_pred]

    return class_pred


def purity(y_true: np.ndarray, clus_pred: np.ndarray) -> float:
    """
    Computes Purity Score from predicted and true outcome labels. Purity Score is an external cluster validation tool
    which computes the largest number of individuals from a given class in a given cluster, and consequently averages
    this values over the number of clusters.

    Params:
    - y_true: array-like of shape (N, num_outcs) of true outcome labels in one-hot encoded format.
    - clus_pred: array-like of shape (N, num_clus) of predicted outcome cluster assignments.

    Returns:
    - purity_score: float indicating purity score.
    """

    # Convert clus_pred to categorical cluster assignments
    cm = get_clus_outc_numbers(y_true, clus_pred)  # shape (num_outcs, num_clus)

    # Number of most common class in each cluster
    max_class_numbers = np.amax(cm, axis=0)

    # Compute average
    purity_score = np.sum(max_class_numbers) / np.sum(cm)

    return purity_score


def compute_supervised_scores(y_true: np.ndarray, y_pred: np.ndarray, avg=None):
    """
    Compute set of supervised classification scores between y_true and y_pred. List of metrics includes:
    a) AUROC, b) Recall, c) F1, d) Precision, e) Adjusted Rand Index and f) Normalised Mutual Information Score.

    Params:
    - y_true: array-like of shape (N, num_outcs) of one-hot encoded true class membership.
    - y_pred: array-like of shape (N, num_outcs) of predicted outcome probability assignments.
    - avg: parameter for a), b), c) and d) computation indicating whether class scores should be averaged, and how.
    (default = None, all scores reported).

    Returns:
        - Dictionary of performance scores:
            - "ROC-AUC": list of AUROC One vs Rest values.
            - "Recall": List of Recall One vs Rest values.
            - "F1": List of F1 score One vs Rest values.
            - "Precision": List of Precision One vs Rest values.
            - "ARI": Float value indicating Adjusted Rand Index performance.
            - "NMI": Float value indicating Normalised Mutual Information Score performance.
    """
    if isinstance(y_pred, pd.DataFrame):
        y_pred = y_pred.values

    if isinstance(y_true, pd.DataFrame):
        y_true = y_true.values

    # Get PRC
    prc, auc = np.zeros(shape=y_pred.shape[-1]), np.zeros(shape=y_pred.shape[-1])
    for outc_id in range(y_pred.shape[-1]):

        # Update prc and auc scores
        auc[outc_id] = roc_auc_score(y_true=y_true[:, outc_id], y_score=y_pred[:, outc_id], average=avg)
        prc[outc_id] = average_precision_score(y_true=y_true[:, outc_id], y_score=y_pred[:, outc_id], average=avg)

    # GET ROC AND PRC CURVES
    roc_prc_curves = {}
    for outc_id in range(y_pred.shape[-1]):

        # Add curve to map
        fig, ax = plt.subplots(nrows=1, ncols=2)

        RocCurveDisplay.from_predictions(y_true[:, outc_id], y_pred[:, outc_id], ax=ax[0])
        PrecisionRecallDisplay.from_predictions(y_true[:, outc_id], y_pred[:, outc_id], ax=ax[1])

        # Add info
        ax[0].set_xlabel("FPR")
        ax[0].set_ylabel("TPR")
        ax[0].set_title(f"ROC Curve for outcome {outc_id}")

        # Same for second ax
        ax[1].set_xlabel("FPR")
        ax[1].set_ylabel("TPR")
        ax[1].set_title(f"PRC Curve for outcome {outc_id}")

        # Add to curves
        roc_prc_curves[f"Curves {outc_id}"] = fig, ax

    # Get custom ROC and PRC Curves
    _, tp, fn, fp, tn = _custom_cm_over_threshold(y_true=y_true, y_score=y_pred)
    for outc_id in range(y_pred.shape[-1]):

        # Get sensitivity, sensitivity precision and recall
        tpr = tp / (tp + fn)
        fpr = fp / (fp + tn)
        precision = np.divide(tp ,tp + fp, out=np.zeros_like(tp), where=tp+fp==0)
        recall = tp / (tp + fn)

        # Initialise plots
        fig, ax = plt.subplots(nrows=1, ncols=2)
        baseline = np.linspace(0, 1, num=1000)

        # Plot AUROC
        ax[0].plot(fpr[::-1], tpr[::-1], linestyle="--", color="red", label="pred curve")
        ax[0].plot(baseline, baseline, linestyle="-", color="blue", label="baseline")
        
        ax[0].set_xlabel("FPR")
        ax[0].set_ylabel("TPR")
        ax[0].set_title(f"Custom ROC Curve for outcome {outc_id}")
        ax[0].legend()

        # Plot PRC
        ax[1].plot(recall[::-1], precision[::-1], linestyle="--", color="red", label="pred curve")
        ax[1].plot(baseline, baseline, linestyle="-", color="blue", label="baseline")

        ax[1].set_xlabel("Recall")
        ax[1].set_ylabel("Precision")
        ax[1].set_title(f"Custom PRC Curve for outcome {outc_id}")
        ax[1].legend()

        # Add to curves
        roc_prc_curves[f"Custom Curves {outc_id}"] = fig, ax


    # Compute custom AUROC and AUPRC
    auc_common = custom_auc(y_true=y_true, y_score=y_pred)
    prc_common = custom_prc(y_true=y_true, y_score=y_pred)

    # Convert input arrays to categorical labels
    labels_true, labels_pred = np.argmax(y_true, axis=1), np.argmax(y_pred, axis=1)

    # Compute F1
    f1 = f1_score(labels_true, labels_pred, average=avg)

    # Compute Recall
    rec = recall_score(labels_true, labels_pred, average=avg)

    # Compute Precision
    prec = precision_score(labels_true, labels_pred, average=avg)

    # Compute ARI
    ari = adjusted_rand_score(labels_true, labels_pred)

    # Compute NMI
    nmi = normalized_mutual_info_score(labels_true, labels_pred)

    # Compute Confusion matrix
    cm = confusion_matrix(y_true=labels_true, y_pred=labels_pred, labels=None, sample_weight=None, normalize=None)

    # Return Dictionary
    scores_dic = {
        "ROC-AUC": auc,
        "ROC-PRC": prc,
        "ROC-AUC-custom": auc_common,
        "ROC-PRC-custom": prc_common,
        "F1": f1,
        "Recall": rec,
        "Precision": prec,
        "ARI": ari,
        "NMI": nmi
    }

    return scores_dic, cm, roc_prc_curves


def compute_from_eas_scores(y_true: np.ndarray, scores: np.ndarray, outc_names: np.ndarray = None, **kwargs) -> dict:
    """
    Compute supervised performance metrics given input array scores.


    Params:
    - y_true: array-like of shape (N, num_outcs).
    - scores: array-like of shape (N, ).
    - outc_names: array-like of shape (num_outcs, ) with particular outcome names.
    - kwargs: any other arguments. They are kept for coherence.

    Returns:
    - dict with scores ROC-AUC, F1, Recall, Precision per class
    """

    # Useful info
    num_outcs = y_true.shape[-1]

    if outc_names is None:
        outc_names = range(num_outcs)

    # Useful info and initialise output
    SCORE_NAMES = {"ROC-AUC": roc_auc_score, "F1": f1_score, "Recall": recall_score, "Precision": precision_score}
    output_dic = {}

    # Convert to useful format
    if isinstance(scores, pd.Series) or isinstance(scores, pd.DataFrame):
        scores = scores.values.reshape(-1)

    # Convert scores to probability thresholds
    scores_max = np.max(scores)
    scores = scores / scores_max

    # Iterate through the 4 binary scores
    for score_name, score_fn in SCORE_NAMES.items():

        # Get scoring fn
        scoring_fn = SCORE_NAMES[score_name]
        output_dic[score_name] = []

        # Iterate over outcomes
        for outc_id, outc in enumerate(outc_names):

            # Compute score for this particular outcome
            outc_labels_true = y_true[:, outc_id] == 1
            output_dic[score_name].append(scoring_fn(outc_labels_true.astype(int), scores))

    # Return object
    return output_dic


def compute_cluster_performance(X, clus_pred, y_true):
    """
    Compute cluster performance metrics given input data X and predicted cluster probability assignments clus_pred.
    Metrics computed include a) Silhouette Score, b) Davies Bouldin Score, c) Calinski Harabasz Score and d) Purity.
    Performance is computed averaged over features.

    Params:
    - X: array-like of shape (N, T, D_f) where N is the number of patients, T the number of time steps and D_f the
    number of features (+2, the id col and the time col).
    - clus_pred: array-like of shape (N, ) with predicted label assignments.
    - y_true: array-like of shape (N, num_outcs) with one-hot encoding of true class assignments.

    Returns:
        - Dictionary of output cluster performance metrics. Includes:
            - "Silhouette": Silhouette Score computation.
            - "DBI": Davies-Bouldin Index computation.
            - "VRI": Variance-Ratio Criterion (also known as Calinski Harabasz Index).
            - "Purity": Purity Score computation.
    """

    # If not converted to categorical, then convert
    if len(clus_pred.shape) == 2:
        clus_pred = np.argmax(clus_pred, axis=1)

    # Compute the same taking average over each feature dimension
    sil_avg, dbi_avg, vri_avg = 0, 0, 0

    for feat in range(X.shape[-1]):
        sil_avg += silhouette_score(X[:, :, feat], clus_pred, metric="euclidean")
        dbi_avg += davies_bouldin_score(X[:, :, feat], clus_pred)
        vri_avg += calinski_harabasz_score(X[:, :, feat], clus_pred)

    # Compute Purity Score
    purity_score = purity(y_true, clus_pred)

    # Compute average factor
    num_feats = X.shape[-1]

    # Return Dictionary
    clus_perf_dic = {
        "Silhouette": sil_avg / num_feats,
        "DBI": dbi_avg / num_feats,
        "VRI": vri_avg / num_feats,
        "Purity": purity_score / num_feats
    }

    return clus_perf_dic
