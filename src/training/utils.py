#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 21 10:48:57 2021

@author: henrique.aguiar@ds.ccrg.kadooriecentre.org
"""

from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.metrics import roc_auc_score, f1_score, recall_score
import numpy as np
from sklearn.metrics.cluster import contingency_matrix

from src.models.deep_learning.camelot.model import Model as CamelotModel


def purity_score(y_true, y_pred):
    contingency_matrix_ = contingency_matrix(y_true, y_pred)

    return np.sum(np.amax(contingency_matrix_, axis=0)) / np.sum(contingency_matrix_)


def super_scores(y_true, y_pred):
    """Computes supervised scores (AUC, F1 and Recall) for y_pred compared with true labels y_true."""

    # AUC
    auc = roc_auc_score(y_true, y_pred, average="macro", multi_class="ovr")

    # Compute true assignments
    labels_true, labels_pred = np.argmax(y_true, axis=1), np.argmax(y_pred, axis=1)

    # F1
    f1 = f1_score(labels_true, labels_pred, average="macro")

    # Recall
    rec = recall_score(labels_true, labels_pred, average="macro")

    # Purity SCores
    purity = purity_score(labels_true, labels_pred)

    return auc, f1, rec, purity


def unsuper_scores(X, clusters):
    """Compute unsupervised metric scoring."""

    # Convert to 2D format
    X1 = X.reshape(X.shape[0], -1)

    # Compute main metrics
    sil = silhouette_score(X1, clusters, metric="euclidean")
    dbi = davies_bouldin_score(X1, clusters)
    vri = calinski_harabasz_score(X1, clusters)

    # Compute the same taking average over each feature dimension
    sil_avg, dbi_avg, vri_avg = 0, 0, 0

    for feat in range(X.shape[-1]):
        sil_avg += silhouette_score(X[:, :, feat], clusters, metric="euclidean")
        dbi_avg += davies_bouldin_score(X[:, :, feat], clusters)
        vri_avg += calinski_harabasz_score(X[:, :, feat], clusters)

    # Compute average
    num_feats = X.shape[-1]

    return sil, sil_avg / num_feats, dbi, dbi_avg / num_feats, vri, vri_avg / num_feats


def get_model_from_str(model_name, **kwargs):
    """
    Function to load correct model from the model name.

    Params:
    - model_name: name of model.
    - **kwargs: model initialisation parameters.

    returns: Corresponding model class object.
    """
    if "camelot" in model_name.lower():
        model = CamelotModel(**kwargs)

    else:
        raise ValueError(f"Correct Model name not specified. Value {model_str} given.")

    return model
