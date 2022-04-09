#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File to main.py results performance.

Includes evaluation based on predicted outcomes and cluster scores, if available.

@author: Henrique Aguiar, Department of Engineering Science
email: henrique.aguiar@eng.ox.ac.uk
"""
from csv import writer

import pandas as pd

import src.results.results_utils as utils


def evaluate(y_true=None, y_pred=None, clus_pred=None, data_info=None, save_fd=None, avg=None, scores=None,
             **kwargs):
    """
    Evaluate function to print result information given results and/or experiment ids. Returns a dictionary of scores
    given the outputs of the model (for e.g. if the model does not do clustering, then no cluster scores are returned).

    Params:
    - y_true: array-like of shape (N, num_outcs) with one-hot encodings of true class membership. defaults to None.
    - y_pred: array-like of shape (N, num_outcs) with predicted outcome likelihood assignments. defaults to None.
    - clus_pred: array-like of shape (N, num_clus) with predicted cluster membership probabilities. default to None
    - data_info: dict of input data information and objects.
    - save_fd: str, folder where to write scores to.
    - age: str, useful how to average class individual scores (defaults to None, which returns no average).
    - scores: array-like of shape (N, ) of scores. Only relevant for score-based benchmarks, such as NEWS2 and/or ESI.
    - **kwargs: other parameters given to scoring supervised scores.


    Returns:
        - list of supervised performance scores and cluster performance scores (where valid).
        - prints information for each of the associated score.
        - saves score information in relevant folder.
    """
    # Checks for instances Df vs array and loads data properties
    if isinstance(y_true, pd.DataFrame):
        y_true = y_true.values

    if "news" in save_fd.lower() or "esi" in save_fd.lower():

        # Compute scores
        scores = utils.compute_from_eas_scores(y_true=y_true, scores=scores, **kwargs)

        # Definitions for completeness
        cm = None
        clus_metrics = {}

    else:

        if isinstance(y_pred, pd.DataFrame):
            y_pred = y_pred.values

        # Load data relevant properties
        data_properties = data_info["data_properties"]
        outc_names = data_properties["outc_names"]

        # Compute scores and confusion matrix
        scores, cm, Roc_curves = utils.compute_supervised_scores(y_true, y_pred, avg=avg, outc_names=outc_names)

        # Convert Confusion Matrix to pdDataFrame
        cm = pd.DataFrame(cm, index=pd.Index(data=outc_names, name="True Class"),
                          columns=pd.Index(data=outc_names, name="Predicted Class"))

        # If clustering results exist, output cluster performance scores
        clus_metrics = {}
        if clus_pred is not None:

            if isinstance(clus_pred, pd.DataFrame):
                clus_pred = clus_pred.values

            # Compute X_test in 3 dimensional format
            min_, max_ = data_properties["norm_min"], data_properties["norm_max"]
            x_test_3d = data_info["X"][-1] * (max_ - min_) + min_

            # Compute metrics
            try:
                clus_metrics = utils.compute_cluster_performance(x_test_3d, clus_pred=clus_pred, y_true=y_true)
            except ValueError:
                print("Too little predicted labels. Can't compute clustering metrics.")
                clus_metrics = {}

        # Save Confusion matrix
        cm.to_csv(save_fd + "confusion_matrix.csv", index=True, header=True)

    # Jointly compute scores
    scores = {**scores, **clus_metrics}

    # Save
    # for key, value in Roc_curves.items():
    
    #     # Get fig, ax and save
    #     fig, _ = value
    #     fig.savefig(save_fd + key)


    with open(save_fd + "scores.csv", "w+", newline="\n") as f:
        csv_writer = writer(f, delimiter=",")

        # Iterate through score key and score value(s)
        for key, value in scores.items():

            # Define row to save
            if isinstance(value, list):
                row = tuple([key, *value])
            else:
                row = tuple([key, value])

            csv_writer.writerow(row)

    # Print information
    print("\nScoring information for this experiment\n")
    for key, value in scores.items():
        print(f"{key} value: {value}")

    print("\nConfusion Matrix for predicting results", cm, sep="\n")

    return scores
