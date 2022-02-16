#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File to main.py results performance.

Includes evaluation based on predicted outcomes and cluster scores, if available.

@author: Henrique Aguiar, Department of Engineering Science
email: henrique.aguiar@eng.ox.ac.uk
"""
from csv import writer
import src.results.results_utils as utils


def evaluate(y_true, y_pred, clus_pred=None, X_og_3D=None, save_fd=None, avg=None,
             **kwargs):
    """
    Evaluate function to print result information given results and/or experiment ids. Returns a dictionary of scores
    given the outputs of the model (for e.g. if the model does not do clustering, then no cluster scores are returned).

    Params:
    - y_true: array-like of shape (N, num_outcs) with one-hot encodings of true class membership.
    - y_pred: array-like of shape (N, num_outcs) with predicted outcome likelihood assignments.
    - clus_pred: array-like of shape (N, num_clus) with predicted cluster membership probabilities. default to None
    - X_og_3D: array-like of shape (N, T, D_f) of input array data.
    - save_fd: str, folder where to write scores to.
    - age: str, useful how to average class individual scores (defaults to None, which returns no average).
    - **kwargs: other parameters given to scoring supervised scores.


    Returns:
        - list of supervised performance scores and cluster performance scores (where valid).
        - prints information for each of the associated score.
        - saves score information in relevant folder.
    """

    # Compute outcome related scores
    scores = utils.compute_scores(y_true, y_pred, avg=avg)

    # If clustering results exist, output cluster performance scores
    clus_metrics = {}
    if clus_pred:  # clus_pred is not None

        # Compute metrics
        clus_metrics = utils.compute_cluster_performance(X_og_3D, clus_pred=clus_pred, y_true=y_true)

    # Jointly compute scores
    scores = {**scores, **clus_metrics}

    # Save
    with open(save_fd + "scores.csv", "w+") as f:
        csv_writer = writer(f, delimiter=",", newline="\n")

        # Iterate through score key and score value(s)
        for key, value in scores.items():

            # Define row to save
            if isinstance(value, list):
                row = [key] + value
            else:
                row = [key, value]

            csv_writer.writerow(row)

    # Print information
    print("\nScoring information for this experiment\n")
    for key, value in scores.items():
        print(f"{key} value: {value}")

    return scores
