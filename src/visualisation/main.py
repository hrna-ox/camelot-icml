"""
Main File for defining helpful tools to visualise data and results.

Author: Henrique Aguiar
email: henrique.aguiar@eng.ox.ac.uk
"""

import matplotlib.pyplot as plt
from sklearn.metrics.cluster import contingency_matrix as cont_matrix

import pandas as pd
import numpy as np

import os, json

import src.visualisation.visualisation_utils as utils


def visualise_cluster_groups(clus_pred, data_info: dict, save_fd: str, **kwargs):
    """
    Visualise input data to model given input arrays X_og, y_og, and identifier columns.

    Params:
    - clus_pred: pd.DataFrame of shape (N, ) of categorical cluster assignment.
    - data_info: dictionary with a variety of data related information
    - save_fd: str where model results are being saved. We will convert this to visualisations.

    Returns:
    - Visualises cluster average trajectories for multiple features.
    - Saves figures to respective folders.
    """

    # Unpack input data and data properties
    X, _ = data_info["data_og"]
    data_properties = data_info["data_properties"]

    # Save and load config
    save_fd, data_load_config, data_name = utils.get_basic_info(save_fd=save_fd, data_info=data_info)

    # Subset to test set only
    ids_test, id_col = data_info["ids"][-1][:, 0, 0], data_properties["id_col"]
    X_test = X[X[id_col].isin(ids_test)]

    # Compute groups
    clus_onehot = pd.get_dummies(clus_pred)

    # Compute average static variables summary and trajectories
    clus_summary_info, (fig, axs) = utils.make_group_summaries(X_test, clus_onehot, **data_properties)

    # Save configuration if available
    if "model_config" in kwargs.keys():
        # save config
        with open(save_fd + "model_config.json", "w+", newline="\n") as f:
            json.dump(kwargs["model_config"], f, indent=4)
            f.close()

    # Save input avg information
    clus_summary_info.to_csv(save_fd + "clus_summary_info.csv", index=True, header=True)

    # Save axes
    fig.savefig(save_fd + "clus_time_feats_avg_trajs.png", dpi=200)
    print(clus_summary_info)

    return None


def visualise_data_groups(data_info, **kwargs):
    """
    Visualise input data to model given input data information.

    Params:
    - data_info: dict, with patient data information.

    Returns:
    - Visualises input data trajectories and summaries.
    - saves to relevant path with load configuration.
    """
    # Unpack input data
    X, y = data_info["data_og"]
    data_properties = data_info["data_properties"]
    data_load_config = data_info["data_load_config"]

    data_outc_info, (fig, axs) = utils.make_group_summaries(X, y, **data_properties)

    # Get save path
    data_name = data_load_config["data_name"]
    save_fd = f"visualisations/{data_name}/"

    if not os.path.exists(save_fd):
        os.makedirs(save_fd)

    # save config
    with open(save_fd + "data_load_config.json", "w+") as f:
        json.dump(data_load_config, f, indent=4)
        f.close()

    # Save input avg information
    data_outc_info.to_csv(save_fd + "static_feats_description.csv", index=True, header=True)

    # Save axes
    fig.savefig(save_fd + "time_feats_avg_trajectories.png", dpi=200)
    print(data_outc_info)

    return None


def plot_losses(save_fd=None, history=None, data_info:dict = None, **kwargs):
    """
    Plot losses if:
    a) history object has been provided (useful for neural networks)
    b) any loss output has been provided.

    Params:
    - data_info: dictionary with input data information
    - save_fd: save_folder for figures. defaults to None, which does not save plots.
    - history: Tensorflow history object with information about loss functions during training. If None, ignores this
    parameter. (Default = None)
    **kwargs - dictionary key:value objects for multiple output-type objects.

    Returns:
        - Plot loss functions for all key values containing the expression loss.
    """
    save_fd, data_load_config, data_name = utils.get_basic_info(save_fd=save_fd, data_info=data_info)

    # Identify main training losses and plot if they exist
    if history is not None:

        # Compute main training losses - ignore validation, as these will be called manually
        main_training_losses = [loss for loss in history.history.keys() if loss != "lr" and "val_" not in loss]

        # Iterate over losses
        for loss in main_training_losses:

            # Get loss values
            train_values = history.history[loss]
            val_values = history.history[f"val_{loss}"]

            # Plot loss
            (fig, ax) = utils.plot_loss_fn(train_values, val_values)

            ax.set_ylabel(f"Loss {loss} (-)")
            ax.set_title(f"Evolution of loss {loss} during training.")

            # Save figure if save_fd provided
            if save_fd is not None:
                fig.savefig(save_fd + f"{loss}_evolution.png", dpi=200)

    # Now do the same for any other loss value objects
    for key, value in kwargs.items():
        if "loss" in key:

            # Get train and validation values
            train_values = value["train_loss"]
            val_values = value["val_loss"]

            # Make plot
            (fig, ax) = utils.plot_loss_fn(train_values, val_values)

            # Add Info
            ax.set_ylabel(f"Loss {key} (-)")
            ax.set_title(f"Evolution of loss {key} during training.")

            # Save figure if save_fd provided
            if save_fd is not None:
                fig.savefig(save_fd + f"{key}_evolution.png", dpi=200)

    return None


def visualise_cluster_assignment(clus_pred, data_info, pis_pred=None, save_fd=None, **kwargs):
    """
    Visualise distribution of cluster memberships probabilities, and how the different cluster behave with regards to
    existing classes.

    Params:
    - clus_pred: array-like of shape (N, ) with categorical cluster assignment values.
    - data_info: dict of data relevant information and object. Must contain key "y" with a 3-sized tuple object, where
    the last index represents y_test.
    - pis_pred: array-like of shape (N, num_clus) with predicted probability of cluster assignments. If None, then
    ignore this parameter and consider only clus_pred. If not None, this overrides clus_pred.
    - save_fd: where to save results. This will be overridden to save on the visualisations folder.

    Returns:
        - Plots a) average cluster probability assignment distributions.
            b) Outcome distribution across all clusters.
        - Saves plots if save_fd provided.
    """
    # Get y_test
    labels_true = np.argmax(data_info["y"][-1], axis=1)
    save_fd, data_load_config, data_name = utils.get_basic_info(save_fd, data_info=data_info)

    # Check if pis_pred exist
    if pis_pred is not None:
        clus_pred = np.argmax(pis_pred.values, axis=1)

    # Compute contingency matrix
    cm = pd.DataFrame(cont_matrix(labels_true=labels_true, labels_pred=clus_pred),
                      index=data_info["data_properties"]["outc_names"],
                      columns=[f"Clus {k}" for k in np.unique(clus_pred)])

    # Get data driven phenotypes
    data_clus_phens = cm / cm.sum(axis=0)
    data_outc_phens = cm.divide(cm.sum(axis=1), axis=0)


    # Save dfs if not None
    if save_fd is not None:
        cm.to_csv(save_fd + "contingency_matrix.csv", index=True, header=True)
        data_clus_phens.to_csv(save_fd + "data_clus_phens.csv", index=True, header=True)
        data_outc_phens.to_csv(save_fd + "data_outc_phens.csv", index=True, header=True)


    # Print Data driven results
    if cm.shape[0] < cm.shape[1]:
        cm = cm.T
        data_clus_phens = data_clus_phens.T
        data_outc_phens = data_outc_phens.T

    print("Contingency Matrix outcomes x clusters: ", cm, sep="\n")
    print("\nClus-normalised data-driven phenotyes: ", data_clus_phens, sep="\n")
    print("\nOutc-normalised data-driven phenotyes: ", data_outc_phens, sep="\n")



    if pis_pred is not None:
        fig, ax = utils.get_dists_per_clus(pis_pred)
        fig.suptitle("Distribution of cluster membership assignment within each cluster.")
        fig.supxlabel("Clusters")
        fig.supylabel("Prob. Assign.")

        # Save if save_fd is provided
        if save_fd is not None:
            fig.savefig(save_fd + "pis_distribution_per_clus.png", dpi=200)

    return None


def visualise_attention_maps(save_fd=None, data_info:dict = None, clus_pred=None, **kwargs):
    """
    Plot attention maps if those are provided.

    Params:
    - save_fd: str, of where results have been saved.
    - data_info: dict, dictionary of input data information.
    - clus_pred: pd.DataFrame, array-like of categorical cluster assignment.
    - **kwargs: other output results. Any attention weights are saved under "**attention**" keys.

    Returns:
    - Plot of attention maps.
    - Saves attention maps.
    """
    # Save data configuration
    feats = data_info["data_properties"]["feats"]
    save_fd, data_load_config, data_name = utils.get_basic_info(save_fd=save_fd, data_info=data_info)

    # Check if attention weights exist on outputs
    for key, values in kwargs.items():
        if ("attention" in key or "att" in key) and ("unnorm" not in key):

            # Load weights
            alpha, beta, gamma = values

            # Plot attention weights
            (fig1, ax1), (fig2, ax2) = utils.plot_attention(alpha, beta, gamma=gamma, clus_pred=clus_pred, feats=feats)

            # Add suptitles
            fig1.suptitle("Cluster-wise attention with gamma.")
            fig1.supxlabel("Time IDs")
            fig1.supylabel("Features")

            # Same for Fig2
            fig2.suptitle("Cluster-wise attention with subcohort.")
            fig2.supxlabel("Time IDs")
            fig2.supylabel("Features")

            # Save if save_fd
            if save_fd is not None:
                fig1.savefig(save_fd + "attention_v1.png", dpi=200)
                fig2.savefig(save_fd + "attention_v2.png", dpi=200)

    return None
