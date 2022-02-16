"""
Main File for defining helpful tools to visualise data and results.

Author: Henrique Aguiar
email: henrique.aguiar@eng.ox.ac.uk
"""

import matplotlib.pyplot as plt
import pandas as pd

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
    # Re-define save_fd
    save_fd = save_fd.replace("results", "visualisations").replace("experiments", "visualisations")
    if not os.path.exists(save_fd):
        os.makedirs(save_fd)

    # Unpack input data and data properties
    X, _ = data_info["data_og"]
    data_properties = data_info["data_properties"]

    # Subset to test set only
    ids_test, id_col = data_info["ids"][-1][:, 0, 0], data_properties["id_col"]
    X_test = X[X[id_col].isin(ids_test)]

    # Compute groups
    clus_onehot = pd.get_dummies(clus_pred)

    # Compute average static variables summary and trajectories
    clus_summary_info, (fig, axs) = make_group_summaries(X_test, clus_onehot, **data_properties)

    # Save configuration if available
    if "model_config" in kwargs.keys():
        # save config
        with open(save_fd + "data_load_config.json", "w+", newline="\n") as f:
            json.dump(kwargs["model_config"], f, indent=4)
            f.close()

    # Save input avg information
    clus_summary_info.to_csv(save_fd + "clus_summary_info.csv", index=True, header=True)

    # Save axes
    fig.savefig(save_fd + "clus_time_feats_avg_trajs.png", dpi=200)

    return None


def visualise_data_groups(data_og, data_properties: dict, data_load_config: dict, **kwargs):
    """
    Visualise input data to model given input arrays X_og, y_og, and identifier columns.

    Params:
    - data_df: tuple of dataframes (X, y) representing cohort data pre-processing transformations.
    - data_properties: dict of relevant information and objects relevant to processing of data_df. Must include
    "time_col" and "id_col" key objects.
    - data_load_config: dict of input parameters for data processing.

    Returns:
    - Visualises input data trajectories and summaries.
    - saves to relevant path with load configuration.
    """
    # Unpack input data
    X, y = data_og

    data_outc_info, (fig, axs) = make_group_summaries(X, y, **data_properties)

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

    return None


def make_group_summaries(input_data: pd.DataFrame, groups_df: pd.DataFrame, id_col: str, time_col: str, **kwargs):
    """
    Visualisation Function to compute summaries of data x_data subdivided into cohorts specified by groups_df.

    Params:
    - input_data: pd.DataFrame with patient input data. Contains 2 identifier columns (at least), one for patient and
    another for time of observation.
    - groups_df: pd.DataFrame with group one-hot assignment or group probability assignment per each patient.
    - id_col: str, which column of x_data identifies patient ids.
    - time_col: str, which column of x_data identifies temporal info.

    Returns:
        - plots of temporal feature averages per sub-cohort with standard error deviation.
        - prints summary statistics for static variables.
        - Saves visualisations in corresponding folder.
    """
    if len(groups_df.shape) == 1:
        groups_df = pd.get_dummies(groups_df)

    # Get features
    feats = input_data.columns.tolist()

    # Separate into static and temporal feats
    _, static_vars, time_vars = utils.separate_vars_by_type(feats)

    # Compute per-group information
    group_data_dic = utils.get_data_per_group(input_data, groups_df, id_col)

    # Make summary statistics of static variables
    summary_info = utils.make_summary_statistics(group_data_dic, static_vars, id_col)

    # Make temporal plots
    nrows, ncols = utils._nrows_ncols(len(time_vars))
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, sharey=False)
    ax = utils.make_temporal_trajs(group_data_dic, time_vars, id_col, time_col, ax=ax)
    plt.show()

    return summary_info, (fig, ax)


def visualise_clus_assignments():
    pass


def plot_losses(**kwargs):
    """
    Plot losses if any parameters passed containing the word loss.

    Params:
    **kwargs - dictionary key:value objects for multiple output-type objects.

    Returns:
        - Plot loss functions for all key values containing the expression loss.
    """

    for key, value in kwargs.items():
        if "loss" in key:
            # DO SOMETHING
            pass


def visualise_attention():
    pass
