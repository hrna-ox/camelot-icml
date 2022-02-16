"""
Main File for defining helpful tools to visualise data and results.

Author: Henrique Aguiar
email: henrique.aguiar@eng.ox.ac.uk
"""

import matplotlib.pyplot as plt
import pandas as pd

import src.visualisation.visualisation_utils as utils


def make_group_summaries(x_data: pd.DataFrame, groups_df: pd.DataFrame, id_col: str, time_col: str, **kwargs):
    """
    Visualisation Function to compute summaries of data x_data sub-divided into cohorts specified by groups_df.

    Params:
    - x_data: pd.DataFrame with patient input data. Contains 2 identifier columns (at least), one for patient and
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
    feats = x_data.columns.tolist()

    # Separate into static and temporal feats
    _, static_vars, time_vars = utils.separate_vars_by_type(feats)

    # Compute per-group information
    group_data_dic = utils.get_data_per_group(x_data, groups_df, id_col)

    # Make summary statistics of static variables
    summary_info = utils.make_summary_statistics(group_data_dic, static_vars, id_col)

    # Make temporal plots
    nrows, ncols = utils._nrows_ncols(len(time_vars))
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, sharey=False)
    ax = utils.make_temporal_trajs(group_data_dic, time_vars, id_col, time_col, ax=ax)
    plt.show()

    return summary_info, ax


def visualise_clus_assignments():
    pass


def plot_losses():
    pass


def visualise_attention():
    pass
