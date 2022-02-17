"""

"""
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import seaborn as sns

import json, os
from typing import Union, List, Tuple

from src.data_processing.data_loading_utils import _is_temporal_feat, _is_id_feat, _is_static_feat

# ------------------ Useful objects ----------------------
colors = get_cmap("tab10").colors

with open("data/HAVEN/processed/units_dic.json", "r") as f:
    UNITS_DIC = json.load(f)
    f.close()

with open("data/MIMIC/processed/units_dic.json", "r") as f:
    UNITS_DIC = {**UNITS_DIC, **json.load(f)}
    f.close()


def plot_attention(alpha, beta, gamma, clus_pred = None):
    """
    Plot attention maps given alpha, beta maps and predicted cluster maps.

    Params:
    - alpha, beta: alpha, beta attention weights.
    - clus_pred: array-like of shape (N, ) with categorical cluster assignment.

    Returns: Tuple of tuple of (fig, ax) objects representing:
        - general alpha, beta, gamma for cluster membership.
        - alpha, beta attention maps for cluster cohorts identified in clus_pred.
    """
    # Compute number of clusters
    K = gamma.shape[1]
    nrows, ncols = _nrows_ncols(K)

    # Initialise plot 1
    fig1, ax1 = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, sharey=True)
    axes1 = ax1.reshape(-1)

    # Iterate over clusters
    for k in range(K):

        # first version of attention map computation
        gamma_k = gamma[:, k, :]
        heatmap_k = _get_attention_v1(alpha, beta, gamma_k)

        # Plot map
        # sns.heatmap()

    return None


def _get_attention_v1(alpha, beta, gamma_k):
    """
    Attention maps computation version 1.

    Params:
    - alpha: attention weights of shape (N, T, D_f)
    - beta: attention weights of shape (1, T, 1)
    - gamma_k: cluster k attention weights of shape (N, D_f)

    Returns:
    - heatmaps array of shape: (K, T, D_f) with the resulting average attention weights.
    """
    # Get time multiplication
    per_pat_time_feat_map = np.multiply(alpha, beta)                      # shape (N, T, D_f)

    # Approximate by cluster
    cohort_attention = np.multiply(per_pat_time_feat_map, np.expand_dims(gamma_k, axis=1))       # shape (N, T, D_f)

    return np.mean(cohort_attention, axis=0)



def get_basic_info(save_fd: str = None, data_info: dict = None):
    """
    Given input data information, given in data_info, and save_fd, re-compute save_fd for visualisation folder,
    save config and get data_name.

    Params:
    - save_fd: str, where results are currently saved. If save_fd is None, return None. If not None, then re-compute
    save_fd to the visualisation folder.
    - data_info: dict, with input data basic information.

    Returns:
    - save_fd: re-computed save_fd for visualisation.
    - data_config: list of data_configuration parameters for this input data.
    - data_name: data_name of this dataset.
    """

    # Data configuration and name
    data_load_config = data_info["data_load_config"]
    data_name = data_load_config["data_name"]

    # Re-compute save_fd if it was provided
    if save_fd is not None:

        # Recompute save_fd and makedirs if it does not exist
        _, model_name, run_num, _ = save_fd.split("/")
        save_fd = f"visualisations/{data_name}/{model_name}/{run_num}/"

        if not os.path.exists(save_fd):
            os.makedirs(save_fd)

        # Consider if it exists and require it to be the same
        if os.path.exists(save_fd + "data_load_config.json"):
            with open(save_fd + "data_load_config.json", "r") as f:
                save_config = json.load(f)
                assert save_config == data_load_config

        # Save if it doesn't exist
        else:
            with open(save_fd + "data_load_config.json", "w+", newline="\n") as f:
                json.dump(data_load_config, f, indent=4)
                f.close()

    return save_fd, data_load_config, data_name



def plot_loss_fn(train_values: List, val_values: List):
    """
    Plot loss_function evolution during training given loss_values during training and validation.

    Params:
    - train_values: list object with loss values on training data.
    - val_values: list object with loss values on validation data.

    Returns:
    - Tuple (Fig, ax) of figure, plt.ax objects with corresponding plot of loss_values.
    """
    # Compute Length of list
    N = len(train_values)

    # Initialise plot
    fig, ax = plt.subplots()
    ax.plot(range(1, N + 1), train_values, color="b")
    ax.plot(range(1, N + 1), val_values, color="tab:orange")

    # Add labels
    ax.set_xlabel("Epochs")

    return fig, ax


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
    _, static_vars, time_vars = separate_vars_by_type(feats)

    # Compute per-group information
    group_data_dic = get_data_per_group(input_data, groups_df, id_col)

    # Make summary statistics of static variables
    summary_info = make_summary_statistics(group_data_dic, static_vars, id_col)

    # Make temporal plots
    nrows, ncols = _nrows_ncols(len(time_vars))
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, sharey=False)
    ax = make_temporal_trajs(group_data_dic, time_vars, id_col, time_col, ax=ax)
    plt.show()

    return summary_info, (fig, ax)


def separate_vars_by_type(feats: List) -> Tuple[List, List, List]:
    """
    Separate features into id features, time variables and static variables.

    Params:
    - feats: list of str objects, with name of all variables in dataset.

    Returns:
        - Tuple (id_vars, static_vars, time_vars) with the corresponding variables in each element of the tuple.
    """

    # Get id_vars
    id_vars = [feat for feat in feats if _is_id_feat(feat)]

    # Get static_vars
    static_vars = [feat for feat in feats if _is_static_feat(feat)]

    # Get time_vars
    time_vars = [feat for feat in feats if _is_temporal_feat(feat)]

    return id_vars, static_vars, time_vars


def _get_ids_in_group(groups_df: pd.DataFrame, group: Union[int, str]) -> np.array:
    """
    Compute ids of cohort assigned to a particular group. Ids are identified by the group for which their likelihood
    is highest, or where the group where they have been one-hot encoded to.

    Params:
    - groups_df: pd.DataFrame of shape (N, num_groups) with one-hot encoded membership of group assignment, or
    group assignment probability.
    - group: str/int. Particular group being considered. group is a column of groups_df.

    Returns:
        ids_in_group: np. array of ids which are present in the particular group.
    """

    # Subset to ids with group labels
    subset = groups_df[groups_df.idxmax(axis=1) == group]

    # Compute index to obtain ids in group
    is_in_group = subset.index.values

    return is_in_group


def get_data_per_group(x_data: pd.DataFrame, groups_df: pd.DataFrame, id_col: str) -> dict:
    """
    Divide input data into data derived from sub-cohorts as specified by groups_df. This is saved in a dictionary
    indexed by the group names.

    Params:
    - x_data: pd.DataFrame with input data, and patient id given by id_col.
    - groups_df: pd.DataFrame with group membership data. Group names are specified in the group columns.
    - id_col: str, which column of x_data indicates patient id.
    """

    # Get list of groups
    group_names = groups_df.columns.tolist()

    # Iterate over groups
    output_dic = {}
    for group_id, group in enumerate(group_names):
        # Compute those ids in this group
        ids_in_group = _get_ids_in_group(groups_df, group)

        # Subset
        data_in_group = x_data[x_data[id_col].isin(ids_in_group)]

        # Append to output dictionary
        output_dic[group] = data_in_group

    return output_dic


def make_summary_statistics(group_data_dic: Union[pd.DataFrame, dict], feats: Union[str, List[str]], id_col: str):
    """
    Make list of summary statistics of those variables defined by features given input

    Params:
    - group_data_dic: dictionary with sub-cohort input data, or pd.DataFrame with the whole cohort.
    - feats: str or list of str, indicating which features to consider for description.
    - id_col: str, name of main id column for input data.

    Returns:
        - pd.DataFrame with summary statistics per group for each feature in feats.
        - print information.
    """

    # Convert input data to dict if all data (no sub-cohort analysis)
    if not isinstance(group_data_dic, dict):
        assert isinstance(group_data_dic, pd.DataFrame)
        group_data_dic = dict(all_cohort=group_data_dic)

    # Useful definition
    groups = list(group_data_dic.keys())

    # Initialise output array
    output = pd.DataFrame(np.empty(shape=(1, len(groups))), columns=groups)

    # Iterate over features
    for feat in feats:

        # Iterate over groups
        for group_id, (group, group_data) in enumerate(group_data_dic.items()):

            # Compute feature summary values - if static, take only first value, if temporal (or any other) take all.
            if _is_static_feat(feat):
                feat_dist = group_data.groupby(id_col).apply(lambda x: x[feat].iloc[0]).values

            else:
                if not _is_id_feat(feat):  # ensuring id cols not used for computations.
                    feat_dist = group_data[feat].values

            assert not np.any(np.isnan(feat_dist))

            # Compute mean, std
            mean, std = np.nanmean(feat_dist), np.nanstd(feat_dist)

            # Compute min, 25%, median, 75%, max
            q_0, q_25, q_50, q_75, q_100 = np.nanquantile(feat_dist, q=[0, 0.25, 0.5, 0.75, 1.0])

            # Append to output
            output.loc[f"{feat} mean", group] = f"{mean:.2f} \u00B1 {std:.2f}"
            output.loc[f"{feat} IQR", group] = f"{q_50:.2f} ({q_25:.2f} - {q_75:.2f}"
            output.loc[f"{feat} min-max", group] = f"{q_0:.2f} - {q_100:.2f}"

    return output


def _nrows_ncols(N: int) -> Tuple[int, int]:
    """
    Compute the number of rows and cols in plot given number of variables, given by N.

    Params:
    - N: int.

    Returns:
    - nrows, ncols: tuple of integers.
    """
    if N == 1:
        nrows, ncols = 1, 1

    elif 1 < N <= 10:
        nrows = 2
        ncols = int(np.ceil(N / 2))

    elif N > 10:
        nrows = int(np.floor(N / 4))
        ncols = int(np.ceil(N / nrows))

    else:
        raise ValueError(f"N is not a valid format. Value {N} was passed.")

    return nrows, ncols


def make_temporal_trajs(group_data_dic, feats, id_col, time_col, ax=None):
    """
    Plot Mean temporal trajectories with standard error on ax given input data, feats list and list of ids.

    Params:
    - group_data_dic: pd.DataFrame (if all cohort) or dic with input data for each sub-cohort.
    - feats: str or list of str, indicating which variables to plot.
    - id_col: str, identifier of patient. Must be element of columns of each input data dataframe.
    - time_col: str, identifier of time. Must be element of columns of each input data dataframe.

    Returns:
        - ax object with each subplot computing average trajectory per each sub-cohort.
    """
    # Convert input data to dict if all data (no sub-cohort analysis)
    if not isinstance(group_data_dic, dict):
        assert isinstance(group_data_dic, pd.DataFrame)
        group_data_dic = dict(all_cohort=group_data_dic)

    if ax is None:
        ax = plt.gca()
    axes = ax.reshape(-1)

    # Iterate through features
    for feat_id, feat in enumerate(feats):

        # Iterate through each group
        for group_id, (group, group_data) in enumerate(group_data_dic.items()):
            # Group properties
            group_color = colors[group_id]
            group_label = f"{group} - (N = {group_data[id_col].nunique()})"

            # Compute mean and sterror trajectories
            time_ids, feat_mean = _compute_mean(group_data, time_col=time_col, feat=feat)
            _, feat_sterror = _compute_sterror(group_data, time_col=time_col, id_col=id_col, feat=feat)

            axes[feat_id].plot(time_ids, feat_mean, linestyle="-", color=group_color, label=group_label)
            axes[feat_id].plot(time_ids, feat_mean + feat_sterror, linestyle="--", color=group_color)
            axes[feat_id].plot(time_ids, feat_mean - feat_sterror, linestyle="--", color=group_color)

        # Add information
        axes[feat_id].set_xlabel("Time to Endpoint (h)")

        try:
            unit = UNITS_DIC[feat]
        except KeyError:
            unit = "-"

        axes[feat_id].set_ylabel(f"{feat} {unit}")

    # Invert axes and add legend
    axes[0].invert_xaxis()
    axes[0].legend()

    return axes


def _compute_mean(data: pd.DataFrame, time_col: str, feat: str) -> Tuple:
    """
    Compute NanMean of feature over time given input data.

    Params:
    - data: pd.DataFrame with input data.
    - time_col: str, which col of data is the time identifier.
    - feat: str, which feature to compute mean for.

    Returns: Tuple of time id and mean values.
    - np. Array of time indices.
    - np. Array of mean values for feature.
    """

    # Compute nanmean
    nanmean = data.groupby(time_col).apply(lambda x: np.nanmean(x[feat]))
    time_ids = nanmean.index

    return time_ids, nanmean.values


def _compute_sterror(data: pd.DataFrame, time_col: str, id_col: str, feat: str) -> Tuple:
    """
    Compute NanSterror of feature over time given input data.

    Params:
    - data: pd.DataFrame with input data.
    - time_col: str, which col of data is the time identifier.
    - id_col: str, which col of data is the main patient identifier.
    - feat: str, which feature to compute mean for.

    Returns: Tuple of time id and sterror values
    - np. Array of time indices.
    - np. Array of sterror values for feature.
    """

    # Compute nanmean
    nansterror = data.groupby(time_col).apply(lambda x: np.nanstd(x[feat])) / np.sqrt(data[id_col].nunique())
    time_ids = nansterror

    return time_ids, nansterror.values


def get_dists_per_clus(pis_pred):
    """
    Visualise distribution information per each cluster membership given outcome likelihood probabilities y_pred.

    Params:
    - y_pred: array-like of shape (N, num_clus).

    Returns:
    - (fig, ax) object with IQR distributions of cluster_assignment for patients in a given cluster.
    """
    sns.set_theme(style="whitegrid")

    if isinstance(pis_pred, pd.DataFrame):
        pis_pred = pis_pred.values

    # Get cluster assignments
    clus_pred = np.argmax(pis_pred, axis=1)
    N = pis_pred.shape[-1]

    # Initialise plot
    nrows, ncols = _nrows_ncols(N)
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, sharey=True)
    axes = ax.reshape(-1)

    # Iterate through each cluster
    for clus_id, clus in enumerate(range(N)):
        # Get ids for this cluster
        clus_subset = pis_pred[clus_pred == clus]

        # Plot IQR ranges
        axes[clus_id].boxplot(x=clus_subset, labels=range(1, N+1))
        axes[clus_id].set_title(f"Clus = {clus}")

    return fig, axes
