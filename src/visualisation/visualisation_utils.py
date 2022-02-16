"""

"""
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap

import json
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
