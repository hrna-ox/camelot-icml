"""
Auxiliary Functions for Main Visualisation Make Plots.py file
"""

import numpy as np
from matplotlib.cm import get_cmap

from src.data_processing.data_loading_utils import get_ids


class GroupVisualiser:
    """
    Data Visualiser class for analysing input data.
    """
    def __init__(self, X_og, y_og, features, outcomes, static_feats: list = ["age", "gender", "ESI", "CCI"],
                 data_name = "MIMIC"):
        """
        Params:
        - X_og: Input Dataframe of shape (NxT, D_f + 2).
        - y_og: Array of target outcomes of shape (N, num_outcs) in one-hot encoded format.
        - features: list of features of observations X_og
        - outcomes: list of outcome names for each column of y_og
        - static_feats: list of static features of input data X (default = ["age", "gender", "ESI"])
        - data_name: str, name of dataset.

        Returns:
            Data Visualiser Object.
        """

        # Load data
        self.data = X_og
        self.targets = y_og

        # Useful definitions
        self.feats, self.static_feats = features, static_feats
        self.time_feats = [feat for feat in self.feats if feat not in self.static_feats]
        self.outc_names = outcomes

        # Get ids
        self.id_col, _ , _ = get_ids(data_name)
        self.time_col = "time_to_end"

        # Other plotting properties
        self.colors = get_cmap("tab10").colors

        # Get groups defined by labels
        self.groups = self._get_groups()

    def _get_groups(self):
        """
        Obtain groups as defined by labels.

        returns:
        - Dictionary attribute with corresponding group data.
        """
        groups = get_subgroups_from_labels(self.data, self.targets, label_names=self.outc_names, id_col=self.id_col)

        return groups

    def describe_static_groups(self):
        """
        Compute Descriptive information per each static variable
        """
        pass

    def plot_time_groups(self, ax=None):
        """
        Plot group average over time.

        Returns:
            axs object with corresponding mean plots for each time feature.
        """
        # Get list of ids
        time_ids = self.data[self.time_col].unique()
        axs = ax.reshape(-1)

        # Iterate through time features
        for feat_id, feat in enumerate(self.time_feats):

            # Iterate through dictionary
            for group_id, group in enumerate(self.groups.keys()):

                # Compute average
                data = self.groups[group]
                mean = get_time_mean(data, feat, self.time_col)
                sterror = get_time_sterror(data, feat, self.time_col)

                # Add to axis
                axs[feat_id].plot(time_ids, mean, label=group, color=self.colors[group_id], linestyle="-")
                axs[feat_id].plot(time_ids, mean + sterror, color=self.colors[group_id], linestyle="--", alpha=0.5)
                axs[feat_id].plot(time_ids, mean - sterror, color=self.colors[group_id], linestyle="--", alpha=0.5)

            axs[feat_id].set_ylabel(feat)

        return axs

    def describe_time_groups:
        pass


def _subset_ids(X, which_ids, id_col="pat_id"):
    """
    Select a subset of the input data based on a boolean array indicating patient subset to consider.

    Params:
    - X: array-like of shape (NxT, D_f + 2)
    - which_ids: array-like of shape (N, ) boolean indicating patients to consider.
    - id_col: name of X column indicating patient membership.

    Returns:
    - X_subset: dataframe of X subset data with ids subsetted as given by which_ids.
    """

    # Get list of ids
    all_ids = X[id_col].unique()
    subset_ids = all_ids[which_ids]

    # Subset data
    subset_data = X[X[id_col].isin(subset_ids)]

    return subset_data


def get_subgroups_from_labels(X, y, label_names=None, id_col="pat_id"):
    """
    Separate X into cohorts as identified by the labels y.

    Params:
    - X: array-like of shape (N x T, D_f + 2)
    - y: array-like of shape (N, num_outcs) with one-hot encoded labels.
    - label_names: list of column names for the targets. (default None, returns range.)
    - id_col: str, which column of X indicates patient membership

    Returns:
        - Dictionary indexed by label_names, of corresponding subgroup data and ids.
    """
    if label_names is None:
        label_names = range(1, y.shape[-1] + 1)

    # Initialise dictionary and iterate through labels
    output_dic = {}

    for label_id, label in enumerate(label_names):
        ids_have_label = y[:, label_id].astype(bool)              # Get boolean indicator for if ids have target label.

        # Subset large array
        data_subset_label = _subset_ids(X, ids_have_label, id_col=id_col)

        # Append to Output dictionary
        output_dic[label] = data_subset_label

    return output_dic

def get_time_mean(X, feat, time_col):
    """
    Compute temporal average of time feature feat in data X, according to time col given by time_col.

    Params:
    - X: Dataframe of shape (N x T, D_f + 2)
    - feat: str, feature to compute average for.
    - time_col: str, column of X representing time column.

    Returns:
        - numpy array with corresponding mean.
    """

    # Compute mean
    mean = X.groupby(time_col).apply(lambda x: np.nanmean(x[feat])).values

    return mean


def get_time_sterror(X, feat, time_col):
    """
    Compute temporal sterror of time feature feat in data X, according to time col given by time_col.

    Params:
    - X: Dataframe of shape (N x T, D_f + 2)
    - feat: str, feature to compute average for.
    - time_col: str, column of X representing time column.

    Returns:
        - numpy array with corresponding mean.
    """

    # Compute mean
    sterror = X.groupby(time_col).apply(lambda x: np.nanstd(x[feat]) / np.sqrt(x.shape[0])).values

    return sterror
