"""
Main visualisation helper file for visualising results and data.
"""
import src.visualisation.vis_utils as utils
import matplotlib.pyplot as plt
import numpy as np

def plot_trajectories_per_data_group(data_info):
    """
    Plot Average Trajectories per Input Data Group.

    Params:
    - data_info: dictionary of data input data configuration.
    - output_results: dictionary of output results.

    Results:
    - Plot (and saved) of figure of data mean trajectories.
    """
    X, y = data_info["X_og"], data_info["y"]
    features, outcomes = data_info["feats"], data_info["outcomes"]
    data_name = data_info["data_loading_config"]["data_name"]

    # Initialise processor class
    data_visualiser = utils.GroupVisualiser(X_og=X, y_og=y, features=features, outcomes=outcomes, data_name=data_name)

    # Get figure
    nrows, ncols = int(np.ceil(len(features)/ 2)), 2
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, sharey=False)

    # Make plots
    ax = data_visualiser.plot_time_groups(ax=ax)

    # Decorate
    ax[0].set_xlabel("Time to End (h)")
    ax[0].invert_xaxis()

    plt.title("Plot of Average Trajectories over time with standard error.")
    plt.legend()
