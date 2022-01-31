import numpy as np

import matplotlib.pyplot as plt
from src.training.data_loader import data_loader
from src.training.data_loader import MIMIC_DEFAULT_LOAD_CONFIG, HAVEN_DEFAULT_LOAD_CONFIG

data_config = MIMIC_DEFAULT_LOAD_CONFIG

data_info = data_loader(data_config)

# Get whole data
X = np.concatenate(data_info["X"], axis = 0)
y = np.concatenate(data_info["y"], axis = 0)
ids = np.concatenate(data_info["id"], axis=0)

data = np.concatenate((ids, X), axis=-1)
feats = data_info["feats"]
time_feats = [feat in feats if feat not in ["ESI", "age", "gender"]]

# Plot Average Trajectories
fig, ax = plt.subplots(nrows = 3, ncols=2, sharex=True, sharey=False)
axs = ax.reshape(-1)
for feat_id, feat in enumerate(time_feats[2:]):

    # Make Plot
    avg_ = np.mean(data[:, :, feats.index(feat)], axis=0)
    sterror_ = np.std(data[:, :, feats.index(feat)], axis=0) / np.sqrt()
    axs[feat_id].plot()
