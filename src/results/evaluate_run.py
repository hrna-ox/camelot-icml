"""
File to compute evaluation scores of a single model run.

Author: Henrique Aguiar
email: henrique.aguiar@eng.ox.ac.uk
"""

import os, json
import argparse

import pandas as pd

from src.data_processing.data_loader import data_loader
from src.results.main import evaluate


parser = argparse.ArgumentParser()
parser.add_argument("model_name", type=str, default=None, help="model name for model used in this particular "
                                                               "experiment.")
parser.add_argument("run_num", type=int, default=1, help="number of experiment number for this model.")
parser = parser.parse_args()

DATA_LOADING_DIC_KEYS = []

# ---------- Load objects related to run ------------------------
model_name, run_num = parser.model_name.upper(), parser.run_num
save_fd = f"results/{model_name}/{run_num}/"
assert os.path.exists(save_fd)

# Load params
with open(save_fd + "config.json", "r") as f:
    run_params = json.load(f)
    f.close()

# Load other objects
y_true = pd.read_csv(save_fd + "y_true.csv", index_col=0, header=0)
y_pred = pd.read_csv(save_fd + "y_pred.csv", index_col=0, header=0)

try:
    clus_pred = pd.read_csv(save_fd + "clus_pred.csv", index_col=0, header=0)
except FileNotFoundError:
    clus_pred = None


# Load data to obtain input data information
data_loading_params = run_params["data_load_config"]
data_info = data_loader(**data_loading_params)
X_og_3D = data_info["X_og_3D"]

# Evaluate results
scores = evaluate(y_true=y_true, y_pred=y_pred, X_og_3D=X_og_3D, clus_pred=clus_pred, avg=None)
