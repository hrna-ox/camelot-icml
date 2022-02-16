"""
Single Run Training

Date Last updated: 24 Jan 2022
Author: Henrique Aguiar
Please contact via henrique.aguiar@eng.ox.ac.uk
"""
import json

import numpy as np

from src.data_processing.data_loader import data_loader
import src.models.model_utils as model_utils
from src.results.main import evaluate
from src.visualisation.main import make_group_summaries

# physical_devices = tf.config.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(physical_devices[0], True)

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# ---------------------------- Load Configurations --------------------------------------
with open("src/training/data_config.json", "r") as f:
    data_config = json.load(f)
    f.close()

with open("src/training/model_config.json", "r") as f:
    model_config = json.load(f)
    f.close()

with open("src/training/training_config.json", "r") as f:
    training_config = json.load(f)
    f.close()

# ----------------------------- Load Data and Plot summary statistics -------------------------------

"Data Loading."
data_info = data_loader(**data_config)
output_dim = data_info["y"][-1].shape[-1]

"Visualise Data Properties"
input_data, groups = data_info["X_og"], data_info["y_og"]
time_col, id_col = data_info["time_col"], data_info["id_col"]
data_outcome_avg, ax1 = make_group_summaries(input_data, groups, time_col=time_col, id_col=id_col)


# -------------------------- Loading and Training Model -----------------------------

"Load model and fit"
print("\n\n\n\n")
model = model_utils.get_model_from_str(**model_config)
model.train(data_info=data_info, **training_config)

"Compute results on test data"
outputs_dic = model.analyse(data_info)

"Evaluate scores on the resulting models."

# Get X_test array pre-normalisation
x_test_norm, norm_min, norm_max = data_info["X"][-1], data_info["norm_min"], data_info["norm_max"]
X_test = np.multiply(x_test_norm, (norm_max - norm_min)) + norm_min

scores = evaluate(X_test, **outputs_dic, avg=None)


# ------------------------ Results Visualisations --------------------------
"Learnt Group averages"

# Get original data subsetted only to test set
ids_test = data_info["ids"][-1][:, 0]
data_test = input_data[input_data[id_col].isin(ids_test)]

# Make summaries
clus_group_avg, ax2 = make_group_summaries(data_test, groups, time_col=time_col, id_col=id_col)

"Losses where relevant"

"Clus assignments where relevant"

"Attention maps where relevant"
