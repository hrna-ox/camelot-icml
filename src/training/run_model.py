"""
Single Run Training

Date Last updated: 24 Jan 2022
Author: Henrique Aguiar
Please contact via henrique.aguiar@eng.ox.ac.uk
"""
import json

from src.data_processing.data_loader import data_loader
import src.models.model_utils as model_utils
from src.results.main import evaluate
import src.visualisation.main as vis_main

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
model_config["output_dim"] = data_info["y"][-1].shape[-1]
data_name = data_config["data_name"]

"Visualise Data Properties"
# vis_main.visualise_data_groups(data_info)

# -------------------------- Loading and Training Model -----------------------------

"Load model and fit"
print("\n\n\n\n")
model = model_utils.get_model_from_str(**model_config)
history = model.train(data_info=data_info, **training_config)

"Compute results on test data"
outputs_dic = model.analyse(data_info)


# -------------------------------------- Evaluate Scores --------------------------------------

"Evaluate scores on the resulting models. Note X_test is converted back to input dimensions."
# scores = evaluate(**outputs_dic, data_info=data_info, avg=None)

# ------------------------ Results Visualisations --------------------------
"Learnt Group averages"

# Get original data subsetted only to test set
# vis_main.visualise_cluster_groups(**outputs_dic, data_info=data_info)

# "Losses where relevant"
# vis_main.plot_losses(history=history, **outputs_dic, data_info=data_info)

# "Clus assignments where relevant"
vis_main.visualise_cluster_assignment(**outputs_dic, data_info=data_info)

# "Attention maps where relevant"
vis_main.visualise_attention_maps(**outputs_dic, data_info=data_info)
