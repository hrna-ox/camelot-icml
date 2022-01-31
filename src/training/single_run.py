"""
Single Run Training

Date Last updated: 24 Jan 2022
Author: Henrique Aguiar
Please contact via henrique.aguiar@eng.ox.ac.uk
"""
import tensorflow as tf
import os

from src.models.camelot.model import Model
from src.training.data_loader import data_loader
from src.training.data_loader import MIMIC_DEFAULT_LOAD_CONFIG, HAVEN_DEFAULT_LOAD_CONFIG

# physical_devices = tf.config.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(physical_devices[0], True)

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


# Data Loading Parameters
data_config = MIMIC_DEFAULT_LOAD_CONFIG
model_config = {"num_clusters": 8,
                "encoder_params": {
                    "hidden_layers": 1, "hidden_nodes": 30
                },
                "identifier_params": {
                    "hidden_layers": 1, "hidden_nodes": 30
                },
                "predictor_params": {
                    "hidden_layers": 1, "hidden_nodes": 30
                }
            }
train_params = {"epochs_init": 10,
                "epochs": 20,
                "lr": 0.001,
                "cbck_str": ""}

# Load data
data_info = data_loader(data_config)

# Load model and fit
print("NEED TO CHECK WHAT IS HAPPENING WITH LCLUS AND LPRED \n\n\n\n")
model = Model(data_info, model_config)
model.fit(**train_params)

# Compute data test
data_test = data_info["X"][-1], data_info["y"][-1]
model.evaluate(data_test)

# Print Model Run results