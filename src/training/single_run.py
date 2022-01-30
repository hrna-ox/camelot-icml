"""
Single Run Training

Date Last updated: 24 Jan 2022
Author: Henrique Aguiar
Please contact via henrique.aguiar@eng.ox.ac.uk
"""
import argparse

from src.models.camelot.model import Model
from src.training.data_loader import data_loader

# Load arguments
parser = argparse.ArgumentParser()

# Data Loading Parameters
default_data_config = {
    "data_name": "HAVEN",
    "id_col": "subject_id",
    "time_col": "feat_set",
    "feat-set": "vit-lab",
    "time_range": (24, 72),
    "include_time": False,
    "train_test_ratio": 0.4,
    "train_val_ratio": 0.6,
    "seed": 2323
}

parser.add_argument('--data_config', default=default_data_config, type=dict, help="data configuration dictionary.")
parser.add_argument('--model_config', default={}, type=dict, help="model configuration.")
parser.add_argument('--train_params', default={}, type=dict, help="useful training parameters.")

# Load data
data_info = data_loader(data_config)

# Load model and fit
model = Model(data_info, model_config)
model.fit(train_params)

# Compute data test
data_test = data_info["X"][-1], data_info["y"][-1]
model.evaluate(data_test)

# Print Model Run results