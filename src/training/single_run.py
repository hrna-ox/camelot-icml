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
default_data_config = {}
parser.add_argument('--data_config', default={}, type=dict, help="data configuration dictionary.")
parser.add_argument('--model_config', default={}, type=dict, help="model configuration.")
parser.add_argument('--train_params', default={}, type=dict, help="useful training parameters.")
parser.add_argument('--data_name', default="HAVEN", type=str, help="which dataset to load for.")
parser.add_argument('--id_col', default="subject_id", type=str, help="identifier column for patient information.")
parser.add_argument('--time_col', default="charttime", type=str, help="identifier for temporal column.")
parser.add_argument('--feat_set', default="vit-lab", type=str, help="feature set to consider for analysis.")
parser.add_argument('--time_range', default=(24, 72) , type=tuple, help="Min-Max values to subset for input.")
parser.add_argument('--include_time', default=False, type=bool, help="Whether to include time difference between observations as another feature.")
parser.add_argument('--train_test_ratio', default=0.4, type=float, help="ratio between train+val sets and all data.")
parser.add_argument('--train_val_ratio', default=0.6, type=float, help="ratio between train set and train + val.")
parser.add_argument('--seed', default=2323, type=int)


data_info = data_loader(data_config)


model = Model(data_info, model_config)
model.fit(train_params)

# Compute data test
data_test = data_info["X"][-1], data_info["y"][-1]
model.evaluate(data_test)

# Print Model Run results