"""
Single Run Training

Date Last updated: 24 Jan 2022
Author: Henrique Aguiar
Please contact via henrique.aguiar@eng.ox.ac.uk
"""
import argparse

from src.models.camelot.model import Model
from src.training.data_loader import data_loader
from src.training.data_loader import MIMIC_DEFAULT_LOAD_CONFIG, HAVEN_DEFAULT_LOAD_CONFIG

# Data Loading Parameters
data_config = MIMIC_DEFAULT_LOAD_CONFIG
model_config = {"num_clusters": 12}
train_params = {""}

# Load data
data_info = data_loader(data_config)

# Load model and fit
model = Model(data_info, model_config)
model.fit()

# Compute data test
data_test = data_info["X"][-1], data_info["y"][-1]
model.evaluate(data_test)

# Print Model Run results