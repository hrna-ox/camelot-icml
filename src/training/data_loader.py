#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Load data into configuration dictionary for use on later end models.
"""

import argparse
import src.training.data_loading_utils as data_utils

import numpy as np
from sklearn.model_selection import train_test_split

# %%  Load configuration for data loader 

parser = argparse.ArgumentParser()
# Data Loading Parameters
parser.add_argument('--data_name', default="HAVEN", type=str, help="which dataset to load for.")
parser.add_argument('--id_col', default="subject_id", type=str, help="identifier column for patient information.")
parser.add_argument('--time_col', default="charttime", type=str, help="identifier for temporal column.")
parser.add_argument('--feat_set', default="vit-lab", type=str, help="feature set to consider for analysis.")
parser.add_argument('--time_range', default=(24, 72) , type=tuple, help="Min-Max values to subset for input.")
parser.add_argument('--include_time', default=False, type=bool, help="Whether to include time difference between observations as another feature.")
parser.add_argument('--train_test_ratio', default=0.4, type=float, help="ratio between train+val sets and all data.")
parser.add_argument('--train_val_ratio', default=0.6, type=float, help="ratio between train set and train + val.")
parser.add_argument('--seed', default=2323, type=int)


# Data Loading Params
params = parser.parse_args()

# Data Loading     
data_load_config = {"data_name": params.data_name, 
                    "id_column": params.id_col,
                    "time_column": params.time_col, 
                    "feat_set": params.feat_set,
                    "time_range": params.time_range, 
                    "include_time": params.include_time,
                    "train_test_ratio": params.train_test_ratio,
                    "train_val_ratio": params.train_val_ratio,
                    "seed": params.seed}


# %% data Loading and processing

def data_loader():
    "Data Loader function."
    
    data_processor = data_utils.DataProcessor(**data_load_config)
    x, y, mask, ids, feats, outcomes, X_og, y_og = data_processor.load_transform()
    print(f"{params.data_name} data successfully loaded.")
    
    # Separate into train, val and test data
    X_train, X_test, y_train, y_test, id_train, id_test, mask_train, mask_test = train_test_split(
        x, y, ids, mask, 
        train_size=data_load_config["train_test_ratio"], random_state=data_load_config["seed"],
        shuffle=True, stratify=np.argmax(y, axis=-1))
    
    X_train, X_val, y_train, y_val, id_train, id_val, mask_train, mask_val = train_test_split(
        X_train, y_train, id_train, mask_train, 
        train_size=data_load_config["train_val_ratio"], random_state=data_load_config["seed"],
        shuffle=True, stratify=np.argmax(y_train, axis=-1))
    
    
    # Normalise and do the same for Validation and Test sets.
    X_train = data_processor.normalise(X_train)
    X_val = data_processor.apply_normalisation(X_val)
    X_test = data_processor.apply_normalisation(X_test)
    
    min_, max_ = data_processor.min, data_processor.max
    
    # Separate into train and validation test
    output_dim = y_train.shape[-1]
    
    # Aggregate into output
    output = {"X": (X_train, X_val, X_test),
              "y": (y_train, y_val, y_test),
              "id": (id_train, id_val, id_test),
              "mask": (mask_train, mask_val, mask_test),
              "feats": feats,
              "norm_min": min_,
              "norm_max": max_,
              "output_dim": output_dim,
              "data_load_config": data_load_config
              }
    
    return output

data_loader()["feats"]