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

HAVEN_DEFAULT_LOAD_CONFIG = {
    "data_name": "HAVEN",
    "outcome_window": 0,
    "feat_set": "vit-lab-sta",
    "time_range": (24, 72),
    "include_time": False,
    "train_test_ratio": 0.4,
    "train_val_ratio": 0.6,
    "seed": 2323
}

MIMIC_DEFAULT_LOAD_CONFIG = {
    "data_name": "MIMIC",
    "outcome_window": 12,
    "feat_set": "vit-sta",
    "time_range": (0, 100),
    "include_time": False,
    "train_test_ratio": 0.7,
    "train_val_ratio": 0.6,
    "seed": 2323
}

def data_loader(data_config):
    """Data Loader function."""

    data_processor = data_utils.DataProcessor(**data_config)
    x, y, mask, ids, feats, outcomes, X_og, y_og = data_processor.load_transform()
    print(f"{data_config['data_name']} data successfully loaded.")

    # Separate into train, val and test data
    X_train, X_test, y_train, y_test, id_train, id_test, mask_train, mask_test = train_test_split(
        x, y, ids, mask,
        train_size=data_config["train_test_ratio"], random_state=data_config["seed"],
        shuffle=True, stratify=np.argmax(y, axis=-1))

    X_train, X_val, y_train, y_val, id_train, id_val, mask_train, mask_val = train_test_split(
        X_train, y_train, id_train, mask_train,
        train_size=data_config["train_val_ratio"], random_state=data_config["seed"],
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
              "data_load_config": data_config
              }

    return output
