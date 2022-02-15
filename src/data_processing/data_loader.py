#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Load data into configuration dictionary for use on later end models.
"""

from typing import Union, List, Tuple
import src.data_processing.data_loading_utils as data_utils

import numpy as np
from sklearn.model_selection import train_test_split


# %%  Load configuration for data loader

def data_loader(data_name: str = "MIMIC", feat_set: Union[List, str] = "vit", time_range: Tuple = (0, 6),
                target_window: int = 4, train_test_ratio: float = 0.6, train_val_ratio: float = 0.5, seed: int = 2323):
    """
    Data Loader function. Given data configuration, convert into input format.

    Params:
    - data_name: Which dataset to load. One of ["MIMIC", "COMPUTE", "HAVEN", "SAMPLE"]
    - feat_set: Indicates feature sets to load. If type list, then a list of the features to load. If
    of type string, then considers sets of variables. For instance, "vit" refers to vital signs, and loads
    Heart-Rate, Respiratory-Rate, SPO2, Temperature, etc...
    - time_range: tuple, time_window considered for time to discharge. This considers only a temporal subset of the
    available observations.
    - target_window: int, window interval on which to identify outcomes. Relevant only for MIMIC data.
    - train_test_ratio: float, ratio between training+val data and test data.
    - train_val_ratio: float, ratio between training data and validation data.
    - seed: int, seed for random generation.
    """

    # Load Data Processor Object
    data_processor = data_utils.DataProcessor(data_name=data_name, feat_set=feat_set, time_range=time_range,
                                              target_window=target_window)

    # Convert to input format, and keep track of useful information
    x, y, mask, ids, feats, outcomes, X_feat, X_feat_3D, y_outc = data_processor.load_transform()
    print(f"{data_name} data successfully loaded.")
    print(f"Basic information \n",
          f"Input shape: {x.shape}, {y.shape} \n Outcome Distribution: {y_outc.sum(axis=0)}")

    # Separate into train, val and test data
    X_train, X_test, y_train, y_test, id_train, id_test, mask_train, mask_test = train_test_split(
        x, y, ids, mask, train_size=train_test_ratio, random_state=seed, shuffle=True, stratify=np.argmax(y, axis=-1))

    X_train, X_val, y_train, y_val, id_train, id_val, mask_train, mask_val = train_test_split(
        X_train, y_train, id_train, mask_train, train_size=train_val_ratio, random_state=seed, shuffle=True,
        stratify=np.argmax(y_train, axis=-1))

    "Normalise Data, and apply normalisation factors to validation and test."
    X_train = data_processor.normalise(X_train)
    X_val = data_processor.apply_normalisation(X_val)
    X_test = data_processor.apply_normalisation(X_test)

    # Get min and max factors.
    min_, max_ = data_processor.min, data_processor.max

    # Get data_config
    data_config = {"data_name": data_name, "feat_set": feat_set, "time_range (h)": time_range, "window": 4,
                   "train-test-ratio": train_test_ratio, "train-val-ratio": train_val_ratio, "seed": seed}

    # Aggregate into output
    output = {"X": (X_train, X_val, X_test),
              "y": (y_train, y_val, y_test),
              "ids": (id_train, id_val, id_test),
              "mask": (mask_train, mask_val, mask_test),
              "feats": feats,
              "id_col": data_processor.id_col,
              "time_col": data_processor.time_col,
              "norm_min": min_,
              "norm_max": max_,
              "outcomes": outcomes,
              "X_og": X_feat,
              "y_og": y_outc,
              "X_og_3D": X_feat_3D,
              "data_load_config": data_config
              }

    return output
