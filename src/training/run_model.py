#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Python script to run a model with a target dataset.
"""

#%% Import Libraries

import os
from src.training.data_loader import data_loader as data_loader


#%% Load data

input_info = data_loader()

