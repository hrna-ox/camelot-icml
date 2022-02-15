#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 21 10:48:57 2021

@author: henrique.aguiar@ds.ccrg.kadooriecentre.org
"""

"""
SOME OF THESE ARE FOR EVALUATION.
"""


from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.metrics import roc_auc_score, f1_score, recall_score
import numpy as np
from sklearn.metrics.cluster import contingency_matrix

from src.models.deep_learning.camelot.model import Model as CamelotModel


def get_model_from_str(model_name, **kwargs):
    """
    Function to load correct model from the model name.

    Params:
    - model_name: name of model.
    - **kwargs: model initialisation parameters.

    returns: Corresponding model class object.
    """
    if "camelot" in model_name.lower():
        model = CamelotModel(**kwargs)

    else:
        raise ValueError(f"Correct Model name not specified. Value {model_str} given.")

    return model
