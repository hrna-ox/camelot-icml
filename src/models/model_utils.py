#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 21 10:48:57 2021

@author: henrique.aguiar@ds.ccrg.kadooriecentre.org
"""

from src.models.deep_learning.camelot.model import Model as CamelotModel
from src.models.traditional_classifiers.svm_all import SVMAll
from src.models.traditional_classifiers.svm_per_feat import SVMFeat
from src.models.traditional_classifiers.xgb_all import XGBAll
from src.models.traditional_classifiers.xgb_per_feat import XGBFeat
from src.models.traditional_clustering.TSKM import TSKM
from src.models.traditional_classifiers.news import NEWS


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

    elif "svm" in model_name.lower() and "all" in model_name.lower():
        model = SVMAll(**kwargs)

    elif "svm" in model_name.lower() and "feat" in model_name.lower():
        model = SVMFeat(**kwargs)

    elif "xgb" in model_name.lower() and "all" in model_name.lower():
        model = XGBAll(**kwargs)

    elif "xgb" in model_name.lower() and "feat" in model_name.lower():
        model = XGBFeat(**kwargs)

    elif "tskm" in model_name.lower():
        model = TSKM(**kwargs)

    elif "news" in model_name.lower():
        model = NEWS(**kwargs)

    else:
        raise ValueError(f"Correct Model name not specified. Value {model_name} given.")

    return model
