#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 21 10:48:57 2021

@author: henrique.aguiar@ds.ccrg.kadooriecentre.org
"""

# Import Models
from src.models.deep_learning.camelot.model import Model as CamelotModel
from src.models.deep_learning.actpc.model import Model as ActpcModel
from src.models.deep_learning.enc_pred.model import Model as EncPredModel
from src.models.traditional_classifiers.svm_all import SVMAll
from src.models.traditional_classifiers.svm_per_feat import SVMFeat
from src.models.traditional_classifiers.xgb_all import XGBAll
from src.models.traditional_classifiers.xgb_per_feat import XGBFeat
from src.models.traditional_clustering.TSKM import TSKM
from src.models.traditional_classifiers.news import NEWS

import tensorflow as tf
import os

import absl.logging

absl.logging.set_verbosity(absl.logging.ERROR)


def get_model_from_str(data_info: dict, model_config: dict, training_config: dict):
    """
    Function to load correct model from the model name.

    Params:
    - data_info: dictionary with input data information.
    - model_config: model_configuration dictionary
    - training_config: model training configuration dictionary.

    returns: Corresponding model class object.
    """
    model_name = model_config["model_name"]
    gpu = training_config["gpu"] if "gpu" in training_config.keys() else None

    # Load the corresponding model
    if "camelot" in model_name.lower():

        # Check if GPU is accessible
        if gpu is None or gpu == 0:

            # Train only on CPU
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
            model = CamelotModel(data_info=data_info, model_config=model_config, training_config=training_config)

        # If GPU usage
        else:

            # Identify physical devices and limit memory growth
            physical_devices = tf.config.list_physical_devices('GPU')[0]
            print("\nPhysical Devices for Computation: ", physical_devices, sep="\n")
            tf.config.experimental.set_memory_growth(physical_devices, True)

            # If distributed strategy
            if gpu == "strategy":

                # Load strategy
                strategy = tf.distribute.MirroredStrategy(devices=None)

                with strategy.scope():
                    model = CamelotModel(data_info=data_info, model_config=model_config,
                                         training_config=training_config)

            else:
                model = CamelotModel(data_info=data_info, model_config=model_config, training_config=training_config)

    # ACTPC
    elif "actpc" in model_name.lower():

        # Check if GPU is accessible
        if gpu is None or gpu == 0:

            # Train only on CPU
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
            model = ActpcModel(data_info=data_info, model_config=model_config, training_config=training_config)

        # If GPU usage
        else:

            # Identify physical devices and limit memory growth
            physical_devices = tf.config.list_physical_devices('GPU')[0]
            print("\nPhysical Devices for Computation: ", physical_devices, sep="\n")
            tf.config.experimental.set_memory_growth(physical_devices, True)

            # If distributed strategy
            if gpu == "strategy":

                # Load strategy
                strategy = tf.distribute.MirroredStrategy(devices=None)

                with strategy.scope():
                    model = ActpcModel(data_info=data_info, model_config=model_config,
                                       training_config=training_config)

            else:
                model = ActpcModel(data_info=data_info, model_config=model_config, training_config=training_config)

    # Encoder Predictor Model
    elif "enc" in model_name.lower() and "pred" in model_name.lower():

        # Check if GPU is accessible
        if gpu is None or gpu == 0:

            # Train only on CPU
            # os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
            model = EncPredModel(data_info=data_info, model_config=model_config, training_config=training_config)

        # If GPU usage
        else:

            # Identify physical devices and limit memory growth
            physical_devices = tf.config.list_physical_devices('GPU')[0]
            print("\nPhysical Devices for Computation: ", physical_devices, sep="\n")
            tf.config.experimental.set_memory_growth(physical_devices, True)

            # If distributed strategy
            if gpu == "strategy":

                # Load strategy
                strategy = tf.distribute.MirroredStrategy(devices=None)

                with strategy.scope():
                    model = EncPredModel(data_info=data_info, model_config=model_config,
                                         training_config=training_config)

            else:
                model = EncPredModel(data_info=data_info, model_config=model_config, training_config=training_config)

    elif "svm" in model_name.lower() and "all" in model_name.lower():
        model = SVMAll(data_info=data_info, **model_config)

    elif "svm" in model_name.lower() and "feat" in model_name.lower():
        model = SVMFeat(data_info=data_info, **model_config)

    elif "xgb" in model_name.lower() and "all" in model_name.lower():
        model = XGBAll(data_info=data_info, **model_config)

    elif "xgb" in model_name.lower() and "feat" in model_name.lower():
        model = XGBFeat(data_info=data_info, **model_config)

    elif "tskm" in model_name.lower():
        model = TSKM(data_info=data_info, **model_config)

    elif "news" in model_name.lower():
        model = NEWS(data_info=data_info, **model_config)

    else:
        raise ValueError(f"Correct Model name not specified. Value {model_name} given.")

    return model
