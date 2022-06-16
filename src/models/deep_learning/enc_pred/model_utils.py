#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Loss, Metrics and Callback functions to use for model

@author: henrique.aguiar@ds.ccrg.kadooriecentre.org
"""
import os

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.callbacks as cbck
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics import roc_auc_score as roc

# ----------------------------------------------------------------------------------
"Utility Functions and Global Params"

LOGS_DIR = "experiments/ENCPRED/"


def tf_log(tensor):
    return tf.math.log(tensor + 1e-8)


def class_weighting(y_true):
    """
    Function to compute inverse class proportion weighting given array of true class assignments.

    Params:
    - y_true: array-like of shape (N, num_outcs) with corresponding one-hot encoding assignment of true class

    Returns:
    - weights: array-like of shape (num_outcs) with weights inversely proportional to number of true class examples.
    """

    class_numbers = tf.reduce_sum(y_true, axis=0)

    # Check no class is missing
    if not tf.reduce_all(class_numbers > 0):
        class_numbers += 1

    # Compute reciprocal
    inv_class_numbers = 1 / class_numbers

    return inv_class_numbers / tf.reduce_sum(inv_class_numbers)


# ------------------------------------------------------------------------------------
"""Loss Functions"""


def l_pred(y_true, y_pred, weights=None, name='pred_clus_loss'):
    """
    Negative weighted predictive clustering loss. Computes Cross-entropy between categorical y_true and y_pred.
    This is minimised when y_pred matches y_true.

    Params:
    - y_true: array-like of shape (bs, num_outcs) of one-hot encoded true class.
    - y_pred: array-like of shape (bs, num_outcs) of probability class predictions.
    - weights: array-like of shape (num_outcs) of weights to multiply cross-entropy terms. (default None).
    - name: name to give to operation.

    If weights is None, defaults to regular cross-entropy calculation.

    Returns:
    - loss_value: score indicating corresponding loss.
    """

    # If weights is None, return weights as set of 1s.
    if weights is None:
        weights = tf.cast(tf.constant(np.ones(shape=y_true.shape) / y_true.shape[-1]), dtype=np.float32)

    # Compute batch
    batch_neg_ce = - tf.reduce_sum(weights * y_true * tf_log(y_pred))

    # Average over batch
    loss_value = tf.reduce_mean(batch_neg_ce, name=name)

    return loss_value


# ----------------------------------------------------------------------------------
"Callback methods to update training procedure."


class ConfusionMatrix(cbck.Callback):
    """
    Callback method to print Confusion Matrix over data.

    Output is a matrix indicating the amount of patients assigned to a target class and with a certain true class.

    Params:
    - validation_data: tuple of X_val, y_val data
    - interval: interval between epochs on which to print values. (default = 5)
    """

    def __init__(self, validation_data: tuple, interval: int = 5):
        super().__init__()
        self.interval = interval
        self.X_val, self.y_val = validation_data

        # Compute number of outcomes
        self.C = self.y_val.shape[-1]

    def on_epoch_end(self, epoch, *args, **kwargs):

        # Print information if matches interval epoch length
        if epoch % self.interval == 0:

            # Initialise output Confusion matrix
            cm_output = np.zeros(shape=(self.C, self.C))

            # Compute prediction and true values in categorical format.
            y_pred = (self.model(self.X_val)).numpy()
            class_pred = np.argmax(y_pred, axis=-1)
            class_true = np.argmax(self.y_val, axis=-1)

            # Iterate through classes
            for true_class in range(self.C):
                for pred_class in range(self.C):
                    num_samples = np.logical_and(class_pred == pred_class, class_true == true_class).sum()
                    cm_output[true_class, pred_class] = num_samples

            # Print as pd.dataframe
            index = [f"TC{class_}" for class_ in range(1, self.C + 1)]
            columns = [f"PC{class_}" for class_ in range(1, self.C + 1)]

            cm_output = pd.DataFrame(cm_output, index=index, columns=columns)

            print("End of Epoch {:d} - Confusion matrix: \n {}".format(epoch, cm_output.astype(int)))


class AUROC(cbck.Callback):
    """
    Callback method to display AUROC value for predicted y.

    Params:
    - validation_data: tuple of X_val, y_val data
    - interval: interval between epochs on which to print values. (default = 5)
    """

    def __init__(self, validation_data: tuple, interval: int = 5):
        super().__init__()
        self.interval = interval
        self.X_val, self.y_val = validation_data

    def on_epoch_end(self, epoch, *args, **kwargs):
        if epoch % self.interval == 0:
            # Compute predictions
            y_pred = self.model(self.X_val).numpy()

            # Compute ROC
            roc_auc_score = roc(y_true=self.y_val, y_score=y_pred,
                                average=None, multi_class='ovr')

            print("End of Epoch {:d} - OVR ROC score: {}".format(epoch, roc_auc_score))


class SupervisedTargetMetrics(cbck.Callback):
    """
    Callback method to display supervised target metrics: Normalised Mutual Information, Adjusted Rand Score and
    Purity Score

    Params:
    - validation_data: tuple of X_val, y_val data
    - interval: interval between epochs on which to print values. (default = 5)
    """

    def __init__(self, validation_data: tuple, interval: int = 5):
        super().__init__()
        self.interval = interval
        self.X_val, self.y_val = validation_data

    def on_epoch_end(self, epoch, *args, **kwargs):
        if epoch % self.interval == 0:

            # Compute y_pred, y_true in categorical format.
            model_output = (self.model(self.X_val)).numpy()
            class_pred = np.argmax(model_output, axis=-1)
            class_true = np.argmax(self.y_val, axis=-1).reshape(-1)

            # Target metrics
            nmi = normalized_mutual_info_score(labels_true=class_true, labels_pred=class_pred)
            ars = adjusted_rand_score(labels_true=class_true, labels_pred=class_pred)

            print("End of Epoch {:d} - NMI {:.2f} , ARS {:.2f}".format(epoch, nmi, ars))


def cbck_list(summary_name: str, interval: int = 5, validation_data: tuple = ()):
    """
    Shorthand for callbacks above.

    Params:
    - summary_name: str containing shorthands for different callbacks.
    - interval: int interval to print information on.
    """
    extra_callback_list = []

    if "auc" in summary_name.lower() or "roc" in summary_name.lower():
        extra_callback_list.append(AUROC(interval=interval, validation_data=validation_data))

    if "cm" in summary_name.lower() or "conf_matrix" in summary_name.lower():
        extra_callback_list.append(ConfusionMatrix(interval=interval, validation_data=validation_data))

    if "sup_scores" in summary_name.lower():
        extra_callback_list.append(SupervisedTargetMetrics(interval=interval, validation_data=validation_data))

    return extra_callback_list


def get_callbacks(validation_data, data_name: str, track_loss: str, interval: int = 5, other_cbcks: str = "",
                  early_stop: bool = True, lr_scheduler: bool = True, tensorboard: bool = True,
                  min_delta: float = 0.0001, patience_epochs: int = 200):
    """
    Generate complete list of callbacks given input configuration.

    Params:
        - validation_data: tuple (X, y) of validation data.
        - data_name: str, data name on which the model is running
        - track_loss: str, name of main.py loss to keep track of.
        - interval: int, interval to print information on.
        - other_cbcks: str, list of other callbacks to consider (default = "", which selects None).
        - early_stop: whether to stop training early in case of no progress. (default = True)
        - lr_scheduler: dynamically update learning rate. (default = True)
        - tensorboard: write tensorboard friendly logs which can then be visualised. (default = True)
        - min_delta: if early stopping, the interval on which to check improvement or not.
        - patience_epochs: how many epochs to wait until checking for improvements.
        """

    # Initialise empty
    callbacks = []

    # Handle saving paths and folders
    logs_dir = LOGS_DIR
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)

    # Save Folder is first run that has not been previously computed
    run_num = 1
    while os.path.exists(logs_dir + f"{data_name}/run{run_num}/"):
        run_num += 1

    # Save as new run
    save_fd = logs_dir + f"{data_name}/run{run_num}/"
    assert not os.path.exists(save_fd)

    os.makedirs(save_fd)
    os.makedirs(save_fd + "logs/")
    os.makedirs(save_fd + "training/")

    # ------------------ Start Loading callbacks ---------------------------

    # Load custom callbacks first
    callbacks.extend(cbck_list(other_cbcks, interval, validation_data=validation_data))

    # Model Weight saving callback
    checkpoint = cbck.ModelCheckpoint(filepath=save_fd + "models/checkpoints/epoch-{epoch}", save_best_only=True,
                                      monitor=track_loss, save_freq="epoch")
    callbacks.append(checkpoint)

    # Logging Loss values)
    csv_logger = cbck.CSVLogger(filename=save_fd + "training/loss_tracker", separator=",", append=False)
    callbacks.append(csv_logger)

    # Check if Early stoppage is added
    if early_stop is True:
        callbacks.append(cbck.EarlyStopping(monitor='val_' + track_loss, mode="min", restore_best_weights=True,
                                            min_delta=min_delta, patience=patience_epochs))

    # Check if LR Scheduling is in place
    if lr_scheduler is True:
        callbacks.append(cbck.ReduceLROnPlateau(monitor='val_' + track_loss, mode='min', cooldown=15,
                                                min_lr=0.00001, factor=0.25))

    # Check if Tensorboard is active
    if tensorboard is True:
        callbacks.append(cbck.TensorBoard(log_dir=save_fd + "logs/", histogram_freq=1))

    return callbacks, run_num
