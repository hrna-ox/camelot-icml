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

import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay

# ----------------------------------------------------------------------------------
"Utility Functions and Global Params"

LOGS_DIR = "experiments/CAMELOT/"


def tf_log(tensor):
    return tf.math.log(tensor + 1e-8)


def tf_divide(tensor1, tensor2):
    return tf.math.divide(tensor1, tensor2 + 1e-8)


def np_log(array):
    return np.log(array + 1e-8)


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


def l_crit(y_true, y_pred, weights=None, name='pred_clus_loss'):
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


def l_clus(cluster_reps, name='embedding_sep_loss'):
    """Cluster representation separation loss. Computes negative euclidean distance summed over pairs of cluster 
    representation vectors. This loss is minimised as cluster vectors are separated 

    Params:
    - cluster_reps: array-like of shape (K, latent_dim) of cluster representation vectors.
    - name: name to give to operation.

    Returns:
    - norm_loss: score indicating corresponding loss.
    """

    # Expand input to allow broadcasting
    embedding_column = tf.expand_dims(cluster_reps, axis=1)  # shape (K, 1, latent_dim)
    embedding_row = tf.expand_dims(cluster_reps, axis=0)  # shape (1, K, latent_dim)

    # Compute pairwise Euclidean distance between cluster vectors, and sum over pairs of clusters.
    pairwise_loss = tf.reduce_sum((embedding_column - embedding_row) ** 2, axis=-1)
    loss = - tf.reduce_mean(pairwise_loss, axis=None, name=name)

    return loss


def l_clus_dist(clusters_prob, name="loss_clus_dist"):
    """
    Cluster distribution loss. Computes negative entropy of cluster distribution probability values.
    This is minimised when the cluster distribution is uniform (all clusters similar size).

    Params:
    - clusters_prob: array-like of shape (bs, K) of cluster_assignments distributions.
    - name: name to give to operation.

    Returns:
    - loss_value: score indicating corresponding loss.
    """

    # Calculate average cluster assignment distribution
    clus_avg_prob = tf.reduce_mean(clusters_prob, axis=0)

    # Compute negative entropy
    neg_entropy = tf.reduce_sum(clus_avg_prob * tf_log(clus_avg_prob), name=name)

    return neg_entropy


def l_pat_dist(clusters_prob):
    """
    Sample Cluster Entropy Loss. Computes negative entropy of cluster assignment over the batch.
    This is minimised when all samples are confidently assigned.

    Params:
    - clusters_prob: array-like of shape (bs, K) of cluster_assignments distributions.
    - name: name to give to operation.

    Returns:
    - loss_value: score indicating corresponding loss.
    """

    # Compute Entropy
    entropy = - tf.reduce_sum(clusters_prob * tf_log(clusters_prob), axis=-1)

    # Compute negative entropy
    batch_loss = tf.reduce_mean(entropy)

    return batch_loss


def l_dist(y_pred, true_dist):
    """
    Computes KL divergence between probability assignments and true outcome distribution.

    Params:
    - y_pred: tensor of shape (N, num_outcs) of outcome probability assignments.
    - true_dist: tensor of shape (num_outcs) of true outcome probability assignments.

    Returns:
    - loss_value: score indicating corresponding loss.
    """

    # Compute pred outcome distribution
    pred_dist = tf.reduce_mean(y_pred, axis=0)

    # Compute KL
    _log_divide = tf_log(tf_divide(pred_dist, true_dist))
    batch_loss = tf.reduce_sum(pred_dist * _log_divide)

    return batch_loss


# --------------- CALLBACK METHODS ----------------------
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


class PrCurves(cbck.Callback):
    """
    Callback method to display AUROC value for predicted y.

    Params:
    - validation_data: tuple of X_val, y_val data
    - interval: interval between epochs on which to print values. (default = 5)
    """

    def __init__(self, validation_data: tuple, interval: int = 5, save_fd: str = None):
        super().__init__()
        self.interval = interval
        self.X_val, self.y_val = validation_data
        self.C = self.y_val.shape[-1]
        self.save_fd = save_fd

    def on_epoch_end(self, epoch, *args, **kwargs):
        if epoch % self.interval == 0:
            # Compute predictions
            y_pred = self.model(self.X_val).numpy()

            # Initialise scores
            fig1, ax1 = plt.subplots(nrows=2, ncols=2)
            ax1 = ax1.reshape(-1)
            fig2, ax2 = plt.subplots(nrows=2, ncols=2)
            ax2 = ax2.reshape(-1)

            # Compute ROC
            roc, prc = np.zeros(self.C), np.zeros(self.C)
            for outc_ in range(self.C):

                try:

                    # Get roc and prc values
                    roc[outc_] = roc_auc_score(y_true=self.y_val[:, outc_], y_score=y_pred[:, outc_])
                    prc[outc_] = average_precision_score(y_true=self.y_val[:, outc_], y_score=y_pred[:, outc_])

                    # Plot Curves
                    PrecisionRecallDisplay.from_predictions(y_true=self.y_val[:, outc_], y_pred=y_pred[:, outc_],
                                                            ax=ax1[outc_])
                    RocCurveDisplay.from_predictions(y_true=self.y_val[:, outc_], y_pred=y_pred[:, outc_], ax=ax2[outc_])

                    # Fix figures
                    ax1[outc_].set_xlabel("Sensitivity")
                    ax1[outc_].set_ylabel("Specificity")
                    ax2[outc_].set_xlabel("Recall")
                    ax2[outc_].set_ylabel("Precision")

                    ax1[outc_].set_title(f"ROC Curve for outcome {outc_}")
                    ax1[outc_].set_title(f"PRC Curve for outcome {outc_}")

                except ValueError:
                    pass

            # Save figures
            if self.save_fd is not None:
                fig1.savefig(self.save_fd + f"ROC_Curve_{epoch}")
                fig2.savefig(self.save_fd + f"PR_Curve{epoch}")

            print("End of Epoch {:d} - OVR ROC score: {}".format(epoch, roc))
            print("End of Epoch {:d} - OVR PRC score: {}".format(epoch, prc))


class PrintClusterInfo(cbck.Callback):
    """
    Callback method to display cluster distribution information assignment, and cluster separation.

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

            # Compute cluster_predictions
            clus_pred = self.model.compute_pis(self.X_val)
            clus_assign = self.model.clus_assign(self.X_val)
            clus_phens = self.model.Predictor(self.model.cluster_rep_set).numpy()

            # Define K
            K = self.model.K

            # Compute "hard" cluster assignment numbers
            hard_cluster_num = np.zeros(shape=K)
            for clus_id in range(self.model.K):
                hard_cluster_num[clus_id] = np.sum(clus_assign == clus_id)

            # Compute average cluster distribution
            avg_cluster_dist = np.mean(clus_pred, axis=0)

            # Print Information
            print(f"End of Epoch {epoch:d} - Cluster Info", f"Clus dist: {hard_cluster_num}",
                  f"Avg assignment: {avg_cluster_dist}", f"Phenotypes: {clus_phens}", sep="\n")


def cbck_list(summary_name: str, interval: int = 5, validation_data: tuple = (), save_fd: str = None):
    """
    Shorthand for callbacks above.

    Params:
    - summary_name: str containing shorthands for different callbacks.
    - save_fd: str containing save path.
    - interval: int interval to print information on.
    """
    extra_callback_list = []

    if "cm" in summary_name.lower() or "conf_matrix" in summary_name.lower():
        extra_callback_list.append(ConfusionMatrix(interval=interval, validation_data=validation_data))

    if "clus_info" in summary_name.lower():
        extra_callback_list.append(PrintClusterInfo(interval=interval, validation_data=validation_data))

    if "auc" in summary_name.lower() or "prc" in summary_name.lower() or "curve" in summary_name.lower():
        extra_callback_list.append(PrCurves(interval=interval, validation_data=validation_data, save_fd=save_fd))

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
    callbacks.extend(cbck_list(other_cbcks, interval, validation_data=validation_data, save_fd=save_fd))

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
