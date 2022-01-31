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

from sklearn.metrics import roc_auc_score as roc
from sklearn.metrics.cluster import contingency_matrix
from sklearn.metrics import adjusted_rand_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.metrics import normalized_mutual_info_score, silhouette_score

# ----------------------------------------------------------------------------------
"Utility Functions and Global Params"

LOGS_DIR = "experiments/camelot/"


def log(tensor):
    return tf.cast(tf.math.log(tensor + 1e-8), dtype="float32")


def np_log(array):
    return np.log(array + 1e-8)


def purity_score(y_true, y_pred):
    # compute confusion matrix
    contingency_matrix_ = contingency_matrix(y_true, y_pred)

    return np.sum(np.amax(contingency_matrix_, axis=0)) / np.sum(contingency_matrix_)


def supervised_scores(y_true, y_pred):
    """
    Print Supervised Metric scores given set of true outcomes and predicted class.

    Params:
    - y_true: array-like of shape (N, num_outcs) with one-hot encoded true classes.
    - y_pred: array-like of shape (N, num_outcs) with predicted outcome distribution.

    Returns:
        Tuple of scores:
        - ROC AUC SCORE (One vs Rest)
        - Normalised Mutual Information
        - Adjusted Rand Score
        - Purity
    """

    # Convert to categorical
    class_true = np.argmax(y_true, axis=-1)
    class_pred = np.argmax(y_pred, axis=-1)

    # Compute scores
    auc = roc(y_true, y_pred, multi_class="ovr")
    nmi = normalized_mutual_info_score(labels_true=class_true, labels_pred=class_pred)
    ars = adjusted_rand_score(labels_true=class_true, labels_pred=class_pred)
    purity = purity_score(y_true=class_true, y_pred=class_pred)

    return auc, nmi, ars, purity


def unsupervised_scores(x, y_pred=None, seed: int = 4347):
    """
    Print Supervised Metric scores given set of true outcomes and predicted class.

    Params:
    - X: array-like input data in two dimensions.
    - y_pred: array-like of shape (N, num_outcs) with predicted outcome distribution.
    - seed: int, seed to use for computing silhouette score. (default=4347)

    Returns:
        Tuple of scores:
        - Davies Bouldin Scores
        - Calinski Harabasz
        - Silhouette Score
    """

    # Compute predicted class
    clus_pred = np.argmax(y_pred, axis=-1)

    # Compute metrics
    dbs = davies_bouldin_score(x, labels=clus_pred)
    chs = calinski_harabasz_score(x, labels=clus_pred)
    sil = silhouette_score(X=x, labels=clus_pred, random_state=seed)

    return dbs, chs, sil


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
    batch_neg_ce = tf.reduce_sum(weights * y_true * log(y_pred), axis=-1)

    # Average over batch
    loss_value = tf.reduce_mean(batch_neg_ce, name=name)

    return loss_value


def l_clus(cluster_reps, name='embedding_sep_loss'):
    """Compute Embedding separation Loss on embedding vectors."""
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
    loss = - tf.reduce_sum(pairwise_loss, axis=None, name=name)

    # normalise by factor
    K = cluster_reps.get_shape()[0]
    norm_loss = loss / (K * (K - 1))

    return norm_loss


def l_dist(clusters_prob):
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
    neg_entropy = tf.reduce_sum(clus_avg_prob * log(clus_avg_prob))

    return neg_entropy


# ----------------------------------------------------------------------------------
"Callback methods to update training procedure."


class CEClusSeparation(cbck.Callback):
    """
    Callback method to print Normalised Cross-Entropy separation between cluster phenotypes.
    Higher values indicate higher separation (which is good!)

    Params:
    - validation_data: tuple of X_val, y_val data
    - weighted: bool, indicating whether outcomes should be weighted. (default = False)
    - interval: interval between epochs on which to print values. (default = 5)
    """

    def __init__(self, validation_data: tuple, weighted: bool = False, interval: int = 5):
        super().__init__()
        self.interval = interval
        self.X_val, self.y_val = validation_data

        # Compute outcome weights if weighted=True
        if weighted is True:
            class_numbers = tf.reduce_sum(self.y_val, axis=0).numpy()
            self.weights = class_numbers / np.sum(class_numbers)

        else:
            self.weights = np.ones(shape=(self.y_val.get_shape()[0]))

    def on_epoch_end(self, epoch, **kwargs):

        # Print information if matches interval epoch length
        if epoch % self.interval == 0:

            # Initialise callback value, and determine K
            cbck_value, K = 0, self.model.cluster_reps.numpy().shape[0]
            clus_phenotypes = self.model.Predictor(self.model.cluster_reps).numpy()

            # Iterate over all pairs of clusters and compute symmetric CE
            for i in range(K):
                for j in range(i + 1, K):
                    cbck_value += - np.sum(clus_phenotypes[i, :] * np_log(clus_phenotypes[j, :]))
                    cbck_value += - np.sum(clus_phenotypes[j, :] * np_log(clus_phenotypes[i, :]))

            # normalise and print output
            cbck_value = cbck_value / (K * (K + 1))

            print("End of Epoch {:d} - CE sep : {:.4f}".format(epoch, cbck_value))


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
        self.num_outcs = self.y_val.shape[-1]

    def on_epoch_end(self, epoch, **kwargs):

        # Print information if matches interval epoch length
        if epoch % self.interval == 0:

            # Initialise output Confusion matrix
            cm_output = np.zeros(shape=(self.num_outcs, self.num_outcs))

            # Compute prediction and true values in categorical format.
            y_pred = (self.model(self.X_val)).numpy()
            class_pred = np.argmax(y_pred, axis=-1)
            class_true = np.argmax(self.y_val, axis=-1)

            # Iterate through classes
            for true_class in range(self.num_outcs):
                for pred_class in range(self.num_outcs):
                    num_samples = np.logical_and(class_pred == pred_class, class_true == true_class).sum()
                    cm_output[true_class, pred_class] = num_samples

            # Print as pd.dataframe
            index = [f"TC{class_}" for class_ in range(1, self.num_outcs + 1)]
            columns = index

            cm = pd.DataFrame(cm_output, index=index, columns=columns)

            print("End of Epoch {:d} - Confusion matrix: \n {}".format(epoch, cm.astype(int)))


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

    def on_epoch_end(self, epoch, **kwargs):
        if epoch % self.interval == 0:
            # Compute predictions
            y_pred = self.model(self.X_val).numpy()

            # Compute ROC
            roc_auc_score = roc(y_true=self.y_val, y_score=y_pred,
                                average=None, multi_class='ovr')

            print("End of Epoch {:d} - OVR ROC score: {}".format(epoch, roc_auc_score))


class PrintClusterInfo(cbck.Callback):
    """
    Callback method to display cluster distribution information assignment.

    Params:
    - validation_data: tuple of X_val, y_val data
    - interval: interval between epochs on which to print values. (default = 5)
    """

    def __init__(self, validation_data: tuple, interval: int = 5):
        super().__init__()
        self.interval = interval
        self.X_val, self.y_val = validation_data

    def on_epoch_end(self, epoch, **kwargs):
        if epoch % self.interval == 0:

            # Compute cluster_predictions
            clus_pred = self.model.Identifier(self.model.Encoder(self.X_val)).numpy()
            clus_assign = np.argmax(clus_pred, axis=-1)
            K = clus_pred.shape[-1]

            # Compute "hard" cluster assignment numbers
            hard_cluster_num = np.zeros(shape=K)
            for clus_id in range(self.K):
                hard_cluster_num[clus_id] = np.sum(clus_assign == clus_id)
            hard_cluster_num = pd.Series(hard_cluster_num, index=[f"Clus {k}" for k in range(1, K + 1)])

            # Compute average cluster distribution
            avg_cluster_dist = np.mean(clus_pred, axis=0)

            # Print Information
            print("End of Epoch {:d} - hard_cluster_info {} - avg cluster prob dist {}".format(epoch, hard_cluster_num,
                                                                                               avg_cluster_dist))


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

    def on_epoch_end(self, epoch, **kwargs):
        if epoch % self.interval == 0:
            # Compute y_pred, y_true in categorical format.
            model_output = (self.model(self.X_val)).numpy()
            class_pred = np.argmax(model_output, axis=-1)
            class_true = np.argmax(self.y_val, axis=-1).reshape(-1)

            # Target metrics
            nmi = normalized_mutual_info_score(labels_true=class_true, labels_pred=class_pred)
            ars = adjusted_rand_score(labels_true=class_true, labels_pred=class_pred)
            purity = purity_score(y_true=class_true, y_pred=class_pred)

            print("End of Epoch {:d} - NMI {:.2f} , ARS {:.2f} , Purity {:.2f}".format(epoch, nmi, ars, purity))


class UnsupervisedTargetMetrics(cbck.Callback):
    """
    Callback method to display unsupervised target metrics: Davies-Bouldin Score, Calinski-Harabasz Score,
    Silhouette Score

    Params:
    - validation_data: tuple of X_val, y_val data
    - interval: interval between epochs on which to print values. (default = 5)
    """

    def __init__(self, validation_data: tuple, interval: int = 5):
        super().__init__()
        self.interval = interval
        self.X_val, self.y_val = validation_data

    def on_epoch_end(self, epoch, **kwargs):
        if epoch % self.interval == 0:
            # Compute predictions and latent representations
            latent_reps = self.model.Encoder(self.X_val)
            model_output = (self.model(self.X_val)).numpy()
            clus_pred = np.argmax(model_output, axis=-1)

            # Reshape input data and allow feature comparison
            X_val_2d = np.reshape(self.X_val, (self.X_val[0], -1))

            # Compute metrics
            dbs = davies_bouldin_score(X_val_2d, labels=clus_pred)
            dbs_l = davies_bouldin_score(latent_reps, labels=clus_pred)
            chs = calinski_harabasz_score(X_val_2d, labels=clus_pred)
            chs_l = calinski_harabasz_score(latent_reps, labels=clus_pred)
            sil = silhouette_score(X=X_val_2d, labels=clus_pred, random_state=self.model.seed)
            sil_l = silhouette_score(X=latent_reps, labels=clus_pred, random_state=self.model.seed)

            print(f"""End of Epoch {epoch:d} (score, latent score): 
                        DBS {dbs:.2f}, {dbs_l:.2f}   
                        CHS {chs:.2f}, {chs_l:.2f}  
                        SIL {sil:.2f}, {sil_l:.2f}""")


def cbck_list(summary_name: str):
    """
    Shorthand for callbacks above.

    Params:
    - summary_name: str containing shorthands for different callbacks.
    """
    extra_callback_list = set([])

    if "auc" in summary_name.lower() or "roc" in summary_name.lower():
        extra_callback_list.update([AUROC])

    if "clus_sep" in summary_name.lower() or "clus_phen" in summary_name.lower():
        extra_callback_list.update([CEClusSeparation])

    if "cm" in summary_name.lower() or "conf_matrix" in summary_name.lower():
        extra_callback_list.update([ConfusionMatrix])

    if "clus_info" in summary_name.lower():
        extra_callback_list.update([PrintClusterInfo])

    if "sup_scores" in summary_name.lower():
        extra_callback_list.update([SupervisedTargetMetrics])

    if "unsup_scores" in summary_name.lower():
        extra_callback_list.update([UnsupervisedTargetMetrics])

    return list(extra_callback_list)


def get_callbacks(track_loss: str, other_cbcks: str = "", early_stop: bool = True, lr_scheduler: bool = True,
                  tensorboard: bool = True,
                  min_delta: float = 0.0001, patience: int = 100):
    """
    Generate complete list of callbacks given input configuration.

    Params:
        - track_loss: str, name of main loss to keep track of.
        - other_cbcks: str, list of other callbacks to consider (default = "", which selects None).
        - early_stop: whether to stop training early in case of no progress. (default = True)
        - lr_scheduler: dynamically update learning rate. (default = True)
        - tensorboard: write tensorboard friendly logs which can then be visualised. (default = True)
        - min_delta: if early stopping, the interval on which to check improvement or not.
        - patience: how many epochs to wait until checking for improvements.
        """
    callbacks = set([])

    # Handle saving paths and folders
    logs_dir = LOGS_DIR
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)

    # Save Folder is first run that has not been previously computed
    run_num = 1
    while os.path.exists(logs_dir + f"run{run_num}/"):
        run_num += 1

    # Save as new run
    save_fd = logs_dir + f"run{run_num}/"
    assert not os.path.exists(save_fd)

    os.makedirs(save_fd)
    os.makedirs(save_fd + "logs/")

    # ------------------ Start Loading callbacks ---------------------------

    # Load custom callbacks first
    callbacks.update(cbck_list(other_cbcks))

    # Model Weight saving callback
    checkpoint = cbck.ModelCheckpoint(filepath=save_fd + "models/checkpoints/epoch-{epoch}", save_best_only=True,
                                      monitor=track_loss, save_freq="epoch")
    callbacks.update([checkpoint])

    # Logging Loss values
    csv_logger = cbck.CSVLogger(filename=save_fd + "training/loss_tracker.csv", separator=",", append=True)
    callbacks.update([csv_logger])

    # Check if Early stoppage is added
    if early_stop:
        callbacks.update([cbck.EarlyStopping(monitor='val_' + track_loss, mode="min", restore_best_weights=True,
                                             min_delta=min_delta, patience=patience)])

    # Check if LR Scheduling is in place
    if lr_scheduler:
        callbacks.update([cbck.ReduceLROnPlateau(monitor='val_' + track_loss, mode='min', cooldown=15,
                                                 min_lr=0.00001, factor=0.25)])

    # Check if Tensorboard is active
    if tensorboard:
        callbacks.update([cbck.TensorBoard(log_dir=save_fd + "logs/", histogram_freq=1)])

    return callbacks, run_num
