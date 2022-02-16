#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Loss, Metrics and Callback functions to use for model

@author: henrique.aguiar@ds.ccrg.kadooriecentre.org
"""

# ----------------------------------------------------------------------------------
"Imports"
import os
import numpy as np

import tensorflow as tf
from tensorflow.math import squared_difference, multiply, divide
import tensorflow.keras.callbacks as cbck

from sklearn.metrics import roc_auc_score as roc
from sklearn.metrics.cluster import contingency_matrix
from sklearn.metrics import adjusted_rand_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.metrics import normalized_mutual_info_score, silhouette_score

# ----------------------------------------------------------------------------------
"Utility Functions and Global Params"


def log(tensor):
    return tf.cast(tf.math.log(tensor + 1e-8), dtype="float32")


def purity_score(y_true, y_pred):
    # compute confusion matrix
    contingency_matrix_ = contingency_matrix(y_true, y_pred)

    return np.sum(np.amax(contingency_matrix_, axis=0)) / np.sum(contingency_matrix_)

# ------------------------------------------------------------------------------------
"""Loss Functions"""


def l_pred(y_true, y_pred, weights=None, name='pred_clus_L'):
    """ Predictive clustering loss."""

    # Check for whether weights are given or not
    if weights is None:
        weights = tf.cast(tf.constant(np.ones(shape=y_true.shape) / y_true.shape[-1]), dtype=np.float32)
    else:
        weights = tf.convert_to_tensor(value=weights, dtype="float32")

    # Compute batch and return
    batch_loss = - tf.reduce_mean(tf.reduce_sum(weights * y_true * log(y_pred), axis=-1), name=name)

    return batch_loss


def l_clus(clusters, name='emb_sep_L'):
    """Compute Embedding separation Loss on embedding vectors."""

    embedding_column = tf.expand_dims(clusters, axis=1)
    embedding_row = tf.expand_dims(clusters, axis=0)

    # Compute L1 distance
    pairwise_loss = tf.reduce_sum((embedding_column - embedding_row) ** 2, axis=-1)  # shape K, K
    loss = - tf.reduce_sum(pairwise_loss, axis=None, name=name)

    # normalise by K(K-1=/2
    norm_factor = tf.math.subtract(tf.math.square(clusters.get_shape()[0]), clusters.get_shape()[0])
    norm_loss = tf.math.divide(loss, tf.cast(norm_factor, dtype="float32"))

    return norm_loss


def l_dist(y_prob):
    """Cluster loss to encourage exploration of all available clusters."""

    # Compute average distribution over each cluster and compute negative entropy of resulting distribution
    avg_prob_per_clust = tf.reduce_mean(y_prob, axis=-1, name="average")
    entropy = tf.reduce_sum(multiply(avg_prob_per_clust, log(avg_prob_per_clust + 1E-8)))

    return entropy


# ----------------------------------------------------------------------------------
"Useful information to print during training."


class CESeparation(cbck.Callback):
    """Compute normalised Cross-Entropy Loss between cluster phenotypes. Smaller the better."""

    def __init__(self, validation_data=(), interval=5):
        super().__init__()
        self.interval = interval
        self.X_val, _ = validation_data

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            ce_sep, epsilon = 0, 1e-9
            K = self.model.embeddings.numpy().shape[0]

            # Compute embedding phenotypes
            clus_phenotypes = self.model.Predictor(self.model.embeddings).numpy() + epsilon

            # Iterate over all pairs of clusters and compute symmetric CE
            for i in range(K):
                for j in range(i + 1, K):
                    ce_sep += - np.sum(clus_phenotypes[i, :] * np.log(clus_phenotypes[j, :]))
                    ce_sep += - np.sum(clus_phenotypes[j, :] * np.log(clus_phenotypes[i, :]))

            # normalise and print output
            norm_loss = ce_sep / (K * (K + 1))
            print("End of Epoch {:d} - CE sep : {:.4f}".format(epoch, norm_loss))


class ConfusionMatrix(cbck.Callback):
    """Display Confusion Matrix of predicted outcomes vs target outcomes."""

    def __init__(self, validation_data=(), interval=5):
        super().__init__()
        self.interval = interval
        self.X_val, self.y_val = validation_data
        self.C = self.y_val.shape[-1]  # num_classes

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            cm_output = np.zeros(shape=(self.C, self.C))

            # Compute prediction and true values in categorical format.
            model_output = (self.model(self.X_val)).numpy()
            y_pred = np.argmax(model_output, axis=-1)
            y_true = np.argmax(self.y_val, axis=-1)

            # Iterate through classes
            for true_class in range(self.C):
                for pred_class in range(self.C):
                    num_samples = np.logical_and(y_pred == pred_class, y_true == true_class).sum()
                    cm_output[true_class, pred_class] = num_samples

            print("End of Epoch {:d} - Confusion matrix: \n {}".format(epoch, cm_output.astype(int)))


class AUROC(cbck.Callback):
    """Compute AUROC on Validation Data."""

    def __init__(self, validation_data=(), interval=5):
        super().__init__()
        self.interval = interval
        self.X_val, self.y_val = validation_data

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            # Compute predictions
            y_pred = self.model(self.X_val).numpy()

            # Compute ROC
            roc_auc_score = roc(y_true=self.y_val, y_score=y_pred,
                                average=None, multi_class='ovr')

            print("End of Epoch {:d} - ROC score: {}".format(epoch, roc_auc_score))


class PrintClustersUsed(cbck.Callback):
    """Print Number of Clusters and Cluster Distribution with samples assigned to them."""

    def __init__(self, validation_data=(), interval=5):
        super().__init__()
        self.interval = interval
        self.X_val, self.y_val = validation_data

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            # Compute cluster assignment
            clus_pred = self.model.Identifier(self.model.Encoder(self.X_val)).numpy()
            num_clusters = np.unique(np.argmax(clus_pred, axis=-1))
            avg_cluster_dist = np.mean(clus_pred, axis=-1)

            # Print Information
            print("End of Epoch {:d} - num_clusters {} - cluster dist {}".format(epoch, num_clusters, avg_cluster_dist))


class SupervisedTargetMetrics(cbck.Callback):
    """Print Scores for all given Supervised Target Metrics (NMI, ARS, Purity) on validation data during training."""

    def __init__(self, validation_data=(), interval=5):
        super().__init__()
        self.interval = interval
        self.X_val, self.y_val = validation_data

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            # Compute y_pred, y_true in categorical format.
            model_output = (self.model(self.X_val)).numpy()
            y_pred = np.argmax(model_output, axis=-1)
            y_true = np.argmax(self.y_val, axis=-1).reshape(-1)

            # Target metrics
            nmi = normalized_mutual_info_score(labels_true=y_true, labels_pred=y_pred)
            ars = adjusted_rand_score(labels_true=y_true, labels_pred=y_pred)
            purity = purity_score(y_true=y_true, y_pred=y_pred)

            print("End of Epoch {:d} - NMI {:.2f} , ARS {:.2f} , Purity {:.2f}".format(epoch, nmi, ars, purity))


class UnsupervisedTargetMetrics(cbck.Callback):
    """Print Scores for all Unsupervised metrics (DBS, CHS, SIL) on validation data (inc. latent) over training."""

    def __init__(self, validation_data=(), interval=5):
        super().__init__()
        self.interval = interval
        self.latent_reps = None
        self.X_val, self.y_val = validation_data

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            # Compute predictions and latent representations
            self.latent_reps = self.model.Encoder(self.X_val)
            model_output = (self.model(self.X_val)).numpy()
            y_pred = np.argmax(model_output, axis=-1)

            # Reshape input data and allow feature comparison
            X_val_2d = np.reshape(self.X_val, (self.X_val[0], -1))

            # Compute metrics
            dbs = davies_bouldin_score(X_val_2d, labels=y_pred)
            dbs_l = davies_bouldin_score(self.latent_reps, labels=y_pred)
            chs = calinski_harabasz_score(X_val_2d, labels=y_pred)
            chs_l = calinski_harabasz_score(self.latent_reps, labels=y_pred)
            sil = silhouette_score(X=X_val_2d, labels=y_pred, random_state=self.model.seed)
            sil_l = silhouette_score(X=self.latent_reps, labels=y_pred, random_state=self.model.seed)

            print(f"""End of Epoch {epoch:d} (score, score on latent): 
                        DBS {dbs:.2f}, {dbs_l:.2f}   
                        CHS {chs:.2f}, {chs_l:.2f}  
                        SIL {sil:.2f}, {sil_l:.2f}""")


def compute_metric(metric_name):
    """Given metric shorthand, return corresponding callback."""
    if "auc" == metric_name.lower() or "roc" == metric_name.lower():
        return AUROC

    elif "ce_sep" == metric_name.lower():
        return CESeparation

    elif "conf_matrix" == metric_name.lower():
        return ConfusionMatrix

    elif "clus_dist" == metric_name.lower():
        return PrintClustersUsed

    elif "sup_targets" == metric_name.lower():
        return SupervisedTargetMetrics

    elif "unsup_targets" == metric_name.lower():
        return UnsupervisedTargetMetrics


def get_callbacks(track_loss, early_stop=True, lr_scheduler=True, tensorboard=True, min_delta=0.0001, patience=100):
    """Generate list of callbacks, given input params.

    Params:
        - track_loss: str, name of evaluate.py loss to keep track of.
        - early_stop: whether to stop training early in case of no progress. (default = True)
        - lr_scheduler: dynamically update learning rate. (default = True)
        - tensorboard: write tensorboard friendly logs which can then be visualised. (default = True)
        - min_delta: if early stopping, the interval on which to check improvement or not.
        - patience: how many epochs to wait until checking for improvements.
        """
    cbck_list = []

    # Handle saving paths and folders
    logs_dir = "experiments/evaluate.py/"
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)

    # Save Folder is first run that has not been previously computed
    run_num = 1
    while os.path.exists(logs_dir + f"run{run_num}/"):
        run_num += 1
    save_fd = logs_dir + f"run{run_num}/"
    assert not os.path.exists(save_fd)
    os.makedirs(save_fd)
    os.makedirs(save_fd + "logs/")


    # Model Weight saving callback
    checkpoint = cbck.ModelCheckpoint(filepath=save_fd + "models/checkpoints/epoch-{epoch}", save_best_only=True,
                                      monitor=track_loss, save_freq="epoch")
    csv_logger = cbck.CSVLogger(filename=save_fd + "loss_tracker.csv", separator=",", append=True)
    cbck_list.append(checkpoint)
    cbck_list.append(csv_logger)

    if early_stop:
        cbck_list.append(cbck.EarlyStopping(monitor='val_' + track_loss, mode="min", restore_best_weights=True,
                                            min_delta=min_delta, patience=patience))

    if lr_scheduler:
        cbck_list.append(cbck.ReduceLROnPlateau(monitor='val_' + track_loss, mode='min', cooldown=15,
                                                min_lr=0.00001, factor=0.25))

    if tensorboard:
        cbck_list.append(cbck.TensorBoard(log_dir=save_fd + "logs/", histogram_freq=1))

    return cbck_list, run_num
