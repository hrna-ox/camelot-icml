#!/usr/bin/env python3

"""
Model definition for simple LSTM - MLP Predictor model.

Date Last updated: 24 Jan 2022
Author: Henrique Aguiar
Please contact via henrique.aguiar@eng.ox.ac.uk
"""
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import optimizers
from sklearn.metrics import confusion_matrix

import os, json
from typing import Union

import src.models.deep_learning.enc_pred.model_utils as model_utils
from src.models.deep_learning.model_blocks import MLP, LSTMEncoder


class EncPred(tf.keras.Model):
    """
    Model Class for Encoder-Predictor model architecture.

    Params:

        (General)
    - latent_dim: dimensionality of latent space. (default = 32)
    - output_dim: dimensionality of output space. (default = 4)
    - seed: Seed to run analysis on. (default = 4347)
    - name: Name to give the model. (default = "CAMELOT")

        (Regularisation Params)
    - regulariser_params: tuple of l1_l2 float weights. (default = (0.01, 0.01))
    - dropout: float corresponding to dropout value. (default = 0.6)

        (Encoder Params)
    - encoder_params: Dictionary indicating parameters for Encoder architecture, as follows:
            - activation: activation function of custom feature projection component (default = "linear")
            - hidden_layers: Number of "hidden"/intermediate LSTM layers.  (default = 1)
            - hidden_nodes: Dimensionality of the intermediate state computation. (default = 20)
            - state_fn: The activation function to use on cell state/output. (default = 'tanh')
            - recurrent_activation: The activation function to use on F/I/O gates. (default = 'sigmoid')
            - recurrent_dropout: dropout rate to be used on forget/input/output gates. (default = 0.0)
    Default value is {}, which resets to default parameters.

        (Predictor Params)
    - predictor_params: Dictionary indicating parameters for Predictor block, as follows:
        - hidden_layers: int, Number of "hidden" feedforward layers. (default = 2)
        - hidden_nodes: int, For hidden feedforward layers, the dimensionality of the output space. (default = 30)
        - activation_fn: str/fn, The activation function to use. (default = 'sigmoid')
    Default value is {"name": "Predictor"}, which resets to default parameters.

        (Others)
    - weighted_loss: whether to use weights on predictive clustering loss (default = "True")
    """

    def __init__(self, latent_dim=32, seed=4347, output_dim=4, name="ENCPRED",
                 regulariser_params=(0.01, 0.01), dropout=0.6,
                 encoder_params=None, predictor_params=None, weighted_loss=True):

        super().__init__(name=name)

        # General params
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.seed = seed

        # Common to all Networks
        self.regulariser = regulariser_params
        self.dropout = dropout

        # Build Networks
        self.encoder_params = encoder_params if encoder_params is not None else {}
        self.predictor_params = predictor_params if predictor_params is not None else {}

        self.Encoder = LSTMEncoder(latent_dim=self.latent_dim, dropout=self.dropout,
                                   regulariser_params=self.regulariser, name="Encoder",
                                   **self.encoder_params)
        self.Predictor = MLP(output_dim=self.output_dim, dropout=self.dropout, output_fn="softmax",
                             regulariser_params=self.regulariser, seed=self.seed, name="Predictor",
                             **self.predictor_params)

        # Others
        self.weighted_loss = weighted_loss
        self.loss_weights = None

    # Build and Call Methods
    def build(self, input_shape):
        """Build method to serialise layers."""
        self.Encoder.build(input_shape)

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        """
        Call method for model.

        Params:
        - inputs: array-like of shape (bs, T, D_f)

        Returns: tuple of arrays:
            - y_pred: array-like of shape (bs, outcome_dim) with probability assignments.
        """
        y_pred = self.forward_pass(inputs)

        return y_pred

    # TRAINING RELATED METHODS
    def forward_pass(self, inputs):
        """
        Single forward pass given input data. Inputs are encoded. Encoded representations used to predict outcome.

        Params:
        - inputs: array-like of shape (bs, T, D_f) of input data.

        Returns: tuple of arrays:
            - y_pred: array-like of shape (bs, outcome_dim) with probability assignments.
        """
        # Encode data
        z = self.Encoder(inputs)

        # Make predictions
        y_pred = self.Predictor(z)

        return y_pred

    def train_step(self, inputs):
        """
        Method to train model.

        Params:
            - inputs: tuple of input (X, y) data:
                - X: array-like of shape (bs, T, D_f) of input time-series feature data.
                - y: array-like of shape (bs, num_outcs) of input outcome class data.

        Method updates model weights according to loss backpropagation.
        """

        # Unpack inputs
        x, y = inputs

        # Define variables for each network
        train_vars = [var for var in self.trainable_variables if 'predictor' in var.name.lower() or
                      "encoder" in var.name.lower()]

        # ------------------------------------------ OPTIMISE PREDICTOR ----------------------------------------

        # Initialise GradientTape to compute gradients
        with tf.GradientTape(watch_accessed_variables=False, persistent=True) as tape:
            tape.watch(train_vars)

            # Make forward pass
            y_pred = self.forward_pass(x)

            if self.weighted_loss is True:
                self.loss_weights = model_utils.class_weighting(y)

            # compute losses
            loss = model_utils.l_pred(y, y_pred, weights=self.loss_weights)

        # Compute gradients
        grad = tape.gradient(target=loss, sources=train_vars)

        # Apply gradients
        self.optimizer.apply_gradients(zip(grad, train_vars))

    def test_step(self, inputs):
        """
        Method to compute test step model.

        Params:
            - inputs: tuple of input (X, y) data:
                - X: array-like of shape (bs, T, D_f) of input time-series feature data.
                - y: array-like of shape (bs, num_outcs) of input outcome class data.
        """

        # Unpack input data
        x, y = inputs

        # Make forward pass
        y_pred = self.forward_pass(x)

        # Update loss weights depending on batch
        if self.weighted_loss is True:
            self.loss_weights = model_utils.class_weighting(y)

        # compute losses
        loss = model_utils.l_pred(y, y_pred, weights=self.loss_weights)

        return {"loss": loss}

    def compute_pis(self, x):
        """Obtain representations."""
        zs = self.Encoder(x)

        return zs.numpy()

    def get_config(self):
        """Update configuration for layer."""
        config = {}

        # Update configuration
        config.update({f"{self.name}-latent_dim": self.latent_dim,
                       f"{self.name}-output_dim": self.output_dim,
                       f"{self.name}-seed": self.seed,
                       f"{self.name}-regulariser": self.regulariser,
                       f"{self.name}-dropout": self.dropout,
                       f"{self.name}-weighted_loss": self.weighted_loss})

        # Update configuration for each model
        config.update(self.Encoder.get_config())
        config.update(self.Predictor.get_config())

        return config


ENCPRED_INPUT_PARAMS = ["latent_dim", "seed", "output_dim", "name", "regulariser_params", "dropout", "encoder_params",
                        "predictor_params", "weighted_loss"]


class Model(EncPred):
    """
    Model Class Wrapper for ENCPRED model with train and analyse methods.
    """

    def __init__(self, data_info: dict, model_config: dict, training_config: dict):
        """
        Initialise object with model configuration.

        Params:
        - data_info: dict, contains basic data information, objects and properties
        - kwargs: other model configuration parameters
        """
        # Get output dimension
        output_dim = data_info["y"][-1].shape[-1]

        # Useful information
        relevant_params = {key: value for key, value in model_config.items() if key in ENCPRED_INPUT_PARAMS}
        self.model_config = {"output_dim": output_dim, **relevant_params}
        self.callback_lst = None
        self.run_num = 1
        self.model_name = "ENCPRED"

        # Initialise training parameters
        self.training_params = training_config

        super().__init__(**self.model_config)

        self.build_model(data_info)

    def build_model(self, data_info):
        """
        Build all model parameters.

        Params:
        - data_info: dictionary with data information and parameters.

        Returns:
            if applicable, model built.
        """

        # Get data useful info
        data_name = data_info["data_load_config"]["data_name"]
        cbck_str, patience_epochs = self.training_params["cbck_str"], self.training_params["patience_epochs"]
        lr = self.training_params["lr"]

        # Unpack relevant data information
        X_train, X_val, X_test = data_info["X"]
        y_train, y_val, y_test = data_info["y"]

        # Initialise model
        self.build(X_train.shape)

        # Load optimizer
        optimizer = optimizers.Adam(learning_rate=lr)
        self.compile(optimizer=optimizer, run_eagerly=True)

        # Load Checkpoint
        callbacks, run_num = model_utils.get_callbacks((X_val, y_val), data_name=data_name, track_loss="loss",
                                                       other_cbcks=cbck_str, patience_epochs=patience_epochs,
                                                       early_stop=True, lr_scheduler=True, tensorboard=True)

        # Update run num
        self.run_num = run_num
        self.callback_lst = callbacks

        return None

    def train(self, data_info, lr: float = 0.001, epochs: int = 100, bs: int = 32,
              patience_epochs: int = 200, gpu: Union[str, None] = None, **kwargs):
        """
        Fit method for training ENCPRED model.

        Params:
        - data_info: dictionary with data information and parameters.
        - "lr": learning rate for training (default = 0.001)
        - "epochs": number of epochs for main.py training (default = 100)
        - "bs": batch size (default = 32)
        - "patience_epochs": int, maximum number of epochs to wait for improvement. (default=200)
        - gpu: str or None indicating how to use gpu training. If None, then no gpu is used. If "strategy", then
        parallelise. Otherwise, regular GPU training.
        """
        self.training_params.update({
            "lr": lr,
            "epochs": epochs,
            "bs": bs,
            "patience_epochs": patience_epochs,
            "gpu": gpu
        })

        # Unpack relevant data information
        X_train, X_val, X_test = data_info["X"]
        y_train, y_val, y_test = data_info["y"]

        # Train model on initialisation procedure
        train_data = X_train, y_train
        val_data = X_val, y_val

        # Main Training phase
        print("-" * 20, "\n", "STARTING MAIN TRAINING PHASE")
        train_data = tf.data.Dataset.from_tensor_slices(train_data).shuffle(1000).batch(bs)
        val_data = tf.data.Dataset.from_tensor_slices(val_data).shuffle(1000).batch(bs)

        # # Disable Autoshard
        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
        train_data, val_data = train_data.with_options(options), val_data.with_options(options)

        # Fit model
        history = self.fit(train_data, validation_data=val_data, epochs=epochs,
                           verbose=2)

        return history

    def analyse(self, data_info):
        """
        Evaluation method for computing output results.

        Params:
        - data_info: dictionary with data information.

        Returns:
            - y_pred: dataframe of shape (N, output_dim) with outcome probability prediction.
            - outc_pred: Series of shape (N, ) with predicted outcome based on most likely outcome prediction.
            - y_true: dataframe of shape (N, output_dim) ith one-hot encoded true outcome.
            - pis_pred: dataframe of shape (N, K) of cluster probability assignment.
            - clus_pred: Series of shape (N, ) with cluster assignment based on most likely cluster probability.
            - clus_phenotypes: DataFrame of shape (K, output_dim) with predicted cluster outcome probability.

        Saves a variety of model information, as well.
        """

        # Unpack test data
        _, _, X_test = data_info["X"]
        _, _, y_test = data_info["y"]
        data_properties = data_info["data_properties"]
        data_load_config = data_info["data_load_config"]

        # Source outcome names and patient id info
        id_info = data_info["ids"][-1]
        pat_ids = id_info[:, 0, 0]
        outc_names = data_properties["outc_names"]
        data_name = data_load_config["data_name"]

        # Define save_fd
        save_fd = f"results/{data_name}/{self.model_name}/run{self.run_num}/"
        track_fd = f"experiments/{data_name}/{self.model_name}/run{self.run_num}/"

        if not os.path.exists(save_fd):
            os.makedirs(save_fd)

        if not os.path.exists(track_fd):
            os.makedirs(track_fd)

        # Other useful definitions
        output_test = self.predict(X_test)

        # Firstly, compute predicted y estimates
        y_pred = pd.DataFrame(output_test, index=pat_ids, columns=outc_names)
        outc_pred = pd.Series(np.argmax(output_test, axis=-1), index=pat_ids)
        y_true = pd.DataFrame(y_test, index=pat_ids, columns=outc_names)

        # Secondly, compute predicted cluster assignments
        pis_pred = pd.DataFrame(output_test, index=pat_ids, columns=outc_names)
        clus_pred = pd.Series(np.argmax(output_test, axis=-1), index=pat_ids)

        # Thirdly, compute cluster phenotype information
        cm = confusion_matrix(y_true=y_test, y_pred=output_test)
        clus_phenotypes = pd.DataFrame(cm / np.sum(cm, axis=0, keepdims=True), index=outc_names, columns=outc_names)

        # Fourth, get configuration
        model_config = self.get_config()

        # ----------------------------- Save Output Data --------------------------------
        # Useful objects
        y_pred.to_csv(save_fd + "y_pred.csv", index=True, header=True)
        outc_pred.to_csv(save_fd + "outc_pred.csv", index=True, header=True)
        y_true.to_csv(save_fd + "y_true.csv", index=True, header=True)
        pis_pred.to_csv(save_fd + "pis_pred.csv", index=True, header=True)
        clus_pred.to_csv(save_fd + "clus_pred.csv", index=True, header=True)
        clus_phenotypes.to_csv(save_fd + "clus_phenotypes.csv", index=True, header=True)

        # save model parameters
        save_params = {**data_info["data_load_config"], **self.model_config, **self.training_params}
        with open(save_fd + "config.json", "w+") as f:
            json.dump(save_params, f, indent=4)

        with open(track_fd + "config.json", "w+") as f:
            json.dump(save_params, f, indent=4)

        with open(save_fd + "model_config.json", "w+") as f:
            json.dump(model_config, f, indent=4)

        with open(track_fd + "model_config.json", "w+") as f:
            json.dump(model_config, f, indent=4)

        # Return objects
        outputs_dic = {
            "y_pred": y_pred, "class_pred": outc_pred, "y_true": y_true, "pis_pred": pis_pred, "clus_pred": clus_pred,
            "clus_phenotypes": clus_phenotypes, "logs": track_fd + "logs",
            "save_fd": save_fd, "model_config": self.model_config
        }

        # Print Data
        print(f"\n\n Experiments saved under {track_fd} and {save_fd}")

        return outputs_dic
