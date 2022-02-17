#!/usr/bin/env python3

"""
Model definition for Camelot data

Date Last updated: 12 August 2021
Author: Henrique Aguiar
Please contact via henrique.aguiar@eng.ox.ac.uk
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras import metrics
from tensorflow.keras import optimizers


# Auxiliary function

# Loading libraries
from src.models.traditional_clustering.ENC_PRED.model_blocks import MLP, LSTMEncoder
import src.models.traditional_clustering.ENC_PRED.model_utils as utils

# Loss tracking
L1 = metrics.Mean(name="L1")

# Validation loss tracking
val_L1 = metrics.Mean(name="val_L1")

class LSTMEP(tf.keras.Model):
    """
        Model Class for CAMELOT architecture.

        Params:
            (General)
        - output_dim          : dimensionality of target predicted output. (default = 4)
        - latent_dim          : dimensionality of latent space. (default = 32)
        - seed                : Seed to run analysis on. (default = 4347)
        - name                : Name to give the model. (default = "Camelot")


            (Encoder Params)
        - encoder_params: Dictionary indicating parameters for Encoder architecture, as follows:

                - hidden_layers       : Number of "hidden"/intermediate LSTM layers.  (default = 1)
                - hidden_nodes        : Dimensionality of the intermediate state computation. (default = 20)
                - state_fn            : The activation function to use on cell state/output. (default = 'tanh')
                - recurrent_activation: The activation function to use on F/I/O gates. (default = 'sigmoid')
                - recurrent_dropout   : dropout rate to be used on forget/input/output gates. (default = 0.0)
                - name                : Name on which to save component. (default = 'ENCODER')
        Default value is {}, which resets to default parameters.

            (Predictor Params)
        - predictor_params: Dictionary indicating parameters for Predictor block, as follows:
            - hidden_layers : int, Number of "hidden" feedforward layers. (default = 2)
            - hidden_nodes : int, For hidden feedforward layers, the dimensionality of the output space. (default = 30)
            - activation_fn : str/fn, The activation function to use. (default = 'sigmoid')
            - name : str, name on which to save layer. (default = 'PREDICTOR')
        Default value is {}, which resets to default parameters.

    """

    def __init__(self, output_dim=4, latent_dim=32, seed=4347, name="Camelot",
                 regulariser_params=(0.01, 0.01), dropout=0.6,
                 encoder_params=None, predictor_params=None, cluster_rep_lr=0.01,
                 optimizer_init="adam"):

        super().__init__(name=name)

        # General params
        self.output_dim = output_dim
        self.latent_dim = latent_dim
        self.seed = seed


        # Common to all Networks
        self.regulariser = regulariser_params
        self.dropout = dropout

        # Build Networks
        self.encoder_params = encoder_params if encoder_params is not None else {}
        self.predictor_params = predictor_params if predictor_params is not None else {}

        self.Encoder = LSTMEncoder(latent_dim=self.latent_dim, dropout=self.dropout,
                                           regulariser_params=self.regulariser, name="ENCODER",
                                           **self.encoder_params)

        self.Predictor = MLP(output_dim=self.output_dim, dropout=self.dropout, output_fn="softmax",
                             regulariser_params=self.regulariser, seed=self.seed, name="PREDICTOR",
                             **self.predictor_params)

        inter_ = np.array([2499, 768, 72, 59])
        self.loss_weights = (1 / inter_) / np.sum(1/inter_)
        self.optimizer = optimizers.Adam(0.001)

    # Build and Call Methods
    def build(self, input_shape):
        """Build method to serialise layers."""
        self.Encoder.build(input_shape)

        super().build(input_shape)

    def call(self, inputs):
        """Call method for model."""
        y_pred = self.forward_pass(inputs)

        return y_pred

    # TRAINING RELATED METHODS
    def forward_pass(self, inputs):
        """Single forward pass given some data"""
        z = self.Encoder(inputs)

        # Make predictions
        y_pred = self.Predictor(z)

        return y_pred
        
    def test_step(self, data):
        x, y = data
        
        y_pred = self.Predictor(self.Encoder(x))
        loss = utils.l_pred(y, y_pred, weights=self.loss_weights)
            
        val_L1.update_state(loss)
        
        return {"L1": val_L1.result()}  
            
            
            
            
    # LOGISTICAL METHODS
    @classmethod
    def from_config(cls, config):
        return cls(**config)

    @property
    def metrics(self):
        """Return initialised metrics"""
        return [L1, val_L1]
    
    
    def get_config(self):
        """Update configuration for layer."""
        config = super().get_config().copy()
        config.update({"latent_dim": self.latent_dim, "hidden_layers": self.hidden_layers,
                       "hidde_nodes": self.hidden_nodes, "state_fn": self.state_fn,
                       "recurrent_fn": self.recurrent_fn, "return_sequences": self.return_sequences,
                       "dropout": self.dropout, "recurrent_dropout": self.recurrent_dropout,
                       "regulariser_params": self.regulariser_params})

        return config


