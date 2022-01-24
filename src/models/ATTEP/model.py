#!/usr/bin/env python3

"""
Model definition for Camelot data

Date Last updated: 12 August 2021
Author: Henrique Aguiar
Please contact via henrique.aguiar@eng.ox.ac.uk
"""
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.cluster import KMeans
from tensorflow.keras import metrics
from tensorflow.keras import optimizers

import sys

# Auxiliary function
import src.models.benchmarks.ATTEP.model_utils as model_utils

# Loading libraries
from src.models.benchmarks.ATTEP.model_blocks import MLP, AttentionRNNEncoder

# Loss tracking
L1 = metrics.Mean(name="L1")
L2 = metrics.Mean(name="L2")
L3 = metrics.Mean(name="L3")

# Validation loss tracking
val_L1 = metrics.Mean(name="val_L1")
val_L2 = metrics.Mean(name='val_L2')
val_L3 = metrics.Mean(name='val_L3')


class ATTEP(tf.keras.Model):
    """
        Model Class for CAMELOT architecture.

        Params:
            (General)
        - num_clusters        : number of clusters. (default = 10)
        - output_dim          : dimensionality of target predicted output. (default = 4)
        - latent_dim          : dimensionality of latent space. (default = 32)
        - seed                : Seed to run analysis on. (default = 4347)
        - name                : Name to give the model. (default = "Camelot")

            (Loss function)
        - alpha               : L2 weighting in cluster entropy. (default = 0.01)
        - beta                : L3 weighting in clustering representation separation. (default = 0.01)


            (Encoder Params)
        - encoder_params: Dictionary indicating parameters for Encoder architecture, as follows:
                - activation          : activation function of custom feature projection component (default = "linear")
                - hidden_layers       : Number of "hidden"/intermediate LSTM layers.  (default = 1)
                - hidden_nodes        : Dimensionality of the intermediate state computation. (default = 20)
                - state_fn            : The activation function to use on cell state/output. (default = 'tanh')
                - recurrent_activation: The activation function to use on F/I/O gates. (default = 'sigmoid')
                - recurrent_dropout   : dropout rate to be used on forget/input/output gates. (default = 0.0)
                - name                : Name on which to save component. (default = 'ENCODER')
        Default value is {}, which resets to default parameters.

            (Identifier Params)
        - identifier_params: Dictionary indicating parameters for Identifier block, as follows:
            - hidden_layers : int, Number of "hidden" feedforward layers. (default = 2)
            - hidden_nodes : int, For hidden feedforward layers, the dimensionality of the output space. (default = 30)
            - activation_fn : str/fn, The activation function to use. (default = 'sigmoid')
            - name : str, name on which to save layer. (default = 'IDENTIFIER')
        Default value is {}, which resets to default parameters.

            (Predictor Params)
        - predictor_params: Dictionary indicating parameters for Predictor block, as follows:
            - hidden_layers : int, Number of "hidden" feedforward layers. (default = 2)
            - hidden_nodes : int, For hidden feedforward layers, the dimensionality of the output space. (default = 30)
            - activation_fn : str/fn, The activation function to use. (default = 'sigmoid')
            - name : str, name on which to save layer. (default = 'PREDICTOR')
        Default value is {}, which resets to default parameters.

            (Cluster Representation Params)
        - cluster_rep_lr: Learning rate for update of cluster_representations. (default = 0.01)

            (Others)
        - optimizer_init: optimizer to use for initialisation training. (default = "adam")
    """

    def __init__(self, num_clusters=10, output_dim=4, latent_dim=32, seed=4347, name="Camelot",
                 alpha=0.01, beta=0.01, regulariser_params=(0.01, 0.01), dropout=0.6,
                 encoder_params=None, identifier_params=None, predictor_params=None, cluster_rep_lr=0.01,
                 optimizer_init="adam"):

        super().__init__(name=name)

        # General params
        self.K = num_clusters
        self.output_dim = output_dim
        self.latent_dim = latent_dim
        self.seed = seed

        # Loss function params
        self.alpha = alpha
        self.beta = beta

        # Common to all Networks
        self.regulariser = regulariser_params
        self.dropout = dropout

        # Build Networks
        self.encoder_params = encoder_params if encoder_params is not None else {}
        self.identifier_params = identifier_params if identifier_params is not None else {}
        self.predictor_params = predictor_params if predictor_params is not None else {}

        self.Encoder = AttentionRNNEncoder(units=self.latent_dim, dropout=self.dropout,
                                           regulariser_params=self.regulariser, name="ENCODER",
                                           **self.encoder_params)
        self.Identifier = MLP(output_dim=self.K, dropout=self.dropout, output_fn="softmax",
                              regulariser_params=self.regulariser, seed=self.seed, name="IDENTIFIER",
                              **self.identifier_params)
        self.Predictor = MLP(output_dim=self.output_dim, dropout=self.dropout, output_fn="softmax",
                             regulariser_params=self.regulariser, seed=self.seed, name="PREDICTOR",
                             **self.predictor_params)

        # Cluster Representation params
        self.cluster_rep = tf.Variable(initial_value=tf.zeros(shape=[self.K, self.latent_dim], dtype='float32'),
                                       trainable=True, name='cluster_rep')
        self.cluster_rep_lr = cluster_rep_lr
        self.cluster_opt = optimizers.Adam(learning_rate=self.cluster_rep_lr)

        # Initialisation loss trackers
        self._enc_pred_loss_tracker = None
        self._iden_loss_tracker = None

        # Others
        self._optimizer_init = tf.keras.optimizers.get(optimizer_init)
        self.loss_weights = None

    # Build and Call Methods
    def build(self, input_shape):
        """Build method to serialise layers."""
        self.Encoder.build(input_shape)
        self.Encoder.feature_time_att_layer.build(input_shape)

        super().build(input_shape)

    def call(self, inputs):
        """Call method for model."""
        y_pred, pi = self.forward_pass(inputs)

        return y_pred

    # TRAINING RELATED METHODS
    def forward_pass(self, inputs):
        """Single forward pass given some data"""
        z = self.Encoder(inputs)
        pi = self.Identifier(z)

        # Sample from cluster assignments and assign corresponding cluster representations
        cluster_samp = self._sample_from_probs(pi)
        sample_representations = self._select_representations_from_sample(cluster_samp)

        # Make predictions
        y_pred = self.Predictor(sample_representations)

        return y_pred, pi

    def _sample_from_probs(self, probabilities):
        """Sample categorically given vector of probabilities for each patient."""
        logits = tf.math.log(tf.reshape(probabilities, shape=[-1, self.K]))
        sample = tf.random.categorical(logits, num_samples=1, seed=self.seed)

        return tf.squeeze(sample)  # shape (bs, )

    def _select_representations_from_sample(self, cluster_samples):
        """Given sampled cluster assignment, assign it to corresponding cluster representation"""
        idx_mask = tf.one_hot(cluster_samples, depth=self.K)
        representations = tf.linalg.matmul(idx_mask, self.cluster_rep)

        return representations

    def train_step(self, inputs):
        """Main method specifying how model is trained."""
        x, y = inputs

        # Define variables for each network
        pred_vars = [var for var in self.trainable_variables if 'PREDICTOR' in var.name]
        enc_id_vars = [var for var in self.trainable_variables if 'ENCODER' in var.name or 'IDENTIFIER' in var.name]
        rep_vars = self.cluster_rep

        # Initialise GradientTape to compute gradients
        with tf.GradientTape(watch_accessed_variables=True, persistent=True) as tape:
            y_pred, pi = self.forward_pass(x)

            # compute losses
            l_pred = model_utils.l_pred(y, y_pred, weights=self.loss_weights)
            l_enc_id = model_utils.l_pred(y, y_pred, weights=self.loss_weights) + self.alpha * model_utils.l_dist(pi)
            l_clus = model_utils.l_pred(y, y_pred, weights=self.loss_weights) + self.beta * model_utils.l_clus(rep_vars)

        # Compute gradients
        pred_grad = tape.gradient(target=l_pred, sources=pred_vars)
        enc_id_grad = tape.gradient(target=l_enc_id, sources=enc_id_vars)
        clus_grad = tape.gradient(target=l_clus, sources=rep_vars)

        # Apply gradients
        self.optimizer.apply_gradients(zip(pred_grad, pred_vars))
        self.optimizer.apply_gradients(zip(enc_id_grad, enc_id_vars))
        self.cluster_opt.apply_gradients(zip([clus_grad], [rep_vars]))

        # Update Loss functions to Print
        L1.update_state(l_pred)
        L2.update_state(l_enc_id)
        L3.update_state(l_clus)

        return {'L1': L1.result(), 'L2': L2.result(), 'L3': L3.result()}

    def test_step(self, inputs):
        """Main method specifying how model is trained."""
        x, y = inputs
        y_pred, pi = self.forward_pass(x)

        # compute losses
        l_pred = model_utils.l_pred(y, y_pred, weights=self.loss_weights)
        l_enc_id = model_utils.l_pred(y, y_pred, weights=self.loss_weights) + self.alpha * model_utils.l_dist(pi)
        l_clus = model_utils.l_pred(y, y_pred, weights=self.loss_weights) + self.beta * model_utils.l_clus(self.cluster_rep)

        # Update Loss functions to Print
        val_L1.update_state(l_pred)
        val_L2.update_state(l_enc_id)
        val_L3.update_state(l_clus)

        return {'L1': val_L1.result(), 'L2': val_L2.result(), 'L3': val_L3.result()}

    # MODEL INITIALISATION METHODS
    def initialise_model(self, data, val_data, epochs, learning_rate, batch_size):
        """
        Initialisation Method for Model.

        Params:
            - data: tuple (X, y) of data to train the model.
            - val_data: tuple (X, y) of validation data to see loss performance.
            - epochs: int, number of epochs for training.
            - learning_rate: float, starting learning rate for initialisation training
            - batch_size: int, size of individual batches.
            - kwargs: dictionary arguments for KMeans initialisation.

        Returns:
            Model with initialised parameters and cluster_representations
        """
        # Update weights and initialise optimiser
        reciprocal_prop = 1 / np.sum(data[1], axis = 0)
        self.loss_weights = reciprocal_prop / np.sum(reciprocal_prop)
        self._optimizer_init.learning_rate = learning_rate

        # Go through initialisation steps
        self._initialise_enc_pred(data, val_data, epochs, batch_size)
        clus_pred_oh = self._initialise_clus(data, val_data)

        # Define Identifier targets - subset to training targets first
        train_size = data[0].shape[0]
        clus_pred_train, clus_pred_val = clus_pred_oh[:train_size, :], clus_pred_oh[train_size:, :]
        iden_train_data = data[0], clus_pred_train
        iden_val_data = val_data[0], clus_pred_val

        # Initialise Identifier
        self._initialise_iden(iden_train_data, iden_val_data, epochs, batch_size)

    def _initialise_enc_pred(self, data, val_data, epochs=100, batch_size=64):
        """Initialisation of Encoder-Predictor networks."""

        # Input in the right format
        x, y = data
        x_val, y_val = val_data
        inputs = tf.data.Dataset.from_tensor_slices((x, y)).shuffle(buffer_size=5000).batch(batch_size)

        # Initialise loss tracker
        self._enc_pred_loss_tracker = pd.DataFrame(data=np.nan, index=range(epochs), columns=["train_loss", "val_loss"])
        enc_pred_vars = [var for var in self.trainable_variables if "ENCODER" in var.name or "PREDICTOR" in var.name]

        # Forward Enc-Pred pass and train
        print("-" * 20, "Initialising encoder-predictor training.")
        for epoch in range(epochs):  # Iterate through epochs
            loss = 0
            for step_, (x_batch, y_batch) in enumerate(inputs):  # Iterate through batch

                # One Training Step
                with tf.GradientTape(watch_accessed_variables=False) as tape:
                    tape.watch(enc_pred_vars)

                    # Prediction and loss
                    y_pred = self.Predictor(self.Encoder(x_batch))
                    loss_batch = model_utils.l_pred(y_batch, y_pred, weights=self.loss_weights)

                # Update gradients
                enc_pred_grad = tape.gradient(loss_batch, enc_pred_vars)
                self._optimizer_init.apply_gradients(zip(enc_pred_grad, enc_pred_vars))

                # Update loss
                loss += loss_batch

                # Print current batch loss - clears line and re-writes
                print("Batch Loss %.4f" % loss_batch, end="\r", flush=True)

            # Compute validation loss on validation data
            y_val_pred = self.Predictor(self.Encoder(x_val))
            loss_val = model_utils.l_pred(y_val, y_val_pred, weights=self.loss_weights)

            # Print result and update tracker
            print("End of epoch %d - \n Training loss: %.4f  Validation loss %.4f" % (epoch, loss / step_, loss_val))
            self._enc_pred_loss_tracker.loc[epoch, :] = [loss / step_, loss_val]
            
            # Check if has improved or not
            if self._enc_pred_loss_tracker.iloc[-50:, -1].le(loss_val + 0.001).any():
                break

    def _initialise_clus(self, data, val_data, **kwargs):
        """After initialisation of Enc_Pred, cluster representation initialisation."""
        X = np.concatenate((data[0], val_data[0]), axis=0)
        y = np.concatenate((data[1], val_data[1]), axis=0)

        # Compute Latent Projections
        print("-" * 20, "Initialising cluster representations.")
        z = self.Encoder(X).numpy()
        y_pred = self.Predictor(self.Encoder(X)).numpy()

        # Fit KMeans
        km = KMeans(n_clusters=self.K, init="k-means++", random_state=self.seed, **kwargs)
        km.fit(z)
        print("KMeans fit has completed.")

        # Centers are figureholders for representations and
        centers = km.cluster_centers_
        cluster_pred = km.predict(z)
        print("Initialised Phenotypes: ", self.Predictor(centers))
        print("Estimated Cluster distribution: ", np.unique(cluster_pred, return_counts=True))

        # Initialise embeddings and convert to one-hot encoding for Identifier
        self.cluster_rep.assign(tf.convert_to_tensor(centers, name='cluster_rep', dtype='float32'))
        cluster_pred_oh = np.eye(self.K)[cluster_pred]

        return cluster_pred_oh.astype(np.float32)

    def _initialise_iden(self, data, val_data, epochs=100, batch_size=64):
        """Initialisation of Identifier network."""
        # Input in the right format
        X, clus = data
        X_val, clus_val = val_data
        inputs = tf.data.Dataset.from_tensor_slices((X, clus)).shuffle(buffer_size=5000).batch(batch_size)

        # Initialise loss tracker
        self._iden_loss_tracker = pd.DataFrame(data=np.nan, index=range(epochs), columns=["train_loss", "val_loss"])
        iden_vars = [var for var in self.trainable_variables if "IDENTIFIER" in var.name]

        # Forward Identifier pass and train
        print("-" * 20, "Initialising Identifier training.")
        for epoch in range(epochs):  # Iterate through epochs
            loss = 0
            for step_, (x_batch, clus_batch) in enumerate(inputs):  # Iterate through batch

                # One Training Step
                with tf.GradientTape(watch_accessed_variables=False) as tape:
                    tape.watch(iden_vars)

                    # Prediction and loss
                    clus_pred = self.Identifier(self.Encoder(x_batch))
                    loss_batch = model_utils.l_pred(clus_batch, clus_pred)

                # Update gradients
                iden_grad = tape.gradient(loss_batch, iden_vars)
                self._optimizer_init.apply_gradients(zip(iden_grad, iden_vars))

                # Update loss
                loss += loss_batch

                # Print current batch loss - clears line and re-writes
                print("Batch Loss %.4f" % loss_batch, end="\r", flush=True)

            # Compute validation loss on validation data
            clus_val_pred = self.Identifier(self.Encoder(X_val))
            loss_val = model_utils.l_pred(clus_val, clus_val_pred)

            # Print result and update tracker
            print("End of epoch %d - \n Training loss: %.4f  Validation loss %.4f" % (epoch, loss / step_, loss_val))
            self._iden_loss_tracker.loc[epoch, :] = [loss / step_, loss_val]

            # Check if has improved or not - look at last 50 epoch validation loss and check if 
            if self._iden_loss_tracker.iloc[-50:, -1].le(loss_val + 0.001).any():
                break

    # USEFUL METHODS
    
    def compute_attention_rnn_encoder_scores(self, inputs):
        """Computes alpha, beta, gamma scores with custom Encoder layer"""
        scores = self.Encoder.compute_attention_map_scores(inputs, cluster_reps=self.cluster_rep)

        return scores

    def estimate_alpha_beta(self, inputs):
        """Compute alpha, beta estimates as per optimal approximation."""
        scores = self.Encoder.estimate_alpha_beta(inputs)

        return scores
    
    def compute_clus_phens(self):
        """Compute Phenotypes given cluster representations."""
        phens = self.Predictor(self.cluster_rep).numpy()
        
        return phens
    
    def predict_clus(self, X):
        """Compute cluster assignments."""
        pi = self.Identifier(self.Encoder(X)).numpy()
        clus_pred = np.argmax(pi, axis = 1)
        
        return clus_pred
    
    def predict_latent_clus(self, z):
        """Compute cluster assignments for samples on latent space."""
        clus_pred = self.Identifier(z).numpy()
        
        return clus_pred
    
    def compute_latents(self, X):
        """Compute latent projections given input data."""
        z = self.Encoder(X).numpy()
        
        return z
    
    def get_cluster_reps(self):
        return self.cluster_rep.numpy()

    def compute_pis(self, X):
        """Obtain cluster assignment probabilities."""
        pis = self.Identifier(self.Encoder(X))
        
        return pis.numpy()
    

    # LOGISTICAL METHODS
    @classmethod
    def from_config(cls, config):
        return cls(**config)

    @property
    def metrics(self):
        """Return initialised metrics"""
        return [L1, L2, L3, val_L1, val_L2, val_L3]

    def get_config(self):
        """Update configuration for layer."""
        config = super().get_config().copy()
        config.update({"latent_dim": self.latent_dim, "hidden_layers": self.hidden_layers,
                       "hidde_nodes": self.hidden_nodes, "state_fn": self.state_fn,
                       "recurrent_fn": self.recurrent_fn, "return_sequences": self.return_sequences,
                       "dropout": self.dropout, "recurrent_dropout": self.recurrent_dropout,
                       "regulariser_params": self.regulariser_params})

        return config


