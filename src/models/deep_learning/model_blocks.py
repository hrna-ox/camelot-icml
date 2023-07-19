"""
File to define useful model block components. Includes other blocks that can be played with.
"""

# ------------ Import Libraries ---------------
import tensorflow as tf
from tensorflow import linalg
from tensorflow.keras.layers import Dense, Dropout, Layer, LSTM
from tensorflow.keras.regularizers import l1_l2 as l1_l2_reg
from tensorflow.keras.initializers import GlorotNormal, GlorotUniform


# ------------ Utility Functions --------------


def _estimate_alpha(feature_reps, targets):
    """
    alpha parameters OLS estimation given projected input features and targets.

    Params:
    - feature_reps: array-like of shape (bs, T, d, units)
    - targets: array-like of shape (bs, T, units)

    returns:
    - un-normalised alpha weights: array-like of shape (bs, T, d)
    """
    X_T, X = feature_reps, linalg.matrix_transpose(feature_reps)

    # Compute matrix inversion
    X_TX_inv = linalg.inv(linalg.matmul(X_T, X))
    X_Ty = linalg.matmul(X_T, tf.expand_dims(targets, axis=-1))

    # Compute likely scores
    alpha_hat = linalg.matmul(X_TX_inv, X_Ty)

    return tf.squeeze(alpha_hat)


def _estimate_gamma(o_hat, cluster_targets):
    """
    Estimate gamma parameters through OLS estimation given projected input features and targets.

    Params:
    - o_hat: array-like of shape (bs, T, units)
    - targets: array-like of shape (K, units)

    returns:
    - gamma_weights: array-like of shape (bs, K, T)
    """
    X_T = tf.expand_dims(o_hat, axis=1)
    X = linalg.matrix_transpose(X_T)
    y = tf.expand_dims(tf.expand_dims(cluster_targets, axis=0), axis=-1)

    # Compute inversion
    X_TX_inv = linalg.inv(linalg.matmul(X_T, X))
    X_Ty = linalg.matmul(X_T, y)

    # Compute gamma
    gamma_hat = linalg.matmul(X_TX_inv, X_Ty)

    return tf.squeeze(gamma_hat)


def _norm_abs(array, axis: int = 1):
    """
    Compute L1 normalisation of array according to axis.

    Params:
    - array: array-like object.
    - axis: integer.

    returns:
    - normalised array according to axis.
    """
    array_abs = tf.math.abs(array) + 1e-8

    # Normalise according to L1
    output = array_abs / tf.reduce_sum(array_abs, axis=axis, keepdims=True)

    return output


# ------------ MLP definition ---------------
class MLP(Layer):
    """
    Multi-layer perceptron (MLP) neural network architecture.

    Params:
    - output_dim : int, dimensionality of output space for each sub-sequence.
    - hidden_layers : int, Number of "hidden" feedforward layers. (default = 2)
    - hidden_nodes : int, For "hidden" feedforward layers, the dimensionality of the output space. (default = 30)
    - activation_fn : str/fn, The activation function to use. (default = 'sigmoid')
    - output_fn : str/fn, The activation function on the output of the MLP. (default = 'softmax').
    - dropout : float, dropout rate (default = 0.6).
    - regulariser_params : tuple of floats for regularization (default = (0.01, 0.01))
    - seed : int, Seed used for random mechanisms (default = 4347)
    - name : str, name on which to save layer. (default = 'MLP')
    """

    def __init__(self, output_dim: int, hidden_layers: int = 2, hidden_nodes: int = 30, activation_fn='sigmoid',
                 output_fn='softmax', dropout: float = 0.6, regulariser_params: tuple = (0.01, 0.01), seed: int = 4347,
                 name: str = 'MLP'):

        # Block parameters
        super().__init__(name=name)
        self.output_dim = output_dim
        self.hidden_layers = hidden_layers
        self.hidden_nodes = hidden_nodes
        self.activation_fn = activation_fn
        self.output_fn = output_fn

        # Regularization params
        self.dropout = dropout
        self.regulariser = l1_l2_reg(*regulariser_params)

        # Seed
        self.seed = seed

        # Add intermediate layers to the model
        for layer_id_ in range(self.hidden_layers):

            # Add Dense layer to model
            layer_ = Dense(units=self.hidden_nodes, activation=self.activation_fn, kernel_regularizer=self.regulariser,
                           activity_regularizer=self.regulariser, kernel_initializer=GlorotNormal(seed=self.seed),
                            bias_initializer=GlorotUniform(seed=self.seed))

            self.__setattr__('layer_' + str(layer_id_), layer_)

            # Add corresponding Dropout Layer
            dropout_layer = Dropout(rate=self.dropout, seed=self.seed + layer_id_)
            self.__setattr__('dropout_layer_' + str(layer_id_), dropout_layer)

        # Input and Output layers
        self.output_layer = Dense(units=self.output_dim, activation=self.output_fn,
                                  kernel_initializer=GlorotNormal(seed=self.seed),
                                  bias_initializer=GlorotUniform(seed=self.seed))

    def call(self, inputs, training=True, **kwargs):
        """Forward pass of layer block."""
        x_inter = inputs

        # Iterate through hidden layer computation
        for layer_id_ in range(self.hidden_layers):

            # Get layers and apply
            layer_ = self.__getattribute__('layer_' + str(layer_id_))
            dropout_layer_ = self.__getattribute__('dropout_layer_' + str(layer_id_))

            # Make layer computations
            x_inter = dropout_layer_(layer_(x_inter, training=training))

        return self.output_layer(x_inter, training=training)

    def get_config(self):
        """Update configuration for layer."""

        # Load existing configuration
        config = super().get_config().copy()

        # Update configuration
        config.update({f"{self.name}-output_dim": self.output_dim,
                       f"{self.name}-hidden_layers": self.hidden_layers,
                       f"{self.name}-hidden_nodes": self.hidden_nodes,
                       f"{self.name}-activation_fn": self.activation_fn,
                       f"{self.name}-output_fn": self.output_fn,
                       f"{self.name}-dropout": self.dropout,
                       f"{self.name}-seed": self.seed})

        return config


class FeatTimeAttention(Layer):
    """
    Custom Feature Attention Layer. Features are projected to latent dimension and approximate output RNN states.
    Approximations are sum-weighted to obtain a final representation.

    Params:
    units: int, dimensionality of projection/latent space.
    activation: str/fn, the activation function to use. (default = "relu")
    name: str, the name on which to save the layer. (default = "custom_att_layer")
    """

    def __init__(self, units: int, activation: str = "linear", name: str = "custom_layer", seed: int =4347):

        # Load layer params
        super().__init__(name=name)

        # Initialise key layer attributes
        self.units = units
        self.activation_name = activation
        self.activation = tf.keras.activations.get(activation)  # get activation from  identifier
        self.seed = seed

        # Initialise layer weights to None
        self.kernel = None
        self.bias = None
        self.unnorm_beta_weights = None


    def build(self, input_shape=None):
        """Build method for the layer given input shape."""
        N, T, Df = input_shape

        # kernel, bias for feature -> latent space conversion
        self.kernel = self.add_weight("kernel", shape=[1, 1, Df, self.units],
                                      initializer=GlorotNormal(seed=self.seed),
                                      trainable=True)
        self.bias = self.add_weight("bias", shape=[1, 1, Df, self.units],
                                    initializer=GlorotUniform(self.seed), trainable=True)

        # Time aggregation learn weights
        self.unnorm_beta_weights = self.add_weight(name='time_agg', shape=[1, T, 1],
                                                   initializer=GlorotUniform(seed=self.seed), trainable=True)

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        """
        Forward pass of the Custom layer - requires inputs and estimated latent projections.

        Params:
        - inputs: tuple of two arrays:
            - x: array-like of input data of shape (bs, T, D_f)
            - latent_reps: array-like of representations of shape (bs, T, units)

        returns:
        - latent_representation (z): array-like of shape (bs, units)
        """

        # Unpack input
        x, latent_reps = inputs

        # Compute output state approximations
        o_hat, _ = self.compute_o_hat_and_alpha(x, latent_reps)

        # Normalise temporal weights and sum-weight approximations to obtain representation
        beta_scores = _norm_abs(self.unnorm_beta_weights)
        z = tf.reduce_sum(tf.math.multiply(o_hat, beta_scores), axis=1)

        return z

    def compute_o_hat_and_alpha(self, x, latent_reps):
        """
        Compute approximation to latent representations, given input feature data.

        Params:
        - x: array-like of shape (bs, T, D_f)
        - latent_reps: array-like of shape (bs, T, units)

        returns:
        - output: tuple of arrays:
           - array-like of shape (bs, T, units) of representation approximations
           - array-like of shape (bs, T, D_f) of alpha_weights
        """

        # Compute feature projections
        feature_projections = self.activation(tf.math.multiply(tf.expand_dims(x, axis=-1), self.kernel) + self.bias)

        # estimate alpha coefficients through OLS
        alpha_t = _estimate_alpha(feature_projections, targets=latent_reps)

        # sum-weight feature projections according to alpha_t to compute latent approximations
        o_hat = tf.reduce_sum(tf.math.multiply(tf.expand_dims(alpha_t, axis=-1), feature_projections), axis=2)

        return o_hat, alpha_t

    def compute_unnorm_scores(self, inputs, latent_reps, cluster_reps=None):
        """
        Compute unnormalised weights for attention values.

        Params:
        - inputs: array-like of shape (bs, T, D_f) of input data
        - latent_reps: array-like of shape (bs, T, units) of RNN cell output states.
        - cluster_reps: array-like of shape (K, units) of cluster representation vectors (default = None). If None,
        gamma computation is skipped.

        Returns:
            - output: tuple of arrays (alpha, beta, gamma) with corresponding values. If cluster_reps is None,
        gamma computation is skipped.
        """

        # Compute alpha weights
        o_hat, alpha_t = self.compute_o_hat_and_alpha(inputs, latent_reps)

        # Load beta weights
        beta = self.unnorm_beta_weights

        # If cluster_reps not None, compute gamma
        if cluster_reps is None:
            gamma_t_k = None
        else:
            gamma_t_k = _estimate_gamma(o_hat, cluster_reps)

        return alpha_t, beta, gamma_t_k

    def compute_norm_scores(self, x, latent_reps, cluster_reps=None):
        """
        Compute normalised attention scores alpha, beta, gamma.

        Params:
        - x: array-like of shape (bs, T, D_f) of input data
        - latent_reps: array-like of shape (bs, T, units) of RNN cell output states.
        - cluster_reps: array-like of shape (K, units) of cluster representation vectors (default = None). If None,
        gamma computation is skipped.

        Returns:
            - output: tuple of arrays (alpha, beta, gamma) with corresponding normalised scores. If cluster_reps
        is None, gamma computation is skipped.
        """

        # Load unnormalised scores
        alpha, beta, gamma = self.compute_unnorm_scores(x, latent_reps, cluster_reps)

        # Normalise
        alpha_norm = _norm_abs(alpha, axis=1)
        beta_norm = _norm_abs(beta, axis=1)

        if gamma is None:
            gamma_norm = None
        else:
            gamma_norm = _norm_abs(gamma, axis=1)

        return alpha_norm, beta_norm, gamma_norm

    def get_config(self):
        """Update configuration for layer."""

        # Load existing configuration
        config = super().get_config().copy()

        # Update configuration
        config.update({f"{self.name}-units": self.units,
                       f"{self.name}-activation": self.activation_name})

        return config


class LSTMEncoder(Layer):
    """
        Class for a stacked LSTM layer architecture.

        Params:
        - latent_dim : dimensionality of latent space for each sub-sequence. (default = 32)
        - hidden_layers : Number of "hidden"/intermediate LSTM layers.  (default = 1)
        - hidden_nodes : For hidden LSTM layers, the dimensionality of the intermediate state. (default = 20)
        - state_fn : The activation function to use on cell state/output. (default = 'tanh')
        - recurrent_activation : The activation function to use on forget/input/output gates. (default = 'sigmoid')
        - return_sequences : Indicates if returns sequence of states on the last layer (default = False)
        - dropout : dropout rate to be used on cell state/output computation. (default = 0.6)
        - recurrent_dropout : dropout rate to be used on forget/input/output gates. (default = 0.0)
        - regulariser_params :  tuple of floats indicating l1_l2 regularisation. (default = (0.01, 0.01))
        - name : Name on which to save component. (default = 'LSTM_Encoder')
    """

    def __init__(self, latent_dim: int = 32, hidden_layers: int = 1, hidden_nodes: int = 20, state_fn="tanh",
                 recurrent_fn="sigmoid", regulariser_params: tuple = (0.01, 0.01), return_sequences: bool = False,
                 dropout: float = 0.6, recurrent_dropout: float = 0.0,
                 seed: int = 4347, name: str = 'LSTM_Encoder'):

        # Block Parameters
        super().__init__(name=name)
        self.latent_dim = latent_dim
        self.hidden_layers = hidden_layers
        self.hidden_nodes = hidden_nodes
        self.state_fn = state_fn
        self.recurrent_fn = recurrent_fn
        self.return_sequences = return_sequences

        # Regularisation Params
        self.dropout = dropout
        self.recurrent_dropout = recurrent_dropout
        self.regulariser_params = regulariser_params
        self.regulariser = l1_l2_reg(*regulariser_params)
        self.seed = seed

        # Add Intermediate Layers
        for layer_id_ in range(self.hidden_layers):
            self.__setattr__('layer_' + str(layer_id_),
                             LSTM(units=self.hidden_nodes, return_sequences=True, activation=self.state_fn,
                                  recurrent_activation=self.recurrent_fn, dropout=self.dropout,
                                  recurrent_dropout=self.recurrent_dropout,
                                  kernel_initializer=GlorotNormal(seed=self.seed),
                                  bias_initializer=GlorotUniform(seed=self.seed),
                                  kernel_regularizer=self.regulariser, recurrent_regularizer=self.regulariser,
                                  bias_regularizer=self.regulariser, return_state=False))

        self.output_layer = LSTM(units=self.latent_dim, activation=self.state_fn,
                                 recurrent_activation=self.recurrent_fn, return_sequences=self.return_sequences,
                                 dropout=self.dropout, recurrent_dropout=self.recurrent_dropout,
                                 kernel_initializer=GlorotNormal(seed=self.seed),
                                 bias_initializer=GlorotUniform(seed=self.seed),
                                 kernel_regularizer=self.regulariser, recurrent_regularizer=self.regulariser,
                                 bias_regularizer=self.regulariser, return_state=False)

    def call(self, inputs, mask=None, training=True, **kwargs):
        """Forward pass of layer block."""
        x_inter = inputs

        # Iterate through hidden layer computation
        for layer_id_ in range(self.hidden_layers):
            layer_ = self.__getattribute__('layer_' + str(layer_id_))
            x_inter = layer_(x_inter, training=training)

        return self.output_layer(x_inter, training=training)

    def get_config(self):
        """Update configuration for layer."""

        # Load existing configuration
        config = super().get_config().copy()

        # Update configuration
        config.update({f"{self.name}-latent_dim": self.latent_dim,
                       f"{self.name}-hidden_layers": self.hidden_layers,
                       f"{self.name}-hidden_nodes": self.hidden_nodes,
                       f"{self.name}-state_fn": self.state_fn,
                       f"{self.name}-recurrent_fn": self.recurrent_fn,
                       f"{self.name}-return_sequences": self.return_sequences,
                       f"{self.name}-dropout": self.dropout,
                       f"{self.name}-recurrent_dropout": self.recurrent_dropout,
                       f"{self.name}-regulariser_params": self.regulariser_params})

        return config


class AttentionRNNEncoder(LSTMEncoder):
    """
        Class for an Attention RNN Encoder architecture. Class builds on LSTM Encoder class.
    """

    def __init__(self, units, activation="linear", seed: int = 4347, **kwargs):
        super().__init__(latent_dim=units, return_sequences=True, seed=seed, **kwargs)
        self.feat_time_attention_layer = FeatTimeAttention(units=units, activation=activation, seed=seed)

    def call(self, x, mask=None, training: bool = True, **kwargs):
        """
        Forward pass of layer block.

        Params:
        - x: array-like of shape (bs, T, D_f)
        - mask: array-like of shape (bs, T) (default = None)
        - training: bool indicating whether to make computation in training mode or not. (default = True)

        Returns:
        - z: array-like of shape (bs, units)
        """

        # Compute LSTM output states
        latent_reps = super().call(x, mask=mask, training=training, **kwargs)

        # Compute representation through feature time attention layer
        attention_inputs = (x, latent_reps)
        z = self.feat_time_attention_layer(attention_inputs)

        return z

    def compute_unnorm_scores(self, x, cluster_reps=None):
        """
        Compute unnormalised scores alpha, beta, gamma given input data and cluster representation vectors.

        Params:
        - x: array-like of shape (bs, T, D_f)
        - cluster_reps: array-like of shape (K, units) of cluster representation vectors. (default = None)

        If cluster_reps is None, compute only alpha and beta weights.

        Returns:
        - Tuple of arrays, containing alpha, beta, gamma unnormalised attention weights.
        """
        latent_reps = super().call(x, training=False)

        return self.feat_time_attention_layer.compute_unnorm_scores(x, latent_reps, cluster_reps)

    def compute_norm_scores(self, inputs, cluster_reps=None):
        """Compute normalised scores alpha, beta, gamma given input data and cluster representation vectors.

        Params:
        - inputs: array-like of shape (bs, T, D_f)
        - cluster_reps: array-like of shape (K, units) of cluster representation vectors. (default = None)

        If cluster_reps is None, compute only alpha and beta weights.

        Returns:
        - Tuple of arrays, containing alpha, beta, gamma normalised attention weights.
        """
        latent_reps = super().call(inputs, training=False)

        return self.feat_time_attention_layer.compute_norm_scores(inputs, latent_reps, cluster_reps)

    def get_config(self):
        """Update configuration for layer."""
        config = super().get_config().copy()

        # Update
        custom_layer_config = self.feat_time_attention_layer.get_config().copy()
        config = {**custom_layer_config, **config}

        return config
