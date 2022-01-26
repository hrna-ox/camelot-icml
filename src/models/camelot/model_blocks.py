"""
File to define useful model block components. Includes other blocks that can be played with.
"""

# ------------ Import Libraries ---------------
import tensorflow as tf
from tensorflow import linalg
from tensorflow.keras.layers import Dense, Dropout, Layer, LSTM
from tensorflow.keras.regularizers import l1_l2 as mix_l1_l2_reg


# ------------ Utility Functions --------------
def _norm_abs(inputs, axis=1):
    """
    Given input data, compute normalisation according to L1 metric for given axis 1.
    """
    abs_input = tf.math.abs(inputs)  # Compute vector of absolute values.

    return tf.math.divide(abs_input, tf.reduce_sum(abs_input, axis=axis, keepdims=True))


def _estimate_alpha(feature_reps, targets):
    """
    alpha parameters OLS estimation given projected input features and targets.
    feature_reps: shape (bs, T, d, units)
    targets: shape (bs, T, units)
    """
    X_T, X = feature_reps, linalg.matrix_transpose(feature_reps)

    # Compute matrix inversion
    X_TX_inv = linalg.inv(linalg.matmul(X_T, X))
    X_Ty = linalg.matmul(X_T, tf.expand_dims(targets, axis=-1))

    # Compute likely scores
    alpha_hat = linalg.matmul(X_TX_inv, X_Ty)

    return tf.squeeze(alpha_hat)  # shape (bs, T, d) (NOT normalised)


def _estimate_gamma(o_hat, cluster_targets):
    """
    Estimate gamma parameters through OLS estimation given projected input features and targets.
    o_hat: shape (bs, T, units)
    targets: shape (K, units)
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


# ------------ MLP definition ---------------
class MLP(Layer):
    """
    Multi-layer perceptron (MLP) neural network architecture.

    Params:
    - output_dim : int, dimensionality of output space for each sub-sequence.
    - hidden_layers : int, Number of "hidden" feedforward layers. (default = 2)
    - hidden_nodes : int, For "hidden" feedforward layers, the dimensionality  of the output space. (default = 30)
    - activation_fn : str/fn, The activation function to use. (default = 'sigmoid')
    - output_fn : str/fn, The activation function on the output of the MLP. (default = 'softmax').
    - dropout : float, dropout rate (default = 0.6).
    - reg_params : tuple of floats for regularization (default = (0.01, 0.01))
    - seed : int, Seed used for random mechanisms (default = 4347)
    - name : str, name on which to save layer. (default = 'decoder')
    """

    def __init__(self, output_dim: int, hidden_layers: int = 2, hidden_nodes: int = 30, activation_fn='sigmoid',
                 output_fn='softmax', dropout: float = 0.6, reg_params=(0.01, 0.01), seed: int = 4347,
                 name: float = 'MLP'):

        # Block parameters
        super().__init__(name=name)
        self.output_dim = output_dim
        self.hidden_layers = hidden_layers
        self.hidden_nodes = hidden_nodes
        self.activation_fn = activation_fn
        self.output_fn = output_fn

        # Regularization params
        self.dropout = dropout
        self.regulariser = mix_l1_l2_reg(reg_params)

        # Seed
        self.seed = seed

        # Add intermediate layers to the model
        for layer_id_ in range(1, self.hidden_layers + 1):
            # Add Dense layer to model
            layer_ = Dense(units=self.hidden_nodes,
                           activation=self.activation_fn,
                           kernel_regularizer=self.regulariser,
                           activity_regularizer=self.regulariser)

            self.__setattr__('layer_' + str(layer_id_), layer_)

        # Add Input and Output Layers
        self.input_layer = Dense(units=self.hidden_nodes, activation=self.activation_fn)
        self.output_layer = Dense(units=self.output_dim, activation=self.output_fn)

        # Dropout layer
        self.dropout_layer = Dropout(rate=self.dropout, seed=self.seed)

    def call(self, inputs, training=True, **kwargs):
        """Forward pass of layer block."""
        x_inter = self.input_layer(inputs)

        # Iterate through hidden layer computation
        for layer_id_ in range(self.hidden_layers):
            # Get layer and run on input
            layer_ = self.__getattribute__('layer_' + str(layer_id_))
            x_inter = self.dropout_layer(layer_(x_inter, training=training), seed=self.seed)

        return self.output_layer(x_inter, training=training, **kwargs)

    def get_config(self):
        """Update configuration for layer."""

        # Get overarching configuration
        config = super().get_config().copy()

        # Update configuration with parameters above
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
    Feature Time Attention Layer. Computes approximations when given output RNN cell states.

    Params:
    units: int, dimensionality of projection/latent space.
    activation: str/fn, the activation function to use. (default = "relu")
    name: str, the name on which to save the layer. (default = "custom_att_layer")
    """

    def __init__(self, units: int, activation="relu", name: str = "custom_layer"):
        super().__init__(name=name)

        # Layer hyper-params
        self.units = units
        self.activation_name = activation
        self.activation = tf.keras.activations.get(activation)  # get activation from  identifier

        # Layer kernel and bias initialised to None (updated on build method)
        self.kernel = None
        self.bias = None
        self.unnorm_beta = None

    def build(self, input_shape=None):
        """Build method for initialising layers when seeing input."""

        # Initialise dimensions
        N, T, D_f = input_shape

        # Define Kernel and Bias for Feature Projection
        self.kernel = self.add_weight("kernel", shape=[1, 1, D_f, self.units],
                                      initializer="glorot_uniform", trainable=True)
        self.bias = self.add_weight("bias", shape=[1, 1, D_f, self.units],
                                    initializer='uniform', trainable=True)

        # Define Time aggregation weights for averaging over time.
        self.unnorm_beta = self.add_weight(name='unnorm_beta_weights', shape=[1, T, 1],
                                           initializer="uniform", trainable=True)

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        """
        Forward pass of the Feature Time layer - requires inputs and estimated latent projections.

        Params:
        inputs: Tuple of tensors
            Input data - Tensor of shape (batch size, T, Df)
            Latent_reps - Tensor of shape (batch_size, T, latent dim)

        Approximations to latent_reps are computed. At a second stage, approximations are sum-weighted according
        to the beta weights.
        """

        # Load tuple
        input_data, latent_reps = inputs

        # Generate latent vector approximation
        o_hat, _ = self.compute_output_state_approximations(input_data, latent_reps)  # shape (bs, T, latent_dim)

        # Weighted sum over time. Weights are not normalised, hence must be converted through softmax.
        time_weights = _norm_abs(self.unnorm_beta)  # shape (1, T, 1)
        z = tf.reduce_sum(tf.math.multiply(o_hat, time_weights), axis=1)  # shape (bs, latent_dim)

        return z

    def compute_output_state_approximations(self, inputs, latent_reps):
        """Given input and targets (latent_reps), compute OLS approximation to targets and weights."""

        # Compute feature projections
        feature_linear_proj = tf.math.multiply(tf.expand_dims(inputs, axis=-1), self.kernel) + self.bias
        feature_projections = self.activation(feature_linear_proj)

        # Estimate OLS coefficients
        alpha_t = _estimate_alpha(feature_projections, targets=latent_reps)

        # Compute OLS estimates given coefficients
        o_hat = tf.reduce_sum(tf.math.multiply(tf.expand_dims(alpha_t, axis=-1), feature_projections), axis=2)

        return o_hat, alpha_t

    def compute_unnorm_scores(self, inputs, latent_reps, cluster_reps):
        """
        Compute unnormalised alpha, beta, gamma scores given
        input data
        latent_representation
        cluster_representations (if None, ignore gamma).
        """

        # alpha estimated from OLS regression
        o_hat, alpha_t = self.compute_output_state_approximations(inputs, latent_reps)  # shape bs, T, D_f

        # beta estimated from beta_weights
        beta = self.unnorm_beta  # shape (1, T, 1)

        # gamma estimated from approximation to cluster reps
        if cluster_reps is None:
            gamma_t_k = None
        else:
            gamma_t_k = _estimate_gamma(o_hat, cluster_reps)  # shape (bs, K, T)

        return alpha_t, beta, gamma_t_k

    def compute_norm_scores(self, inputs, latent_reps, cluster_reps):
        """Compute normalised scores alpha, beta, gamma."""

        # Load weights
        alpha, beta, gamma = self.compute_unnorm_scores(inputs, latent_reps, cluster_reps)

        # Normalise
        alpha = _norm_abs(alpha, axis=1)
        beta = _norm_abs(beta, axis=1)

        # Check if gamma is None
        if gamma is None:
            gamma = None
        else:
            gamma = _norm_abs(gamma, axis=1)

        return alpha, beta, gamma

    def get_config(self):
        """Update configuration for layer."""

        # Get overarching configuration
        config = super().get_config().copy()

        # Update configuration with parameters above
        config.update({f"{self.name}-units": self.units,
                       f"{self.name}-activation": self.activation_name
                       })

        return config


class LSTMEncoder(Layer):
    """
        Class for a stacked LSTM layer architecture.

        Params:
        - latent_dim : dimensionality of latent space for each sub-sequence. (default = 32)
        - hidden_layers : Number of "hidden"/intermediate LSTM layers.  (default = 1)
        - hidden_nodes : For hidden LSTM layers, the dimensionality of the intermediate state. (default = 32)
        - state_fn : The activation function to use on cell state/output. (default = 'tanh')
        - recurrent_activation : The activation function to use on forget/input/output gates. (default = 'sigmoid')
        - return_sequences : Indicates if returns sequence of states on the last layer (default = False)
        - dropout : dropout rate to be used on cell state/output computation. (default = 0.6)
        - recurrent_dropout : dropout rate to be used on forget/input/output gates. (default = 0.0)
        - regulariser_params :  tuple of floats indicating l1_l2 regularisation. (default = (0.01, 0.01))
        - name : Name on which to save component. (default = 'encoder')
        - seed : seed value for dropout layer (default = 4347)
    """

    def __init__(self, latent_dim: int = 32, hidden_layers: int = 1, hidden_nodes: int = 20, state_fn: str = "tanh",
                 recurrent_fn: str = "sigmoid", regulariser_param: tuple = (0.01, 0.01), return_sequences: bool = False,
                 dropout: float = 0.6, recurrent_dropout: float = 0.0, name: str = 'LSTM_Encoder', seed: int = 4347):

        # Block Parameters
        super().__init__(name=name)
        self.latent_dim = latent_dim
        self.hidden_layers = hidden_layers
        self.hidden_nodes = hidden_nodes
        self.state_fn = state_fn
        self.recurrent_fn = recurrent_fn
        self.return_sequences = return_sequences

        # Regularisation parameters
        self.dropout = dropout
        self.recurrent_dropout = recurrent_dropout
        self.regulariser_params = regulariser_param
        self.regulariser = mix_l1_l2_reg(regulariser_param)

        # Seed
        self.seed = seed

        # Add Hidden/Intermediate Layers
        for layer_id_ in range(1, self.hidden_layers + 1):
            self.__setattr__('layer_' + str(layer_id_),
                             LSTM(units=self.hidden_nodes, return_sequences=True, activation=self.state_fn,
                                  recurrent_activation=self.recurrent_fn, dropout=self.dropout,
                                  recurrent_dropout=self.recurrent_dropout,
                                  kernel_regularizer=self.regulariser, recurrent_regularizer=self.regulariser,
                                  bias_regularizer=self.regulariser, return_state=False))

        # Add Input and Output Layers
        self.input_layer = LSTM(units=self.hidden_nodes, activation=self.state_fn,
                                recurrent_activation=self.recurrent_fn, return_sequences=True,
                                dropout=self.dropout, recurrent_dropout=self.recurrent_dropout,
                                kernel_regularizer=self.regulariser, recurrent_regularizer=self.regulariser,
                                bias_regularizer=self.regulariser, return_state=False)

        self.output_layer = LSTM(units=self.latent_dim, activation=self.state_fn,
                                 recurrent_activation=self.recurrent_fn, return_sequences=self.return_sequences,
                                 dropout=self.dropout, recurrent_dropout=self.recurrent_dropout,
                                 kernel_regularizer=self.regulariser, recurrent_regularizer=self.regulariser,
                                 bias_regularizer=self.regulariser, return_state=False)

    def call(self, inputs, mask=None, training=True):
        """Forward pass of layer block."""
        x_inter = self.input_layer(inputs)

        # Iterate through hidden layer computation
        for layer_id_ in range(1, self.hidden_layers + 1):
            layer_ = self.__getattribute__('layer_' + str(layer_id_))
            x_inter = layer_(x_inter, training=training)

        return self.output_layer(x_inter, training=training)

    def get_config(self):
        """Update configuration for layer."""

        # Get overarching configuration
        config = super().get_config().copy()

        # Update configuration with parameters above
        config.update({f"{self.name}-latent_dim": self.latent_dim,
                       f"{self.name}-hidden_layers": self.hidden_layers,
                       f"{self.name}-hidden_nodes": self.hidden_nodes,
                       f"{self.name}-state_fn": self.state_fn,
                       f"{self.name}-recurrent_fn": self.recurrent_fn,
                       f"{self.name}-return_sequences": self.return_sequences,
                       f"{self.name}-dropout": self.dropout,
                       f"{self.name}-recurrent_dropout": self.recurrent_dropout,
                       f"{self.name}-regulariser_params": self.regulariser_params,
                       f"{self.name}-seed": self.seed})

        return config


class AttentionRNNEncoder(LSTMEncoder):
    """
        Class for an Attention RNN Encoder architecture.

        Params:
    units: int, dimensionality of projection/latent space.
    activation: str/fn, the activation function to use. (default = "relu")
    name: str, name on which to save model.
    """

    def __init__(self, units: int, activation: str = "linear", name: str = "AttentionRNNEncoder", **kwargs):
        super().__init__(latent_dim=units, return_sequences=True, name=name, **kwargs)
        self.feat_time_att_layer = FeatTimeAttention(units=units, activation=activation)

    def call(self, inputs, mask=None, training=True):
        """Forward pass of layer block."""

        # Compute latent representations through RNN Encoder
        latent_reps = super().call(inputs, mask=mask, training=training)

        # Attention layer inputs
        attention_inputs = inputs, latent_reps
        z = self.feat_time_att_layer(attention_inputs)

        return z

    def compute_unnorm_scores(self, inputs, cluster_reps=None):
        """
        Compute alpha, beta, gamma un-normalised scores for cluster estimation.

        Params:
        - inputs: input data
        - cluster_reps: set of cluster representation vectors (default = None)
        """
        latent_reps = super().call(inputs, training=False)

        return self.feat_time_att_layer.compute_unnorm_scores(inputs, latent_reps, cluster_reps)

    def compute_norm_scores(self, inputs, cluster_reps=None):
        """
        Compute alpha, beta, gamma normalised scores for cluster estimation.

        Params:
        - inputs: input data
        - cluster reps: set of cluster representation vectors (default = None).
        """

        # Compute latent representations and estimate alpha
        latent_reps = super().call(inputs, training=False)
        alpha, beta, gamma = self.feat_time_att_layer.compute_norm_scores(inputs, latent_reps, cluster_reps)

        return alpha, beta, gamma

    def get_config(self):
        """Update configuration for layer."""
        return super().get_config()
