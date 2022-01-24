"""
File to define useful model block components. Includes other blocks that can be played with.
"""

# ------------ Import Libraries ---------------
import tensorflow as tf
from tensorflow import linalg
from tensorflow.keras.layers import Dense, Dropout, Layer, LSTM
from tensorflow.keras.regularizers import l1_l2 as mix_l1_l2_reg


# ------------ Utility Functions --------------
def _estimate_alpha(feature_reps, targets):
    """ alpha parameters OLS estimation given projected input features and targets.
    feature_reps: shape (bs, T, d, units)
    targets: shape (bs, T, units)
    """
    X_T, X = feature_reps, linalg.matrix_transpose(feature_reps)

    # Compute matrix inversion
    A_inv = linalg.inv(linalg.matmul(X_T, X))
    X_Ty = linalg.matmul(X_T, tf.expand_dims(targets, axis=-1))

    # Compute likely scores
    alpha_hat = linalg.matmul(A_inv, X_Ty)

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
    A_inv = linalg.inv(linalg.matmul(X_T, X))
    X_Ty = linalg.matmul(X_T, y)

    # Compute gamma
    gamma_hat = linalg.matmul(A_inv, X_Ty)

    return tf.squeeze(gamma_hat)


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
    - dropout : float, dropout rate to be used on layer computation (default = 0.6).
    - regulariser_params : tuple of floats indicating regularization. (default = (0.01, 0.01))
    - seed : int, Seed used for random mechanisms (default = 4347)
    - name : str, name on which to save layer. (defult = 'decoder')
    """

    def __init__(self, output_dim, hidden_layers=2, hidden_nodes=30, activation_fn='sigmoid',
                 output_fn='softmax', dropout=0.6, regulariser_params=(0.01, 0.01), seed=4347, name='decoder'):

        # Block parameters
        super().__init__(name=name)
        self.output_dim = output_dim
        self.hidden_layers = hidden_layers
        self.hidden_nodes = hidden_nodes
        self.activation_fn = activation_fn
        self.output_fn = output_fn
        self.dropout = dropout
        self.seed = seed

        l1_param, l2_param = regulariser_params
        self.regulariser = mix_l1_l2_reg(l1_param, l2_param)

        # Add intermediate layers
        for layer_id_ in range(self.hidden_layers):
            layer_ = Dense(units=self.hidden_nodes, activation=self.activation_fn,
                           kernel_regularizer=self.regulariser, activity_regularizer=self.regulariser)
            self.__setattr__('layer_' + str(layer_id_), layer_)

        # Input and Output layers
        self.input_layer = Dense(units=self.hidden_nodes, activation=self.activation_fn)
        self.output_layer = Dense(units=self.output_dim, activation=self.output_fn)

        # Dropout layer
        self.dropout_layer = Dropout(rate=self.dropout, seed=self.seed)

    def call(self, inputs, training=True):
        """Forward pass of layer block."""
        x_inter = self.input_layer(inputs)

        # Iterate through hidden layer computation
        for layer_id_ in range(self.hidden_layers):
            layer_ = self.__getattribute__('layer_' + str(layer_id_))
            x_inter = layer_(x_inter, training=training)

        return self.output_layer(x_inter, training=training)

    def get_config(self):
        """Update configuration for layer."""
        config = super().get_config().copy()
        config.update({"output_dim": self.output_dim, "hidden_layers": self.hidden_layers,
                       "hidden_nodes": self.hidden_nodes, "activation_fn": self.activation_fn,
                       "output_fn": self.output_fn, "dropout": self.dropout, "seed": self.seed})

        return config


class FeatureTimeAttentionLayer(Layer):
    """
    Custom Feature Attention Layer as proposed in the ICLR submission.

    Params:
    units: int, dimensionality of projection/latent space.
    activation: str/fn, the activation function to use. (default = "relu")
    name: str, the name on which to save the layer. (default = "custom_att_layer")
    """

    def __init__(self, units, activation="linear", name="custom_layer"):
        super().__init__(name=name)
        self.units = units
        self.activation_name = activation
        self.activation = tf.keras.activations.get(activation)  # get activation from  identifier

    def build(self, input_shape=None):
        """Build method for the layer given input shape."""
        N, T, Df = input_shape

        # kernel, bias for feature -> latent space conversion
        self.kernel = self.add_weight("kernel", shape=[1, 1, Df, self.units],
                                      initializer="glorot_uniform", trainable=True)
        self.bias = self.add_weight("bias", shape=[1, 1, Df, self.units],
                                    initializer='uniform', trainable=True)

        # Time aggreggation learn weights
        self.beta_weights = self.add_weight(name='time_agg', shape=[1, T, 1],
                                            initializer="uniform", trainable=True)

        super().build(input_shape)

    def call(self, inputs, latent_reps):
        """
        Forward pass of the Custom layer - requires inputs and estimated latent projections.

        Params:
        inputs: Tensor of shape (batch size, T, Df)
        latent_reps: Tensor of shape (batch_size, T, latent dim)
        """
        o_hat, _ = self.compute_o_hat_and_alpha(inputs, latent_reps)
        z = tf.reduce_sum(tf.math.multiply(o_hat, self.beta_weights), axis=1)

        return z  # shape (bs, units)

    def compute_o_hat_and_alpha(self, inputs, latent_reps):
        """Given input and targets (latent_reps), compute OLS approximation to targets and weights."""
        # Compute features and estimated OLS alpha weights
        feature_projections = self._compute_feature_projections(inputs)
        alpha_t = _estimate_alpha(feature_projections, targets=latent_reps)

        # aggreggate over Df, to compute ohat
        o_hat = tf.reduce_sum(tf.math.multiply(tf.expand_dims(alpha_t, axis=-1), feature_projections), axis=2)

        return o_hat, alpha_t

    def compute_all_scores(self, inputs, latent_reps, cluster_reps):
        """Return alpha, beta, gamma weights for attention map computation. """
        alpha = self._alpha_scores(inputs, latent_reps)
        beta = self._beta_scores()
        gamma = self._gamma_scores(inputs, latent_reps, cluster_reps)

        return alpha, beta, gamma  # shape (bs, T, D), (1, T, 1), (bs, K, T)

    def estimate_gamma(self, inputs, latent_targets, cluster_reps):
        """Estimate gamma_n,t^{k} parameter to approximate cluster_reps"""
        o_hat, _ = self.compute_o_hat_and_alpha(inputs, latent_targets)
        gamma_t_k = _estimate_gamma(o_hat, cluster_reps)

        return gamma_t_k

    def _compute_feature_projections(self, inputs):
        """Feature-Time projection to latent space"""

        linear_map = tf.math.multiply(tf.expand_dims(inputs, axis=-1), self.kernel) + self.bias

        return self.activation(linear_map)

    def _alpha_scores(self, inputs, targets):
        """Compute normalised alpha param estimations."""
        _, alpha_t = self.compute_o_hat_and_alpha(inputs, targets)

        return tf.math.abs(alpha_t) / tf.reduce_sum(tf.math.abs(alpha_t), 
                                                    axis = -1, keepdims = True)

    def _beta_scores(self):
        return tf.math.abs(self.beta_weights) / tf.reduce_sum(tf.math.abs(self.beta_weights), 
                                                    axis = -1, keepdims = True)

    def _gamma_scores(self, inputs, targets, cluster_reps):
        gamma_t_k = self.estimate_gamma(inputs, targets, cluster_reps)

        return tf.math.abs(gamma_t_k) / tf.reduce_sum(tf.math.abs(gamma_t_k), 
                                                    axis = -1, keepdims = True)

    def get_config(self):
        """Update configuration for layer."""
        config = super().get_config().copy()
        config.update({"units": self.units, "activation": self.activation_name})

        return config


class LSTMEncoder(Layer):
    """
        Class for a stacked LSTM layer architecture.

        Params:
        - latent_dim          : dimensionality of latent space for each sub-sequence. (default = 32)
        - hidden_layers       : Number of "hidden"/intermediate LSTM layers.  (default = 1)
        - hidden_nodes        : For hidden LSTM layers, the dimensionality of the intermediate state. (default = 32)
        - state_fn            : The activation function to use on cell state/output. (default = 'tanh')
        - recurrent_activation: The activation function to use on forget/input/output gates. (default = 'sigmoid')
        - return_sequences    : Indicates if returns sequence of states on the last layer (default = False)
        - dropout             : dropout rate to be used on cell state/output computation. (default = 0.6)
        - recurrent_dropout   : dropout rate to be used on forget/input/output gates. (default = 0.0)
        - regulariser_params :  tuple of floats indicating l1_l2 regularisation. (default = (0.01, 0.01))
        - name                : Name on which to save component. (default = 'encoder')
    """

    def __init__(self, latent_dim=32, hidden_layers=1, hidden_nodes=20, state_fn="tanh", recurrent_fn="sigmoid",
                 regulariser_params=(0.01, 0.01), return_sequences=False, dropout=0.6, recurrent_dropout=0.0,
                 name='LSTM_enc'):

        # Block Parameters
        super().__init__(name=name)
        self.latent_dim = latent_dim
        self.hidden_layers = hidden_layers
        self.hidden_nodes = hidden_nodes
        self.state_fn = state_fn
        self.recurrent_fn = recurrent_fn
        self.return_sequences = return_sequences
        self.dropout = dropout
        self.recurrent_dropout = recurrent_dropout

        # Regulariser info
        self.regulariser_params = regulariser_params
        l1_param, l2_param = regulariser_params
        self.regulariser = mix_l1_l2_reg(l1_param, l2_param)

        # Add Intermediate Layers
        for layer_id_ in range(self.hidden_layers):
            self.__setattr__('layer_' + str(layer_id_),
                             LSTM(units=self.hidden_nodes, return_sequences=True, activation=self.state_fn,
                                  recurrent_activation=self.recurrent_fn, dropout=self.dropout,
                                  recurrent_dropout=self.recurrent_dropout,
                                  kernel_regularizer=self.regulariser, recurrent_regularizer=self.regulariser,
                                  bias_regularizer=self.regulariser, return_state=False))

        # Input and Output Layers
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
        for layer_id_ in range(self.hidden_layers):
            layer_ = self.__getattribute__('layer_' + str(layer_id_))
            x_inter = layer_(x_inter, training=training)

        return self.output_layer(x_inter, training=training)

    def get_config(self):
        """Update configuration for layer."""
        config = super().get_config().copy()
        config.update({"latent_dim": self.latent_dim, "hidden_layers": self.hidden_layers,
                       "hidde_nodes": self.hidden_nodes, "state_fn": self.state_fn,
                       "recurrent_fn": self.recurrent_fn, "return_sequences": self.return_sequences,
                       "dropout": self.dropout, "recurrent_dropout": self.recurrent_dropout,
                       "regulariser_params": self.regulariser_params})

        return config


class AttentionRNNEncoder(LSTMEncoder):
    """
        Class for an Attention RNN Encoder architecture.

        Params:
    units: int, dimensionality of projection/latent space.
    activation: str/fn, the activation function to use. (default = "relu")
    """

    def __init__(self, units, activation="linear", **kwargs):
        super().__init__(latent_dim=units, return_sequences=True, **kwargs)
        self.feature_time_att_layer = FeatureTimeAttentionLayer(units=units, activation=activation)

    def call(self, inputs, mask=None, training=True):
        """Forward pass of layer block."""
        latent_reps = super().call(inputs, mask=mask, training=training)
        z = self.feature_time_att_layer.call(inputs=inputs, latent_reps=latent_reps)

        return z

    def compute_attention_map_scores(self, inputs, cluster_reps):
        """Compute alpha, beta, gamma scores for cluster estimation."""
        latent_reps = super().call(inputs, training=False)
        return self.feature_time_att_layer.compute_all_scores(inputs, latent_reps, cluster_reps)

    def estimate_alpha_beta(self, inputs):
        """Compute alpha, beta as in the forward call of forward attention layer"""

        # Compute latent representations and estimate alpha
        latent_reps = super().call(inputs, training=False)
        _, alpha_hat = self.feature_time_att_layer.compute_o_hat_and_alpha(inputs, latent_reps)

        # Estimate beta
        beta_hat = self.feature_time_att_layer.beta_weights

        return alpha_hat, beta_hat

    def get_config(self):
        """Update configuration for layer."""
        return super().get_config()
