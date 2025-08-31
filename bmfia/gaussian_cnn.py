"""Gaussian Neural Network regression model."""

import copy
import logging
import os

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from queens.models.surrogates._surrogate import Surrogate
from queens.utils.configure_tensorflow import configure_tensorflow
from queens.utils.logger_settings import log_init_args
from queens.utils.random_process_scaler import StandardScaler
from queens.visualization.gaussian_neural_network_vis import plot_loss

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tfd = tfp.distributions
Dense = tf.keras.layers.Dense
from sklearn.model_selection import train_test_split
from tensorflow.data.experimental import AUTOTUNE
from tensorflow.keras.layers import Dropout

_logger = logging.getLogger(__name__)

import matplotlib as mpl

mpl.use("TkAgg")

configure_tensorflow(tf)

# Use GPU acceleration if possible
if tf.test.gpu_device_name() != "/device:GPU:0":
    _logger.debug("WARNING: GPU device not found.")
else:
    _logger.debug("SUCCESS: Found GPU: %s", tf.test.gpu_device_name())


# Get the list of physical GPU devices --> avoid memory issues
gpus = tf.config.experimental.list_physical_devices("GPU")

if gpus:
    try:
        # Set memory growth on each GPU device
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


class GaussianCNN(Surrogate):
    """Class for creating a neural network that parameterizes a Gaussian.

    The network can handle heteroskedastic noise and an arbitrary nonlinear functions.

    Attributes:
        nn_model (tf.model):  Tensorflow based Bayesian neural network model
        num_epochs (int): Number of training epochs for variational optimization
        optimizer_seed (int): Random seed used for initialization of stochastic gradient decent
                              optimizer
        verbosity_on (bool): Boolean for model verbosity during training. True=verbose
        batch_size (int): Size of data-batch (smaller than the training data size)
        scaler_x (obj): Scaler for inputs
        scaler_y (obj): Scaler for outputs
        loss_plot_path (str): Path to determine whether loss plot should be produced
                              (yes if provided). Plot will be saved at path location.
        mean_function (function): Mean function of the Gaussian Neural Network
        gradient_mean_function (function): Gradient of the mean function of the Gaussian
                                           Neural Network
        training_rate (float): Training rate for the ADAMS gradient decent optimizer
        nodes_per_hidden_layer (lst): List containing number of nodes per hidden layer of
                                      the Neural Network. The length of the list
                                      defines the deepness of the model and the values the
                                      width of the individual layers.
        activation_per_hidden_layer (list): List with strings encoding the activation
                                            function that shall be used for the
                                            respective hidden layer of the  Neural
                                            Network
        nugget_std (float): Nugget standard deviation for robustness
    """

    @log_init_args
    def __init__(
        self,
        num_epochs=None,
        batch_size=None,
        training_rate=None,
        optimizer_seed=None,
        verbosity_on=None,
        nugget_std=None,
        loss_plot_path_dir=False,
        cnn_grid_input=None,
        latent_dim=None,
        num_validation_data=0,
        feature_list=[16, 32, 64],
    ):
        """Initialize an instance of the Gaussian Bayesian Neural Network.

        Args:
            num_epochs (int): Number of epochs used for variational training of the BNN
            batch_size (int): Size of data-batch (smaller than the training data size)
            adams_training_rate (float): Training rate for the ADAMS gradient decent optimizer
            optimizer_seed (int): Random seed for stochastic optimization routine
            verbosity_on (bool): Boolean for model verbosity during training. True=verbose
            nodes_per_hidden_layer_lst (lst): List containing number of nodes per hidden layer of
                                          the Neural Network. The length of the list
                                          defines the deepness of the model and the values the
                                          width of the individual layers.
            activation_per_hidden_layer_lst (list): List with strings encoding the activation
                                                function that shall be used for the
                                                respective hidden layer of the  Neural
                                                Network
            nugget_std (float): Nugget standard deviation for robustness
            loss_plot_path (str): Path to determine whether loss plot should be produced
                                  (yes if provided). Plot will be saved at path location.
            data_scaling (str): Data scaling type
            mean_function_type (str): Mean function type of the Gaussian Neural Network

        Returns:
            Instance of GaussianBayesianNeuralNetwork
        """
        super().__init__()

        self.nn_model = None
        self.num_epochs = num_epochs
        self.optimizer_seed = optimizer_seed
        self.verbosity_on = verbosity_on
        self.batch_size = batch_size
        self.scaler_x = StandardScaler()
        self.scaler_y = StandardScaler()
        self.loss_plot_path_dir = loss_plot_path_dir
        self.mean_function = GaussianCNN.identity_multi_fidelity_mean_function

        self.training_rate = training_rate
        self.nugget_std = nugget_std
        self.cnn_grid_input = cnn_grid_input
        self.input_grid = None
        self.latent_dim = latent_dim
        self.num_validation_data = num_validation_data
        self.feature_list = feature_list
        self.train_dataset = None
        self.test_dataset = None
        self.x_val = None
        self.y_val = None

    @staticmethod
    def identity_multi_fidelity_mean_function(x):
        """Identity multi-fidelity mean function.

        Args:
            x (np.ndarray): input array
            dim_y (np.ndarray): dimension of the output array

        Returns:
            y (np.ndarray): output array
        """
        y = x
        return y

    # -------------- custom function ----------------------------------
    def _build_model(self):
        """Build/compile the neural network.

        We use a regular densely connected
        NN, which is parameterizing mean and variance of a Gaussian
        distribution. The network can be arbitrary deep and wide and can use
        different (nonlinear) activation functions.

        Returns:
            model (obj): Tensorflow probability model instance
        """
        height, width, input_channels = self.cnn_grid_input
        kernel_size = 3
        pooling_size = 2
        padding_strategy = "same"
        activation = tf.keras.layers.ELU()

        dropout = 0.25  # 0.3

        #  ----- Encoder ------------#
        # first layer
        encoder_inputs = tf.keras.Input(
            shape=(height, width, input_channels + 1), dtype=tf.float32
        )

        x = tf.keras.layers.Conv2D(
            self.feature_list[0],
            kernel_size,
            padding="valid",
            dtype=tf.float32,
        )(encoder_inputs)
        x = tf.keras.layers.BatchNormalization()(x)
        x = activation(x)

        # Max Pooling: 50 --> 25
        x = tf.keras.layers.MaxPooling2D(
            kernel_size,
            strides=pooling_size,
            padding=padding_strategy,
            dtype=tf.float32,
        )(
            x
        )  # Reduces each dimension by half

        # second layer
        x = tf.keras.layers.Conv2D(
            self.feature_list[1],
            kernel_size,
            padding="valid",
            dtype=tf.float32,
        )(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = activation(x)

        # Max Pooling: 25 --> 12
        x = tf.keras.layers.MaxPooling2D(
            kernel_size,
            strides=pooling_size,
            padding=padding_strategy,
            dtype=tf.float32,
        )(x)

        # third layer
        x = tf.keras.layers.Conv2D(
            self.feature_list[2],
            kernel_size,
            padding="valid",
            dtype=tf.float32,
        )(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = activation(x)

        # Max Pooling: 12 --> 6
        x = tf.keras.layers.MaxPooling2D(
            kernel_size,
            strides=pooling_size,
            padding=padding_strategy,
            dtype=tf.float32,
        )(x)

        # Reduces each dimension by half
        x = tf.keras.layers.Flatten()(x)

        # dense layer
        x = tf.keras.layers.Dense(self.latent_dim, dtype=tf.float32)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        encoder_outputs = activation(x)
        encoder_outputs = Dropout(dropout)(encoder_outputs)

        # Maintain the number of features from the last pooling layer
        encoder = tf.keras.Model(
            inputs=encoder_inputs, outputs=encoder_outputs, name="encoder"
        )

        # ------ Decoder ------------#
        decoder_inputs = tf.keras.Input(shape=(self.latent_dim,))

        # note we increase to 6x6 cube here instead of 7x7
        x = tf.keras.layers.Dense(
            6 * 6 * self.feature_list[-1],
            dtype=tf.float32,
        )(decoder_inputs)
        x = activation(x)
        x = Dropout(dropout)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Reshape((6, 6, self.feature_list[-1]))(x)

        # 6x6
        # first upsampling layer 6 --> 12
        x = tf.keras.layers.Conv2DTranspose(
            self.feature_list[-1],
            kernel_size,
            strides=pooling_size,
            padding=padding_strategy,
            dtype=tf.float32,
        )(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = activation(x)
        # 12 x 12

        # second upsampling layer: 12 --> 25 with valid padding
        x = tf.keras.layers.Conv2DTranspose(
            self.feature_list[-2],
            kernel_size,
            strides=pooling_size,
            padding="valid",
            dtype=tf.float32,
        )(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = activation(x)

        # 25 x 25

        # third upsampling layer: 25 --> 50
        x = tf.keras.layers.Conv2DTranspose(
            self.feature_list[-3],
            kernel_size,
            strides=pooling_size,
            padding=padding_strategy,
            dtype=tf.float32,
        )(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = activation(x)

        # last probabilistic layer
        num_params = 2 * input_channels  # num_cov + input_channels
        x = tf.keras.layers.Conv2D(
            num_params, 3, activation="linear", padding="same", dtype=tf.float32
        )(x)

        # cropping layer to 50 x 50
        loc = tf.keras.layers.Lambda(lambda t: t[..., :input_channels])(x)
        scale = tf.keras.layers.Lambda(lambda t: t[..., input_channels:])(x)
        scale = tf.keras.layers.Activation(tf.keras.activations.softplus)(scale)
        scale = tf.keras.layers.Lambda(lambda t: t + tf.cast(self.nugget_std, t.dtype))(
            scale
        )

        decoder_outputs = tf.keras.layers.Concatenate(name="gaussian_params")(
            [loc, scale]
        )

        decoder = tf.keras.Model(
            inputs=decoder_inputs, outputs=decoder_outputs, name="decoder"
        )
        # ------- Convolutional Autoencoder -------------------
        encoder_outputs = encoder(encoder_inputs)
        decoder_outputs = decoder(encoder_outputs)

        # compile the Tensorflow model
        autoencoder = tf.keras.Model(inputs=encoder_inputs, outputs=decoder_outputs)
        optimizer = tf.optimizers.Adam(learning_rate=self.training_rate, clipnorm=1.0e3)
        # optimizer = tf.keras.optimizers.SGD(
        #     learning_rate=self.training_rate, clipnorm=1.0e3
        # )

        autoencoder.compile(
            optimizer=optimizer,
            loss=self.negative_log_likelihood,
        )

        # print some infos
        encoder.summary()
        decoder.summary()
        autoencoder.summary()

        return autoencoder

    @staticmethod
    def negative_log_likelihood(y_hf, rv_y):
        """Negative log. likelihood of (tensorflow) random variable rv_y.

        Args:
            y (float): Value/Realization of the random variable / y_train
            rv_y (obj): Tensorflow probability random variable object/prediction

        Returns:
            negative_log_likelihood (float): Negative logarithmic likelihood of rv_y at y
        """
        C = tf.shape(rv_y)[-1] // 2
        loc = rv_y[..., :C]
        scale = rv_y[..., C:]
        dist = tfd.MultivariateNormalDiag(loc=loc, scale_diag=scale)
        return -tf.reduce_mean(dist.log_prob(y_hf))

    def setup(self, x_train, y_train):
        """Setup surrogate model.

        Args:
            x_train (np.array): training inputs
            y_train (np.array): training outputs
        """
        # setup input shape: n x num_coords x 3
        # cnn input shape:   n x height x width x depth x 3
        # scaling input:     (n * num_coords) x 3

        # reshape the arrays in C order:
        num_channels = self.cnn_grid_input[-1]
        self.input_grid = copy.copy(self.cnn_grid_input)
        self.input_grid[-1] = self.input_grid[-1] + 1

        x_train = x_train.reshape(-1, *self.input_grid, order="F")
        y_train = y_train.reshape(-1, *self.cnn_grid_input, order="F")

        # subtract mean function
        y_train -= self.mean_function(x_train)[..., :num_channels]

        # fit/transform and reshape for cnn input
        # note: scaling requires reshaping with C order to keep last dim the same, afterwards just
        # shape back
        self.scaler_x.fit(x_train.reshape(-1, num_channels + 1))
        x_train = (
            self.scaler_x.transform(x_train.reshape(-1, num_channels + 1)).reshape(
                -1,
                *self.input_grid,
            )
        ).astype("float32")

        # fit/transform and reshape for cnn input
        self.scaler_y.fit(y_train.reshape(-1, num_channels))
        y_train = (
            self.scaler_y.transform(y_train.reshape(-1, num_channels)).reshape(
                -1, *self.cnn_grid_input
            )
        ).astype("float32")

        # Split the data
        self.x_train, self.x_val, self.y_train, self.y_val = train_test_split(
            x_train, y_train, test_size=self.num_validation_data, random_state=42
        )
        self.train_dataset = tf.data.Dataset.from_tensor_slices(
            (self.x_train, self.y_train)
        )
        self.test_dataset = tf.data.Dataset.from_tensor_slices((self.x_val, self.y_val))

        self.nn_model = self._build_model()

    def train(self):
        """Train the Bayesian neural network.

        We ues the previous defined optimizers in the model build and
        configuration. We allow tensorflow's early stopping here to stop
        the optimization routine when the loss function starts to
        increase again over several iterations.
        """
        # set the random seeds for optimization/training
        tf.keras.utils.set_random_seed(self.optimizer_seed)

        self.train_dataset = self.train_dataset.shuffle(buffer_size=10240)
        self.train_dataset = self.train_dataset.batch(self.batch_size)
        self.train_dataset = self.train_dataset.prefetch(buffer_size=AUTOTUNE)

        self.test_dataset = self.test_dataset.shuffle(buffer_size=10240)
        self.test_dataset = self.test_dataset.batch(self.batch_size)
        self.test_dataset = self.test_dataset.prefetch(buffer_size=AUTOTUNE)

        history = self.nn_model.fit(
            self.train_dataset,
            epochs=self.num_epochs,
            verbose=self.verbosity_on,
            shuffle=True,
            validation_data=self.test_dataset,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss", patience=5000, restore_best_weights=True
                )
            ],
        )

        # print out the model summary
        self.nn_model.summary()

        if self.loss_plot_path_dir:
            plot_loss(history, self.loss_plot_path_dir)

    def compute_validation_mean_l2_error(self):
        """calculate the l2 error between validation mean prediction and validation data."""
        yhat_val = self.predict_y(self.x_val)["result"].reshape(
            self.y_val.shape, order="F"
        )
        absolute_l2_error = np.sqrt(
            np.sum(np.linalg.norm((yhat_val - self.y_val), axis=3) ** 2, axis=(1, 2))
        )
        true_norm = np.sqrt(
            np.sum(np.linalg.norm(self.y_val, axis=3) ** 2, axis=(1, 2))
        )
        mean_relative_l2_error = np.mean(absolute_l2_error / true_norm)
        std_relative_l2_error = np.std(absolute_l2_error / true_norm)
        _logger.info(f"Mean relative error: {mean_relative_l2_error}")
        _logger.info(f"STD relative error: {std_relative_l2_error}")

    def grad(self, samples, upstream_gradient):
        """Gradient method for the surrogate model."""
        raise NotImplementedError(
            "Gradient method is not implemented here. \n"
            "Instead we directly use the method 'predict_and_gradient'\n"
            "which directly computes gradients w.r.t. the likelihood\n"
            "to avoid large partial derivative tensors."
        )

    def predict(
        self, x_test, support="y", gradient_bool=False, noise_var=None, y_obs=None
    ):
        """Predict the output distribution at x_test.

        Args:
            x_test (np.array): Testing input vector for which the posterior distribution,
                               respectively point estimates should be predicted
            support (str, optional): String to define the support of the output distribution
                                    - 'y': Conditional distribution is defined on the output space
                                    - 'f': Conditional distribution is defined on the latent space
            gradient_bool (bool, optional): Boolean to configure whether gradients should be
                                            returned as well

        Returns:
            output (dict): Dictionary with posterior output statistics
        """
        x_test = x_test.reshape(-1, *self.input_grid, order="F")
        if support == "f":
            raise NotImplementedError('Support "f" is not implemented yet.')

        if gradient_bool:
            output = self.predict_and_gradient(x_test, noise_var=noise_var, y_obs=y_obs)
        else:
            output = self.predict_y(x_test)

        return output

    def predict_y(self, x_test):
        """Predict the posterior mean and variance.

        Prediction is conducted w.r.t. to the output space "y".

        Args:
            x_test (np.array): Testing input vector for which the posterior distribution,
                               respectively point estimates should be predicted

        Returns:
            output (dict): Dictionary with posterior output statistics
        """
        # cnn input shape: n x num_coords x 3
        # predict input shape: n x x width x depth x height x 3
        # mean_function input: as expected y_test
        # scaling input: (n * num_coords) x 3
        num_channels = self.cnn_grid_input[-1]
        if len(self.cnn_grid_input) == 3:
            num_coords = self.cnn_grid_input[0] * self.cnn_grid_input[1]
        else:
            num_coords = (
                self.cnn_grid_input[0] * self.cnn_grid_input[1] * self.cnn_grid_input[2]
            )

        # C - order for scaling and reshaping after scaling also in C
        x_test_transformed = (
            self.scaler_x.transform(x_test.reshape(-1, num_channels + 1))
            .reshape(-1, *self.input_grid)
            .astype("float32")
        )
        output = self.nn_model.predict(x_test_transformed)
        mean_pred = output[..., :num_channels]
        var_pred = np.square(output[..., num_channels:])

        # --- rescale and reshape mean for output ---
        mean_rescaled = (
            self.scaler_y.inverse_transform_mean(
                mean_pred.reshape(-1, num_channels)
            ).reshape(-1, *self.cnn_grid_input)
            + self.mean_function(x_test)[..., :num_channels]
        )

        # now shape back in F order
        mean_rescaled = mean_rescaled.reshape(-1, num_coords * num_channels, order="F")

        # --- rescale and reshape var for output ---
        var_rescaled = (
            self.scaler_y.inverse_transform_std(
                np.sqrt(var_pred.reshape(-1, num_channels))
            )
            ** 2
        ).reshape(-1, *self.cnn_grid_input)
        # now shape back in F order
        var_rescaled = var_rescaled.reshape(-1, num_coords * num_channels, order="F")

        output = {"result": mean_rescaled}
        output["variance"] = var_rescaled

        return output

    def predict_and_gradient(self, x_test, noise_var=None, y_obs=None):
        """Predict the mean, variance and their gradients at x_test.

        Args:
            x_test (np.array): Testing input vector for which the posterior
                               distribution, respectively point estimates should be
                               predicted

        Returns:
            output (dict): Dictionary with posterior output statistics
        """
        # Note: only called per sample not in batch

        # cnn input shape: n x num_coords x 3
        # setup input shape: n x (num_coords * 3)
        # mean_function input: as expected y_test
        # scaling input: (n * num_coords) x 3
        num_channels = self.cnn_grid_input[-1]
        if len(self.cnn_grid_input) == 3:
            num_coords = self.cnn_grid_input[0] * self.cnn_grid_input[1]
        else:
            num_coords = (
                self.cnn_grid_input[0] * self.cnn_grid_input[1] * self.cnn_grid_input[2]
            )

        # reshape first in C order
        y_obs = y_obs.reshape(*self.cnn_grid_input, order="F").astype("float32")
        y_obs = y_obs[np.newaxis, ...]

        x_test_tensorflow = tf.Variable(x_test)
        with tf.GradientTape(persistent=False) as tape:
            tape.watch(x_test_tensorflow)

            # transform and reshape x_test_tensorflow
            # reshaping is here done in C order and only for scaling and then reverted
            x_test_transformed = self.scaler_x.transform(
                tf.reshape(x_test_tensorflow, [-1, num_channels + 1])
            )
            x_test_transformed = tf.reshape(x_test_transformed, [-1, *self.input_grid])

            output = self.nn_model.predict(x_test_transformed)
            mean_pred = output[..., :num_channels]
            var_pred = np.square(output[..., num_channels:])

            # we calculate the gradients for the log lik here directly
            # by defining scalar dependencies in log likelihood and then
            # calculating their gradients wrt x
            mean_rescaled = (
                tf.reshape(
                    self.scaler_y.inverse_transform_mean(
                        tf.reshape(mean_pred, [-1, num_channels])
                    ),
                    [-1, *self.cnn_grid_input],
                )
                + self.mean_function(tf.cast(x_test_tensorflow, tf.float32))[
                    ..., :num_channels
                ]
            )

            var_rescaled = tf.math.square(
                self.scaler_y.inverse_transform_std(
                    tf.math.sqrt(tf.reshape(var_pred, [-1, num_channels]))
                )
            )
            var_rescaled = tf.reshape(var_rescaled, [-1, *self.cnn_grid_input])

            loglik = -0.5 * tf.reduce_sum(
                tf.math.divide(
                    tf.math.square(mean_rescaled - y_obs),
                    (var_rescaled + noise_var),
                )
            ) - 0.5 * tf.reduce_sum(tf.math.log(var_rescaled + noise_var))

        d_log_lik_d_z = tape.gradient(loglik, x_test_tensorflow)
        # clip the gradient by norm
        d_log_lik_d_z = tf.clip_by_norm(d_log_lik_d_z, 1000.0)

        # reshape in F order
        mean_pred = mean_rescaled.numpy().reshape(
            -1, num_coords * num_channels, order="F"
        )
        var_pred = var_rescaled.numpy().reshape(
            -1, num_coords * num_channels, order="F"
        )
        d_log_lik_d_z = d_log_lik_d_z.numpy().reshape(-1, *self.input_grid, order="F")

        output = {"result": mean_pred}
        output["variance"] = var_pred
        output["d_log_lik_d_y"] = d_log_lik_d_z

        return output
