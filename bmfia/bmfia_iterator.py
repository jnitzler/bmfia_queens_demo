"""Iterator for Bayesian multi-fidelity inverse analysis."""

import itertools
import logging
from pathlib import Path

import numpy as np
from queens.iterators._iterator import Iterator
from queens.utils.logger_settings import log_init_args

_logger = logging.getLogger(__name__)


class BmfiaIterator(Iterator):
    """Bayesian multi-fidelity inverse analysis iterator.

    Iterator for Bayesian multi-fidelity inverse analysis. Here, we build
    the multi-fidelity probabilistic surrogate, determine optimal training
    points *X_train* and evaluate the low- and high-fidelity model for these
    training inputs, to yield *Y_LF_train* and *Y_HF_train* training data. The
    actual inverse problem is not solved or iterated in this module but instead
    we iterate over the training data to approximate the probabilistic mapping
    *p(yhf|ylf)*.

    Attributes:
        X_train (np.array): Input training matrix for HF and LF model.
        Y_LF_train (np.array): Corresponding LF model response to *X_train* input.
        Y_HF_train (np.array): Corresponding HF model response to *X_train* input.
        informative_features_train (np.array): Corresponding LF informative features to *X_train* input.
        Z_train (np.array): Corresponding LF informative features to *X_train* input.
        hf_model (obj): High-fidelity model object.
        lf_model (obj): Low-fidelity model object.
        coords_experimental_data (np.array): Coordinates of the experimental data.
        bmfia_spatial_fields_transfer (obj): Object for spatial fields transfer from LF to HF grid.
        data_file_path (str): Path to a data file containing the initial training data

    Returns:
       BmfiaIterator (obj): Instance of the BmfiaIterator
    """
    @log_init_args
    def __init__(
        self,
        parameters,
        hf_model,
        lf_model,
        initial_design,
        num_obs,
        grid_interpolator=None,
        global_settings=None,
        path_trainings_data=None,
    ):
        """Instantiate the BmfiaIterator object.

        Args:
            parameters (obj): Parameters object
            hf_model (obj): High-fidelity model object.
            lf_model (obj): Low-fidelity model object.
            initial_design (dict): Dictionary describing initial design.
            bmfia_spatial_fields_transfer (obj): Object for spatial fields transfer from LF to HF grid.
            data_file_path (str): Path to a data file containing the initial training data
        """
        super().__init__(lf_model, parameters, global_settings)
        # ---------- calculate the initial training samples via class-methods ----------
        x_train_lf, x_train_hf = BmfiaIterator._joint_markov_prior_design(
            initial_design, parameters, grid_interpolator
        )

        self.X_train = x_train_lf
        self.X_train_hf = x_train_hf
        self.Y_LF_train = None
        self.Y_HF_train = None
        self.informative_features_train = None
        self.Z_train = None
        self.hf_model = hf_model
        self.lf_model = lf_model
        self.num_obs = num_obs
        self.grid_interpolator = grid_interpolator
        self.path_trainings_data = Path(path_trainings_data)

    @staticmethod
    def _joint_markov_prior_design(
        initial_design_dict, parameters, bmfia_spatial_fields_transfer
    ):
        rng = np.random.default_rng(seed=initial_design_dict["seed"])
        num_hyper_params_steps = initial_design_dict["num_hyper_params_steps"]
        mean_range = initial_design_dict["mean_range"]
        delta_range = initial_design_dict["delta_range"]
        delta_lst = np.linspace(delta_range[0], delta_range[1], num_hyper_params_steps)
        mean_lst = np.linspace(mean_range[0], mean_range[1], num_hyper_params_steps)
        num_samples_per_delta = int(
            initial_design_dict["num_HF_eval"] / num_hyper_params_steps
        )

        # Generate all possible combinations of delta and mean using itertools
        hyper_params_combinations = list(itertools.product(delta_lst, mean_lst))
        hyper_params_indices = rng.choice(
            len(hyper_params_combinations), num_hyper_params_steps, replace=False
        )
        hyper_params_list = [hyper_params_combinations[i] for i in hyper_params_indices]
        prior_name = parameters.names[0]
        prior = parameters.dict[prior_name]

        x_train = []
        for num, (delta, mean) in enumerate(hyper_params_list):
            prior.update_delta(delta)
            prior.mean = mean * np.ones(prior.dimension)
            my_seed = num + 999
            np.random.seed(my_seed)
            x_train.append(prior.draw(num_samples_per_delta))

        x_train_lf = np.array(x_train).reshape(-1, x_train[0].shape[1], order="C")

        max_x = np.max(np.exp(x_train_lf))
        min_x = np.min(np.exp(x_train_lf))
        _logger.info("Minimum exp(x) = %s, max exp(x) = %s", min_x, max_x)

        x_train_hf = bmfia_spatial_fields_transfer.transfer_lf_to_hf_grid(x_train_lf)
        return x_train_lf, x_train_hf

    # ----------- main methods of the object form here ----------------------------------------
    def core_run(self):
        """Trigger main or core run of the BMFIA iterator.

        It summarizes the actual evaluation of the HF and LF models for these data and the
        determination of LF informative features.

        Returns:
            Z_train (np.array): Matrix with low-fidelity feature training data
            Y_HF_train (np.array): Matrix with HF training data
        """
        # ----- build model on training points and evaluate it -----------------------
        self.eval_model()

        _logger.info(
            "Min / max LF train: %s / %s",
            np.min(self.Y_LF_train),
            np.max(self.Y_LF_train),
        )
        _logger.info(
            "Min / max HF train: %s / %s",
            np.min(self.Y_HF_train),
            np.max(self.Y_HF_train),
        )

        # ------ reshape the simulation data for the probabilistic regression model -----------
        num_samples = self.Y_LF_train.shape[0]
        self.Y_LF_train = self.Y_LF_train.reshape(
            num_samples, self.num_obs, -1, order="F"
        )
        self.Y_HF_train = self.Y_HF_train.reshape(
            num_samples, self.num_obs, -1, order="F"
        )
        self.informative_features_train = self.informative_features_train.reshape(
            num_samples, self.num_obs, -1, order="F"
        )
        self.Z_train = np.concatenate(
            (self.Y_LF_train, self.informative_features_train), axis=2
        )

        return self.Z_train, self.Y_HF_train

    def evaluate_lf_model_for_x_train(self):
        """Evaluate the low-fidelity model for the X_train input data-set."""
        lf_output = self.lf_model.evaluate(self.X_train)
        self.Y_LF_train = lf_output["result"]
        self.informative_features_train = lf_output.get("features")

    def evaluate_hf_model_for_x_train(self):
        """Evaluate the high-fidelity model for the X_train input data-set."""
        x_hf_train = self.X_train_hf
        # TODO: below is a quick hack that overwrites the parameter keys for the HF model
        self.parameters.parameters_keys = [str(i) for i in range(x_hf_train.shape[1])]
        self.Y_HF_train = self.hf_model.evaluate(x_hf_train)["result"]

    def eval_model(self):
        """Evaluate the LF and HF model to for the training inputs."""
        _logger.info(
            "Starting to evaluate the low-fidelity model for training points...."
        )

        self.evaluate_lf_model_for_x_train()

        _logger.info("Successfully calculated the low-fidelity training points!")

        # ---- run HF model on X_train
        _logger.info(
            "Starting to evaluate the high-fidelity model for training points..."
        )
        self.evaluate_hf_model_for_x_train()

        _logger.info("Successfully calculated the high-fidelity training points!")

    def post_run(self):
        """Post-run method of the iterator."""
        # save the training data
        np.savez(
            self.path_trainings_data, self.Y_LF_train, self.Y_HF_train, self.informative_features_train
        )
