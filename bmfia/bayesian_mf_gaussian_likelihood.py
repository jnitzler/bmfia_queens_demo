"""Multi-fidelity Gaussian likelihood model."""

import copy
import logging
import time

import numpy as np
import queens.visualization.bmfia_visualization as qvis
from queens.distributions.mean_field_normal import MeanFieldNormal
from queens.iterators.reparameteriztion_based_variational import RPVI
from queens.models.likelihoods._likelihood import Likelihood
from queens.parameters.parameters import Parameters
from queens.stochastic_optimizers.adam import Adam

from bmfia.mean_field_generalized_gamma import MeanFieldGeneralizedGammaDistribution

_logger = logging.getLogger(__name__)

VALID_NOISE_TYPES = {"em_gamma": "dummy", "fixed_variance": "dummy"}


class BMFGaussianModel(Likelihood):
    """Multi fidelity likelihood function.

    Multi-fidelity likelihood of the Bayesian multi-fidelity inverse
    analysis scheme [1, 2].

    Attributes:
        coords_mat (np.array): Row-wise coordinates at which the observations were recorded
        output_label (str): Name of the experimental outputs (column label in csv-file)
        coord_labels (lst): List with coordinate labels for (column labels in csv-file)
        normal_distribution (obj): Mean field normal distribution object
        noise_var (np.array): Noise variance of the observations
        num_refinement_samples (int): Number of additional samples to train the multi-fidelity
                                      dependency in refinement step
        likelihood_evals_for_refinement (lst):  List with necessary number of likelihood
                                                evaluations before the refinement step is
                                                conducted
        mf_approx (Model): Probabilistic mapping

    Returns:
        Instance of BMFGaussianModel. This is a multi-fidelity version of the
        Gaussian noise likelihood model.


    References:
        [1] Nitzler, J.; Biehler, J.; Fehn, N.; Koutsourelakis, P.-S. and Wall, W.A. (2020),
            "A Generalized Probabilistic Learning Approach for Multi-Fidelity Uncertainty
            Propagation in Complex Physical Simulations", arXiv:2001.02892

        [2] Nitzler J.; Wall, W. A. and Koutsourelakis P.-S. (2021),
            "An efficient Bayesian multi-fidelity framework for the solution of high-dimensional
            inverse problems in computationally demanding simulations", unpublished internal report
    """

    def __init__(
        self,
        forward_model,
        path_trainings_data,
        experimental_coordinates,
        coordinate_labels,
        output_labels,
        y_obs_vec,
        mf_approx,
        num_refinement_samples=None,
        likelihood_evals_for_refinement=None,
        noise_var_iterative_averaging=None,
        noise_variance=None,
        noise_type="None",
        plotting_options=None,
    ):
        """Instantiate the multi-fidelity likelihood class.

        Args:
            forward_model (obj): Forward model to iterate; here: the low fidelity model
            mf_subiterator (obj): Subiterator to select the training data of the probabilistic
                                  regression model
            mf_approx (Model): Probabilistic mapping
            num_refinement_samples (int): Number of additional samples to train the multi-fidelity
                                          dependency in refinement step
            likelihood_evals_for_refinement (lst): List with necessary number of likelihood
                                                   evaluations before the refinement step is
                                                   conducted
            plotting_options (dict): Options for plotting
        """
        self.y_obs = y_obs_vec
        self.coords_mat = experimental_coordinates
        self.coord_labels = coordinate_labels
        self.output_label = output_labels

        super().__init__(forward_model, y_obs_vec)

        # ----------------------- initialize the mean field normal distribution ------------------
        dimension = y_obs_vec.size
        gaussian_distr = MeanFieldNormal(
            mean=y_obs_vec, variance=noise_variance, dimension=dimension
        )

        # ---------------------- initialize some model settings/train surrogates -----------------
        self.build_approximation(
            path_trainings_data,
            mf_approx,
            self.coord_labels,
            self.coords_mat,
        )
        # ----------------------- create visualization object(s) ---------------------------------
        if plotting_options:
            qvis.from_config_create(plotting_options)

        self.mf_approx = mf_approx
        self.noise_var_iterative_averaging = noise_var_iterative_averaging
        self.min_log_lik_mf = None
        self.normal_distribution = gaussian_distr
        self.noise_var = noise_variance
        self.num_refinement_samples = num_refinement_samples
        self.likelihood_evals_for_refinement = likelihood_evals_for_refinement
        self.variational_distribution = (
            None  # TODO: this is a hack atm and not very nice
        )
        self.variational_params = None  # TODO: this is a hack atm and not very nice
        self.parameters_mean = None  # TODO: this is a hack atm and not very nice
        self.noise_type = noise_type
        self.spatial_dim = len(self.coord_labels)
        self.initial_tau_param = None

        # ------- now define stuff for the sub inverse problem wrt tau
        self.posterior_samples_tau = None
        self.temp_posterior_samples_tau = None

        tau = MeanFieldGeneralizedGammaDistribution(
            alpha=1.0e-9, beta=1.0e-9, dimension=1
        )
        tau_param = Parameters(tau=tau)

        result_description = {
            "iterative_field_names": ["variational_parameters", "elbo"],
            "write_results": False,
            "plotting_options": {
                "plot_boolean": False,
                "plotting_dir": "",
                "plot_name": "",
                "save_bool": False,
            },
        }
        variational_distribution = {
            "variational_approximation_type": "mean_field",
            "variational_family": "normal",
            "nugget_var_diag": 1.0e-9,
            "exp_transform": False,
        }

        n_samples_per_iter = 4
        variational_transformation = None
        random_seed = 1
        max_feval = 2500
        max_iteration = 2500
        learning_rate = 0.05
        rel_l1_change_threshold = 1e-4
        rel_l2_change_threshold = 1e-4
        stochastic_optimizer = Adam(
            learning_rate,
            "max",
            max_iteration=max_iteration,
            rel_l1_change_threshold=rel_l1_change_threshold,
            rel_l2_change_threshold=rel_l2_change_threshold,
        )

        # ---------------------------------------------
        # note we will do a exp transform for actual tau
        variational_parameter_initialization = {
            "dimension": 1,
            "mean": 0.0,
            "variance": 1.0,
        }
        natural_gradient = False
        FIM_dampening = False
        decay_start_iteration = 50
        dampening_coefficient = 1e-2
        FIM_dampening_lower_bound = 1e-8
        score_function_bool = False

        self.my_tau_model = PrecisionLikDummy(
            self.normal_distribution, self.spatial_dim
        )

        self.tau_svi = RPVI(
            copy.copy(self.my_tau_model),
            tau_param,
            result_description,
            variational_distribution,
            n_samples_per_iter,
            variational_transformation,
            random_seed,
            max_feval,
            stochastic_optimizer,
            variational_parameter_initialization,
            natural_gradient,
            FIM_dampening,
            decay_start_iteration,
            dampening_coefficient,
            FIM_dampening_lower_bound,
            score_function_bool,
            gnuplot_file_path=plotting_options.get("svi_noise_param_plot_path"),
        )
        self.tau_svi.plot_iterations = plotting_options.get("svi_plot_iter", 100)
        self.tau_svi_lst = []

    def _evaluate(self, samples):
        """Evaluate multi-fidelity likelihood.

        Evaluation with current set of variables
        which are an attribute of the underlying low-fidelity simulation model.

        Args:
            samples (np.ndarray): Evaluated samples

        Returns:
            mf_log_likelihood (np.array): Vector of log-likelihood values per model input.
        """
        # reshape the model output according to the number of coordinates
        num_samples = samples.shape[0]

        # we explicitly cut the array at the variable size as within one batch several chains
        # e.g., in MCMC might be calculated; we only want the last chain here
        output = self.forward_model.evaluate(samples)
        forward_model_output = np.rot90(
            output["result"][:num_samples].reshape(num_samples, 50, 50, 2, order="F"),
            k=1,
            axes=[1, 2],
        ).reshape(num_samples, -1, order="F")
        forward_model_features = np.rot90(
            output.get("meta_data")[:num_samples].reshape(
                num_samples, 50, 50, 1, order="F"
            ),
            k=1,
            axes=[1, 2],
        ).reshape(num_samples, -1, 1, order="F")

        # -------------------------------------
        mf_log_likelihood = self.evaluate_from_output(
            samples, forward_model_output, forward_model_features
        )
        self.response = {
            "forward_model_output": forward_model_output,
            "mf_log_likelihood": mf_log_likelihood,
            "forward_model_features": forward_model_features,
        }

        return {"result": mf_log_likelihood}

    def grad(self, samples, upstream_gradient):
        r"""Evaluate gradient of model w.r.t. current set of input samples.

        Consider current model f(x) with input samples x, and upstream function g(f). The provided
        upstream gradient is :math:`\frac{\partial g}{\partial f}` and the method returns
        :math:`\frac{\partial g}{\partial f} \frac{df}{dx}`.

        Args:
            samples (np.array): Input samples
            upstream_gradient (np.array): Upstream gradient function evaluated at input samples
                                          :math:`\frac{\partial g}{\partial f}`

        Returns:
            gradient (np.array): Gradient w.r.t. current set of input samples
                                 :math:`\frac{\partial g}{\partial f} \frac{df}{dx}`
        """
        partial_grad = self.partial_grad_evaluate(
            samples,
            self.response["forward_model_output"],
            self.response["forward_model_features"],
        )
        gradient = self.forward_model.grad(samples, partial_grad)
        return gradient

    def evaluate_from_output(
        self, samples, forward_model_output, forward_model_features
    ):
        """Evaluate multi-fidelity likelihood from forward model output.

        Args:
            samples (np.ndarray): Samples to evaluate
            forward_model_output (np.ndarray): Forward model output
            forward_model_features (np.ndarray): Forward model features

        Returns:
            mf_log_likelihood (np.array): Vector of log-likelihood values per model input.
        """
        # evaluate the modified multi-fidelity likelihood expression with LF model response
        mf_log_likelihood = self.em_evaluate_mf_likelihood(
            samples, forward_model_output, forward_model_features
        )
        return mf_log_likelihood

    def partial_grad_evaluate(
        self, forward_model_input, forward_model_output, forward_model_features
    ):
        """Implement the partial derivative of the evaluate method.

        The partial derivative w.r.t. the output of the sub-model is for example
        required to calculate gradients of the current model w.r.t. the sample
        input.

        Args:
            forward_model_input (np.array): Sample inputs of the model run (here not required).
            forward_model_output (np.array): Output of the underlying sub- or forward model
                                             for the current batch of sample inputs.
            forward_model_features (np.array): Features of the forward model

        Returns:
            grad_out (np.array): Evaluated partial derivative of the evaluation function
                                 w.r.t. the output of the underlying sub-model.
        """
        # ---- reshape the LF model output ---------
        num_samples = forward_model_output.shape[0]
        num_coords = self.coords_mat.shape[0]
        z_mat = forward_model_output.reshape(num_samples, num_coords, -1, order="F")

        # incorporate the informative features
        if forward_model_features is not None:
            z_mat = np.concatenate((z_mat, forward_model_features), axis=2)

        # Get the response matrices of the multi-fidelity mapping

        def em_grad_log_likelihood(tau_vec):
            """Helper function to calculate the log unnormalized posterior."""
            grad_log_lik_lst = []
            y_obs = self.normal_distribution.mean
            # loop over the samples of lf svi (different lf models)
            for z_vec, tau in zip(z_mat, tau_vec, strict=True):
                # we calculate the entire gradient within tensorflow
                breakpoint()
                d_log_lik_d_y = self.mf_interface.evaluate_and_gradient(
                    z_vec[np.newaxis, ...], noise_var=(1 / tau), y_obs=y_obs
                )
                grad_log_lik_lst.append(d_log_lik_d_y)

            grad_log_likelihood = np.array(grad_log_lik_lst).squeeze()
            return grad_log_likelihood

        # loop over samples of tau_vectors
        test_grad_log_likelihood = np.array(
            [
                em_grad_log_likelihood(tau_vec)
                for tau_vec in self.posterior_samples_tau.T
            ]
        )
        grad_out = np.mean(test_grad_log_likelihood, axis=0)

        return grad_out

    @staticmethod
    def build_approximation(
        path_trainings_data,
        approx,
        coord_labels,
        coords_mat,
    ):
        """Construct the probabilistic surrogate / mapping.

        Surrogate is calculated based on the provided training-data and
        optimize the hyper-parameters by maximizing the data's evidence
        or its lower bound (ELBO).

        Args:
            bmfia_subiterator (bmfia_subiterator): BMFIA subiterator object.
            bmfia_interface (bmfia_interface): BMFIA interface object.
            approx (Model): Approximation for probabilistic mapping.
            coord_labels (list): List of coordinate labels.
            coords_mat (np.array): (Spatial) Coordinates of the experimental data.
        """
        with np.load(path_trainings_data, allow_pickle=True) as training_data:
            array_names = training_data.files
            training_data_lst = []
            for name in array_names:
                training_data_lst.append(training_data[name])

        num_coords = coords_mat.shape[0]

        Y_LF_train = training_data_lst[0]
        num_samples = Y_LF_train.shape[0]
        Y_LF_train = Y_LF_train.reshape(num_samples, num_coords, -1, order="F")

        informative_features_train = training_data_lst[2]
        informative_features_train = informative_features_train.reshape(
            num_samples, num_coords, -1, order="F"
        )

        z_train = np.concatenate((Y_LF_train, informative_features_train), axis=2)

        Y_HF_train = training_data_lst[1].reshape(-1, coords_mat.size, order="F")
        y_hf_train = Y_HF_train.reshape(num_samples, num_coords, -1, order="F")

        # ----- train regression model on the data ----------------------------------------
        # reshape properly before it goes into cnn
        num_samples = y_hf_train.shape[0]
        dim = y_hf_train.shape[2]
        z_train = np.rot90(
            z_train.reshape(num_samples, 50, 50, dim + 1, order="F"), k=1, axes=[1, 2]
        ).reshape(num_samples, -1, dim + 1, order="F")
        y_hf_train = np.rot90(
            y_hf_train.reshape(num_samples, 50, 50, dim, order="F"), k=1, axes=[1, 2]
        ).reshape(num_samples, -1, dim, order="F")

        approx.grid_coordinates = coords_mat
        approx.setup(z_train, y_hf_train)

        # conduct training of probabilistic mapping
        t_s = time.time()
        _logger.info("Starting training of probabilistic mapping")
        approx.train()
        _logger.info("Finished training of probabilistic mapping")

        t_e = time.time()
        t_total = t_e - t_s
        _logger.info("Total time for training of probabilistic mapping: %d s", t_total)
        _logger.info(
            "---------------------------------------------------------------------"
        )
        _logger.info("Probabilistic model was built successfully!")
        _logger.info(
            "---------------------------------------------------------------------"
        )

    def em_evaluate_mf_likelihood(
        self, samples, forward_model_output, forward_model_features
    ):
        """Update the noise variance using the EM algorithm.

        Args:
            samples (np.array): Samples to evaluate
            forward_model_output (np.array): Forward model output
            forward_model_features (np.array): Forward model features
        """
        # the new estimate for the noise variance is the inverse of the
        # expectation of the precisions parameter's (inverse of the noise variance)
        # posterior given the current model output (here forward_model_output)
        # TODO: continue here!!
        num_samples = forward_model_output.shape[0]
        num_coords = self.coords_mat.shape[0]
        z_mat = forward_model_output.reshape(num_samples, num_coords, -1, order="F")
        if forward_model_features is not None:
            z_mat = np.concatenate((z_mat, forward_model_features), axis=2)
        # Get the response matrices of the multi-fidelity mapping
        breakpoint()
        m_f_mat, var_y_mat = self.mf_interface.evaluate(z_mat)

        num_tau_samples = 20  # 10 #20  # 200
        log_lik_lst = []
        posterior_samples_tau = []
        kld_samples = []
        for m_f_vec, var_y_vec in zip(m_f_mat, var_y_mat, strict=True):

            # TODO: try SMC or again MCMC/hamiltonian instead of SVI
            log_lik_lst.append(
                self.single_tau_svi_run(m_f_vec, var_y_vec, num_tau_samples)
            )
            posterior_samples_tau.append(copy.copy(self.temp_posterior_samples_tau))

            sample_batch = np.array(posterior_samples_tau[-1]).reshape(-1, 1)
            variational_params = self.tau_svi.variational_params_optimal
            log_posterior_tau = self.tau_svi.variational_distribution_obj.logpdf(
                variational_params, np.log(sample_batch)
            )
            log_prior_tau = self.tau_svi.parameters.joint_logpdf(sample_batch)
            kld_samples.append(log_prior_tau - log_posterior_tau)

        self.posterior_samples_tau = np.array(posterior_samples_tau)

        log_likelihood_samples = np.array(log_lik_lst).squeeze()
        kld_samples = np.array(kld_samples).squeeze()

        em_log_likelihood = np.mean(log_likelihood_samples + kld_samples, axis=1)
        mean_noise = np.mean(1 / self.posterior_samples_tau)
        std_noise = np.std(1 / self.posterior_samples_tau)
        _logger.info("Mean of noise: {}".format(mean_noise))
        _logger.info("Std of noise: {}".format(std_noise))
        return em_log_likelihood.reshape(-1, 1)

    def single_tau_svi_run(self, m_f_vec, var_vec, num_tau_samples):
        # def single_tau_svi_run(self, y_sample, num_tau_samples):
        self.tau_svi.model.update_unnormalized_posterior(m_f_vec, var_vec)
        self.tau_svi.variational_params = (
            self.tau_svi.variational_distribution_obj.construct_variational_parameters(
                np.array([0.0]), np.array([1])
            )
        )
        self.tau_svi.n_sims = 0
        self.tau_svi.nan_in_gradient_counter = 0
        self.tau_svi.stochastic_optimizer.iteration = 0
        self.tau_svi.stochastic_optimizer.done = False
        for key in self.tau_svi.iteration_data.keys():
            setattr(self.tau_svi.iteration_data, key, [])

        self.tau_svi.run()

        variational_parameters = self.tau_svi.variational_params_optimal

        temp_posterior_samples_tau_param = (
            self.tau_svi.variational_distribution_obj.draw(
                variational_parameters, num_tau_samples
            ).flatten()
        )
        self.temp_posterior_samples_tau = np.exp(temp_posterior_samples_tau_param)

        # loop over tau samples per x sample
        log_likelihood_samples_lst = []

        # TODO here we sample only for one posterior otherwise we would need to hand this in the function
        for tau in self.temp_posterior_samples_tau:
            noise_var = 1 / tau
            var_vec = (noise_var + var_vec).flatten()
            self.tau_svi.model.normal_distribution.update_variance(var_vec)
            log_likelihood_samples_lst.append(
                self.tau_svi.model.normal_distribution.logpdf(m_f_vec).flatten()
            )

        log_likelihood_samples = np.array(log_likelihood_samples_lst).flatten()
        return log_likelihood_samples


class PrecisionLikDummy:
    def __init__(self, normal_distribution, spatial_dim):
        self.parameters_mean = None  # just a dummy value
        self.variational_distribution = None  # can stay empty is just a hack
        self.variational_params = None  # can stay empty is just a hack
        self.m_f = None
        self.var_y = None
        self.y_sample = None
        self.normal_distribution = normal_distribution
        self.spatial_dim = spatial_dim

    def update_unnormalized_posterior(self, m_f, var_y):
        # def update_unnormalized_posterior(self, y_sample):
        self.m_f = m_f
        self.var_y = var_y

    # eval and grad of likelihood wrt tau_param
    def evaluate_and_gradient(self, tau_param_batch):
        log_lik_mf_lst = []
        grad_tau_log_lik_mf_lst = []
        tau_batch = np.exp(tau_param_batch)
        for tau in tau_batch:
            noise_var = 1 / tau
            var_vec = noise_var + self.var_y
            self.normal_distribution.update_variance(var_vec)
            log_lik_mf_lst.append(self.normal_distribution.logpdf(self.m_f))
            grad_logpdf_var = self.normal_distribution.grad_logpdf_var(self.m_f)
            grad_tau_log_lik_mf_lst.append(np.sum(grad_logpdf_var * (-1 / tau)))

        mf_log_likelihood = np.array(log_lik_mf_lst).reshape(-1, 1)
        grad_tau_mf_log_likelihood = np.array(grad_tau_log_lik_mf_lst).reshape(-1, 1)

        return mf_log_likelihood, grad_tau_mf_log_likelihood

    def logpdf(self, tau_param_batch, m_f, var_y):
        log_lik_mf_lst = []
        tau_batch = np.exp(tau_param_batch)
        for tau in tau_batch:
            noise_var = 1 / tau
            var_vec = noise_var + var_y
            self.normal_distribution.update_variance(var_vec)
            log_lik_mf_lst.append(self.normal_distribution.logpdf(m_f))

        mf_log_likelihood = np.array(log_lik_mf_lst).reshape(-1, 1)

        return mf_log_likelihood
