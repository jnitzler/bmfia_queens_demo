from queens.utils.logger_settings import log_init_args
from queens.variational_distributions._variational_distribution import Variational

import numpy as np
import scipy
from scipy.sparse import csr_array, dia_array, vstack as sparse_vstack


class SparseNormalVariational(Variational):
    r"""Sparse multivariate normal distribution.

    Uses the parameterization (as in [1])
    :math:`parameters=[\mu, \lambda]`, where :math:`\mu` are the mean values and
    :math:`\lambda` is an array containing the nonzero entries of the lower Cholesky
    decomposition of the covariance matrix :math:`L`:
    :math:`\lambda=[L_{00},L_{10},L_{11},L_{20},L_{21},L_{22}, ...]`.
    This allows the parameters :math:`\lambda` to be unconstrained.

    References:
        [1]: Kucukelbir, Alp, et al. "Automatic differentiation variational inference."
             The Journal of Machine Learning Research 18.1 (2017): 430-474.

    Attributes:
        half_off_diag_width (int): Half width of the off-diagonal band
        n_parameters (int): Number of parameters used in the parameterization.
        row_idx_chol (np.ndarray): Row indices of the sparse cholesky decomposition
        col_idx_chol (np.ndarray): Column indices of the sparse cholesky decomposition
        row_idx_jacobian (np.ndarray): Row indices of the sparse jacobian cholesky decomposition
        col_idx_jacobian (np.ndarray): Column indices of the sparse jacobian cholesky decomposition
        data_idx_jacobian (np.ndarray): Data indices of the sparse jacobian cholesky decomposition
        nugget_L_diag (np.ndarray): nugget terms on the diagonal of the cholesky decomposition
    """

    @log_init_args
    def __init__(self, dimension, half_off_diag_width=None, nugget_var_diag=None):
        """Initialize sparse variational distribution.

        Args:
            dimension (int): dimension of the RV
            half_off_diag_width (int): Half width of the off-diagonal band
            nugget_var_diag (float): Nugget variance on the diagonal of the covariance matrix
        """
        super().__init__(dimension)
        self.half_off_diag_width = int(half_off_diag_width)
        n_parameters = dimension  # parameters for the mean
        for i in range(0, self.half_off_diag_width + 1):
            n_parameters += dimension - i
        self.n_parameters = n_parameters
        self.row_idx_chol, self.col_idx_chol = (
            SparseNormalVariational._get_sparse_chol_indices(
                self.dimension, self.half_off_diag_width
            )
        )
        (
            self.row_idx_jacobian,
            self.col_idx_jacobian,
            self.data_idx_jacobian,
        ) = SparseNormalVariational._get_sparse_chol_jacobi_indices(
            self.dimension, self.half_off_diag_width, self.n_parameters
        )
        self.nugget_L_diag = np.sqrt(nugget_var_diag)

    @staticmethod
    def _get_sparse_chol_indices(dimension, half_off_diag_width):
        """Get the indices for the sparse cholesky decomposition.

        Args:
            dimension (int): Dimension of the covariance matrix
            half_off_diag_width (int): Half width of the off-diagonal band

        Returns:
            row_idx (np.ndarray): Row indices of the sparse cholesky decomposition
            col_idx (np.ndarray): Column indices of the sparse cholesky decomposition
        """
        col_idx = []
        row_idx = []
        for row in range(0, dimension):
            for col in range(np.max((row - half_off_diag_width, 0)), row + 1):
                col_idx.append(col)
                row_idx.append(row)

        return np.array(row_idx), np.array(col_idx)

    @staticmethod
    def _get_sparse_chol_jacobi_indices(dimension, half_off_diag_width, n_parameters):
        """Get the indices for sparse jac. of chol. decomposition.

        Args:
            dimension (int): Dimension of the covariance matrix
            half_off_diag_width (int): Half width of the off-diagonal band
            n_parameters (int): Number of variational parameters

        Returns:
            row_idx_jacobian (np.ndarray): Row indices of the sparse jacobian cholesky decomposition
            col_idx_jacobian (np.ndarray): Column indices of the sparse jacobian
                                          cholesky decomposition
            data_idx_jacobian (np.ndarray): Data indices of the sparse jacobian cholesky
                                            decomposition
        """
        row_idx_jacobian = np.arange(0, n_parameters - dimension)
        col_idx_jacobian_part_1 = np.repeat(
            np.arange(0, half_off_diag_width), np.arange(1, half_off_diag_width + 1)
        )
        col_idx_jacobian_part_2 = np.repeat(
            np.arange(half_off_diag_width, dimension), half_off_diag_width + 1
        )
        col_idx_jacobian = np.concatenate(
            (col_idx_jacobian_part_1, col_idx_jacobian_part_2)
        )

        data_idx_jacobian = []
        for i in range(0, dimension):
            data_idx_jacobian.append(np.arange(max(0, i - half_off_diag_width), i + 1))
        data_idx_jacobian = np.concatenate(data_idx_jacobian)

        return row_idx_jacobian, col_idx_jacobian, data_idx_jacobian

    def initialize_variational_parameters(self, random=False):
        r"""Initialize variational parameters.

        Default initialization:
            :math:`\mu=0` and :math:`L=diag(1)` where :math:`\Sigma=LL^T`

        Random initialization:
            :math:`\mu=Uniform(-0.1,0.1)` :math:`L=diag(Uniform(0.9,1.1))` where :math:`\Sigma=LL^T`

        Args:
            random (bool, optional): If True, a random initialization is used. Otherwise the
                                     default is selected
        """
        raise NotImplementedError(
            "Random initialization not implemented for sparse Normal"
        )

    def construct_variational_parameters(
        self, mean, covariance
    ):  # pylint: disable=arguments-differ
        """Construct the variational parameters from mean and covariance.

        Args:
            mean (np.ndarray): Mean values of the distribution (n_dim x 1)
            covariance (np.ndarray): Covariance matrix of the distribution (n_dim x n_dim)

        Returns:
            variational_parameters (np.ndarray): Variational parameters
        """
        if mean.size == covariance.size:
            cholesky_covariance = np.sqrt(covariance)
            variational_parameters = mean.flatten()
            cholesky_params = []
            for row, col in zip(self.row_idx_chol, self.col_idx_chol, strict=True):
                if row == col:
                    entry = np.log(cholesky_covariance[row])
                else:
                    entry = 0.0
                cholesky_params.append(entry)

            variational_parameters = np.hstack(
                (variational_parameters, np.array(cholesky_params))
            )
        else:
            raise ValueError(
                f"Dimension of the mean value {len(mean)} does not equal covariance dimension"
                f"{covariance.shape}"
            )
        return variational_parameters

    def reconstruct_distribution_parameters(
        self, variational_parameters, return_cholesky=False
    ):
        """Reconstruct mean value, covariance and its Cholesky decomposition.

        Args:
            variational_parameters (np.ndarray): Variational parameters
            return_cholesky (bool, optional): Return the L if desired
        Returns:
            mean (np.ndarray): Mean value of the distribution (n_dim x 1)
            cov (np.ndarray): Covariance of the distribution (n_dim x n_dim)
            L (np.ndarray): Cholesky decomposition of the covariance matrix (n_dim x n_dim)
        """
        variational_params = variational_parameters.copy()
        mean = variational_params[: self.dimension].reshape(-1, 1)
        cholesky_array = variational_params[self.dimension :]

        # transform the diagonal elements of the following sparse matrix
        cholesky_array[self.row_idx_chol == self.col_idx_chol] = np.exp(
            cholesky_array[self.row_idx_chol == self.col_idx_chol]
        )

        # transform the off diagonal entries with value**3 for better convergence
        cholesky_array[self.row_idx_chol != self.col_idx_chol] = (
            cholesky_array[self.row_idx_chol != self.col_idx_chol]
        ) ** 3

        cholesky_covariance = csr_array(
            (cholesky_array, (self.row_idx_chol, self.col_idx_chol)),
            shape=(self.dimension, self.dimension),
        )

        cov = cholesky_covariance.dot(
            cholesky_covariance.transpose()
        )  # note this is a sparse matrix multiplication
        if return_cholesky:
            return mean, cov, cholesky_covariance

        del variational_params, cholesky_array
        return mean, cov

    def _grad_reconstruct_distribution_parameters(self, variational_params):
        """Gradient of the parameter reconstruction.

        Args:
            variational_parameters (np.ndarray): Variational parameters

        Returns:
            grad_reconstruct_params (np.ndarray): Gradient vector of the reconstruction
                                                w.r.t. the variational parameters
        """
        variational_parameters = variational_params.copy()
        grad_mean = np.ones((1, self.dimension))
        grad_cholesky = np.ones((1, self.n_parameters - self.dimension))
        # incorporate the transformation of the diagonal elements
        grad_cholesky[:, self.row_idx_chol == self.col_idx_chol] = np.exp(
            variational_parameters[self.dimension :][
                self.row_idx_chol == self.col_idx_chol
            ]
        )
        # incorporate the transformation of the off diagonal elements
        grad_cholesky[:, self.row_idx_chol != self.col_idx_chol] = (
            3 * (grad_cholesky[:, self.row_idx_chol != self.col_idx_chol]) ** 2
        )

        grad_reconstruct_params = np.hstack((grad_mean, grad_cholesky))
        del variational_parameters, grad_cholesky
        return grad_reconstruct_params

    def draw(self, variational_parameters, n_draws=1):
        """Draw *n_draw* samples from the variational distribution.

        Args:
            variational_parameters (np.ndarray): Variational parameters
            n_draw (int): Number of samples to draw

        Returns:
            samples (np.ndarray): Row-wise samples of the variational distribution
        """
        mean, _, L = self.reconstruct_distribution_parameters(
            variational_parameters, return_cholesky=True
        )
        sample = L.dot(np.random.randn(self.dimension, n_draws)).T + mean.reshape(1, -1)
        return sample

    def logpdf(self, variational_parameters, x):
        """Logpdf evaluated using the at samples *x*.

        Args:
            variational_parameters (np.ndarray): Variational parameters
            x (np.ndarray): Row-wise samples

        Returns:
            logpdf (np.ndarray): Row vector of the logpdfs
        """
        mean, _, L = self.reconstruct_distribution_parameters(
            variational_parameters, return_cholesky=True
        )
        y = np.atleast_2d(x).T - mean
        z = scipy.sparse.linalg.spsolve_triangular(L, y, lower=True)
        u = scipy.sparse.linalg.spsolve_triangular(
            csr_array(L.transpose()), z, lower=False
        )

        def col_dot_prod(x, y):
            return np.sum(x * y, axis=0)

        logpdf = (
            -0.5 * self.dimension * np.log(2 * np.pi)
            - np.sum(np.log(np.abs(L.diagonal())))
            - 0.5 * col_dot_prod(x.T - mean, u)
        )
        return logpdf.flatten()

    def pdf(self, variational_parameters, x):
        """Pdf of evaluated at given samples *x*.

        First computes the logpdf, which is numerically more stable for exponential distributions.

        Args:
            variational_parameters (np.ndarray): Variational parameters
            x (np.ndarray): Row-wise samples

        Returns:
            pdf (np.ndarray): Row vector of the pdfs
        """
        pdf = np.exp(self.logpdf(variational_parameters, x))
        return pdf

    def grad_params_logpdf(self, variational_parameters, x):
        """Logpdf gradient w.r.t. to the variational parameters.

        Evaluated at samples *x*. Also known as the score function.

        Args:
            variational_parameters (np.ndarray): Variational parameters
            x (np.ndarray): Row-wise samples

        Returns:
            score (np.ndarray): Column-wise scores
        """
        raise NotImplementedError(
            "Gradient of logpdf w.r.t. variational parameters not implemented"
        )

    def grad_logpdf_sample(self, sample_batch, variational_parameters):
        """Computes the gradient of the logpdf w.r.t. the *x*.

        Args:
            sample_batch (np.ndarray): Row-wise samples
            variational_parameters (np.ndarray): Variational parameters


        Returns:
            gradients_batch (np.ndarray): Gradients of the log-pdf w.r.t. the
            sample *x*. The first dimension of the
            array corresponds to the different samples.
            The second dimension to different dimensions
            within one sample. (Third dimension is empty
            and just added to keep slices two-dimensional.)
        """
        mean, K = self.reconstruct_distribution_parameters(
            variational_parameters, return_cholesky=False
        )
        gradient_lst = []
        for sample in sample_batch:
            y = -(sample.reshape(-1, 1) - mean)

            x = scipy.sparse.linalg.spsolve(K, y)

            if np.isnan(x).any():
                breakpoint()
            gradient_lst.append(x.reshape(-1, 1))

        gradients_batch = np.array(gradient_lst)
        return gradients_batch

    def fisher_information_matrix(self, variational_parameters):
        """Compute the Fisher information matrix analytically.

        Args:
            variational_parameters (np.ndarray): Variational parameters
        """
        raise NotImplementedError(
            "Fisher information matrix not implemented for sparse Normal"
        )

    def export_dict(self, variational_parameters):
        """Create a dict of the distribution based on the given parameters.

        Args:
            variational_parameters (np.ndarray): Variational parameters

        Returns:
            export_dict (dictionary): Dict containing distribution information
        """
        mean, cov, L = self.reconstruct_distribution_parameters(
            variational_parameters, return_cholesky=True
        )
        export_dict = {
            "type": "sparse_Normal",
            "mean": mean,
            "covariance": cov,
            "cholesky": L,
            "variational_parameters": variational_parameters,
        }
        return export_dict

    def conduct_reparameterization(self, variational_parameters, n_samples):
        """Conduct a reparameterization.

        Args:
            variational_parameters (np.ndarray): Array with variational parameters
            n_samples (int): Number of samples for current batch

        Returns:
            samples_mat (np.ndarray): Array of actual samples from the variational
            distribution
        """
        standard_normal_sample_batch = np.random.normal(
            0, 1, size=(n_samples, self.dimension)
        )

        mean, _, L = self.reconstruct_distribution_parameters(
            variational_parameters, return_cholesky=True
        )
        samples_mat = mean + L.dot(standard_normal_sample_batch.T)

        return samples_mat.T, standard_normal_sample_batch

    def jacobi_variational_parameters_reparameterization(
        self, standard_normal_sample_batch, variational_parameters
    ):
        r"""Calculate the gradient of the reparameterization.

        Args:
            standard_normal_sample_batch (np.ndarray): Standard normal distributed sample
                                                    batch
            variational_parameters (np.ndarray): Variational parameters

        Returns:
            jacobi_reparameterization_batch (np.ndarray): Tensor with Jacobi matrices for the
            reparameterization trick. The first dimension
            loops over the individual samples, the second
            dimension over variational parameters and the last
            dimension over the dimensions within one sample

        **Note:**
            We assume that *grad_reconstruct_params* is a row-vector containing the partial
            derivatives of the reconstruction mapping of the actual distribution parameters
            w.r.t. the variational parameters.

            The variable *jacobi_parameters* is the (n_parameters :math:`\times` dim_sample)
            Jacobi matrix of the reparameterization w.r.t. the distribution parameters,
            with differentiating after the distribution
            parameters in different rows and different output dimensions of the sample per
            column.
        """
        jacobi_reparameterization_lst = []
        grad_reconstruct_params = self._grad_reconstruct_distribution_parameters(
            variational_parameters
        )
        jacobi_mean = dia_array(
            (np.ones(self.dimension), np.array([0])),
            shape=(self.dimension, self.dimension),
        )

        for sample in standard_normal_sample_batch:
            data = sample[self.data_idx_jacobian]
            jacobi_cholesky = csr_array(
                (data, (self.row_idx_jacobian, self.col_idx_jacobian)),
                shape=(self.n_parameters - self.dimension, self.dimension),
            )

            jacobi_parameters = sparse_vstack((jacobi_mean, jacobi_cholesky))
            jacobi_reparameterization_lst.append(
                (jacobi_parameters.multiply(grad_reconstruct_params.T))
            )

        jacobi_reparameterization_batch = np.array(jacobi_reparameterization_lst)
        return jacobi_reparameterization_batch
