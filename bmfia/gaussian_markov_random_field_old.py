"""Sparse Gaussian Markov Random Field distribution."""

import numpy as np
from scipy.sparse import csr_array
from sksparse.cholmod import cholesky
from queens.utils.logger_settings import log_init_args
from queens.distributions._distribution import Continuous


class GaussianMarkovRandomField(Continuous):
    """Sparse Gaussian Markov Random Field distribution.

    Attributes:
    """

    @log_init_args
    def __init__(
        self,
        mean,
        dimension,
        L,
        neighbor_mapping_path,
        field_dimension=1,
        spatial_dimension=None,
    ):
        """Initialize the sparse Gaussian Markov Random Field distribution.

        Args:
            mean (np.ndarray): Mean vector
            dimension (int): Dimensionality of the distribution
            L (float): Length of the square domain
            neighbor_mapping_path (str): path to the file containing the neighbor mapping
            field_dimension (int): Dimensionality of the field
        """
        self.mean = mean * np.ones(dimension)
        self.dimension = dimension
        self.L = L
        self.delta = None
        self.element_neighbor_mapping = np.load(neighbor_mapping_path)
        self.field_dimension = field_dimension
        self.spatial_dimension = spatial_dimension
        self.precision_raw = GaussianMarkovRandomField.get_precision(
            self.spatial_dimension,
            self.L,
            self.element_neighbor_mapping,
            self.field_dimension,
        )
        self.precision = None
        self.chol = None
        self.logpdf_const = None
        super().__init__(self.mean, self.precision, dimension)  # dummy

    @staticmethod
    def _get_distance_gradient_mappings(element_neighbor_mapping):
        distance_gradient_mappings = []
        for neighbors in element_neighbor_mapping:  # rows
            for i in range(0, neighbors.size, 2):  # columns
                if neighbors[i] != neighbors[i + 1]:
                    distance_gradient_mappings.append([neighbors[i], neighbors[i + 1]])

        distance_gradient_mappings = np.array(distance_gradient_mappings)
        distance_gradient_mappings_sorted = np.sort(distance_gradient_mappings, axis=1)
        distance_gradient_mappings = np.unique(
            distance_gradient_mappings_sorted, axis=0
        )

        return distance_gradient_mappings

    @staticmethod
    def get_precision(spatial_dimension, L, element_neighbor_mapping, field_dimension):
        """Get the precision matrix of the distribution.

        Args:
            n (int): Dimensionality of the 1d distribution (cells per 1d)
            L (float): Length of the square domain
            element_neighbor_mapping (np.ndarray): Mapping of the elements to their neighbors

        Returns:
            precision (scipy_sparse.csc_array): Precision matrix
        """
        # -------- Define 1D precision matrix --------
        num_1d = int(
            round(
                (element_neighbor_mapping.shape[0] / field_dimension)
                ** (1 / spatial_dimension)
            )
        )
        h = L / (num_1d + 1)  # grid spacing

        # define the diagonal entries as array of length field_dimension * num_1d**spatial_dimension
        values = [2 * spatial_dimension] * field_dimension * (num_1d**spatial_dimension)
        row_idx = [i for i in range(field_dimension * num_1d**spatial_dimension)]
        col_idx = [i for i in range(field_dimension * num_1d**spatial_dimension)]

        # define now the entries for neighboring elements
        for neighbors in element_neighbor_mapping:  # rows
            for i in range(0, neighbors.size, 2):  # columns
                if neighbors[i] != neighbors[i + 1]:
                    # note that symmetry will be automatic due to neighbor mapping
                    # as neighbor mapping is symmetric
                    values.append(-1)
                    row_idx.append(neighbors[i])
                    col_idx.append(neighbors[i + 1])

        P = csr_array(
            (values, (row_idx, col_idx)),
            shape=(
                field_dimension * num_1d**spatial_dimension,
                field_dimension * num_1d**spatial_dimension,
            ),
        ) * (1 / h**2)
        return P

    @staticmethod
    def get_logpdf_const(dim, chol):
        """Get the constant for the log pdf.

        Args:
            dim (int): Dimensionality of the distribution
            chol (scipy_sparse.csc_array): Cholesky factor of the precision matrix

        Returns:
            logpdf_const (float): Constant for the log pdf
        """
        log_det_P = 2 * np.sum(np.log(np.abs(chol.D())))
        logpdf_const = -dim / 2 * np.log(2.0 * np.pi) + 0.5 * log_det_P
        return logpdf_const

    def draw(self, num_draws=1):
        """Draw samples from the distribution.

        Args:
            num_draws (int): Number of samples to draw

        Returns:
            samples (np.ndarray): Samples from the distribution
        """
        uncorrelated_vector = np.random.randn(self.dimension, num_draws)
        samples = []
        for vector in uncorrelated_vector.T:
            samples.append(self.mean + self.chol(vector))

        samples = np.array(samples)
        return samples

    def logpdf(self, x):
        """Log of the probability density function.

        Args:
            x (np.ndarray): Positions at which the log pdf is evaluated

        Returns:
            logpdf (np.ndarray): log pdf at evaluated positions
        """
        dist = x - self.mean
        logpdf = []
        for sample in dist:
            logpdf.append(
                -0.5 * (np.dot(sample.T, self.precision.dot(sample.T)))
                + self.logpdf_const
            )
        logpdf = np.array(logpdf)
        return logpdf

    def update_delta(self, delta):
        """Update the precision parameter vector."""
        self.delta = delta
        self.precision = self.precision_raw.multiply(self.delta)
        self.chol = cholesky(self.precision)
        self.logpdf_const = GaussianMarkovRandomField.get_logpdf_const(
            self.dimension, self.chol
        )

    def grad_logpdf(self, x, extended_x):
        """Gradient of the log pdf with respect to *x*.

        Args:
            x (np.ndarray): Positions at which the gradient of log pdf is evaluated

        Returns:
            grad_logpdf (np.ndarray): Gradient of the log pdf evaluated at positions
        """
        diff = x - self.mean
        diff_extended = extended_x - self.mean

        a = 1.0e-9 + 0.5 * diff.shape[1]

        mean = []
        for d_e in diff_extended:
            mean.append(self.precision_raw.dot(d_e).dot(d_e))

        mean = np.array(mean)
        mean = np.nanmean(mean, axis=0)
        b = 1.0e-9 + 0.5 * mean

        delta = a / b

        self.update_delta(delta)
        grad_logpdf = -self.precision.dot(diff.T).T

        return grad_logpdf


    def pdf(self, x):
        """Probability density function.

        Args:
            x (np.ndarray): Positions at which the pdf is evaluated

        Returns:
            pdf (np.ndarray): pdf at evaluated positions
        """
        pdf = np.exp(self.logpdf(x))
        return pdf

    def cdf(self, x):
        raise NotImplementedError()

    def ppf(self, q):
        raise NotImplementedError()
