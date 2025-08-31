"""Sparse Gaussian Markov Random Field distribution."""

import numpy as np
from queens.distributions._distribution import Continuous
from queens.utils.logger_settings import log_init_args
from scipy.sparse import csr_array
from sksparse.cholmod import cholesky


class GaussianMarkovRandomField(Continuous):
    """Sparse Gaussian Markov Random Field distribution."""

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
        self.chol = None  # lazy, rebuilt per process
        self.logpdf_const = None  # recomputed with factor
        super().__init__(self.mean, self.precision, dimension)  # dummy

    # ensure factor exists in the *current* process
    def _ensure_factor(self):
        if self.chol is None:
            # CHOLMOD expects CSC; convert at factorization time
            try:
                # scipy >=1.8: csr_array/csc_array; >=1.4: csr_matrix/csc_matrix
                to_csc = getattr(self.precision, "tocsc", None)
                if to_csc is not None:
                    self.precision = to_csc()
            except Exception:
                pass
            self.chol = cholesky(self.precision)
            self.logpdf_const = GaussianMarkovRandomField.get_logpdf_const(
                self.dimension, self.chol
            )

    # make object pickle-safe: drop non-picklable factor on ship
    def __getstate__(self):
        d = self.__dict__.copy()
        d["chol"] = None
        d["logpdf_const"] = None
        return d

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
        """Build precision (Laplacian-like) as sparse array."""
        num_1d = int(
            round(
                (element_neighbor_mapping.shape[0] / field_dimension)
                ** (1 / spatial_dimension)
            )
        )
        h = L / (num_1d + 1)
        size = field_dimension * (num_1d**spatial_dimension)

        values = [2 * spatial_dimension] * size
        row_idx = list(range(size))
        col_idx = list(range(size))

        for neighbors in element_neighbor_mapping:
            for i in range(0, neighbors.size, 2):
                if neighbors[i] != neighbors[i + 1]:
                    values.append(-1)
                    row_idx.append(int(neighbors[i]))
                    col_idx.append(int(neighbors[i + 1]))

        P = csr_array((values, (row_idx, col_idx)), shape=(size, size)) * (1 / h**2)
        return P

    @staticmethod
    def get_logpdf_const(dim, chol):
        log_det_P = 2 * np.sum(np.log(np.abs(chol.D())))
        return -dim / 2 * np.log(2.0 * np.pi) + 0.5 * log_det_P

    def draw(self, num_draws=1):
        """Draw samples from the distribution."""
        self._ensure_factor()
        uncorrelated_vector = np.random.randn(self.dimension, num_draws)
        samples = []
        for vector in uncorrelated_vector.T:
            samples.append(self.mean + self.chol(vector))
        return np.array(samples)

    def logpdf(self, x):
        """Log of the probability density function."""
        self._ensure_factor()
        dist = x - self.mean
        out = []
        # Use matrix ops; keep your original structure
        for sample in dist:
            out.append(
                -0.5 * (sample.T @ (self.precision @ sample.T)) + self.logpdf_const
            )
        return np.array(out)

    def update_delta(self, delta):
        """Update the precision parameter vector."""
        self.delta = delta
        self.precision = self.precision_raw.multiply(self.delta)
        # Invalidate factor; rebuild lazily where needed
        self.chol = None
        self.logpdf_const = None

    def grad_logpdf(self, x):
        """Gradient of the log pdf w.r.t. x."""
        # If grad recomputes delta, invalidate factor first
        diff = x - self.mean
        a = 1.0e-9 + 0.5 * diff.shape[1]

        mean = []
        for d_e in diff:
            mean.append(self.precision_raw.dot(d_e).dot(d_e))
        mean = np.array(mean)
        mean = np.nanmean(mean, axis=0)
        b = 1.0e-9 + 0.5 * mean

        delta = a / b
        self.update_delta(delta)  # sets chol=None
        # No need to refactor for gradient itself
        grad_logpdf = -(self.precision @ diff.T).T
        return grad_logpdf

    def pdf(self, x):
        return np.exp(self.logpdf(x))

    def cdf(self, x):
        raise NotImplementedError()

    def ppf(self, q):
        raise NotImplementedError()
