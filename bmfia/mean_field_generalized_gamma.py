"""Mean-field normal distribution."""

import numpy as np
from queens.distributions._distribution import Continuous
from scipy.special import gamma  # pylint:disable=no-name-in-module
from scipy.special import digamma, gammainc, gammaln, polygamma


class MeanFieldGeneralizedGammaDistribution(Continuous):
    """Mean-field Gamma distribution.

    Attributes:
        alpha (np.ndarray): shape parameter vector
        beta (np.ndarray): rate parameter vector
    """

    def __init__(self, alpha, beta, dimension):
        """Initialize normal distribution.

        Args:
            alpha (np.ndarray): shape parameter vector
            beta (np.ndarray): rate parameter vector
            dimension (int): dimensionality of the distribution
        """
        self.alpha = alpha
        self.beta = beta
        mean = digamma(alpha) - np.log(beta)
        covariance = polygamma(1, alpha)
        self.standard_deviation = np.sqrt(covariance)
        super().__init__(mean, covariance, dimension)

    def update_parameters(self, alpha, beta):
        """Update the parameters of the mean field distribution.

        Args:
            alpha (np.array): New alpha vector
            beta (np.array): New beta vector
        """
        self.alpha = alpha
        self.beta = beta
        self.mean = alpha / beta
        self.covariance = alpha / beta**2
        self.standard_deviation = np.sqrt(self.covariance)

    def cdf(self, x):
        """Cumulative distribution function.

        Args:
            x (np.ndarray): Positions at which the cdf is evaluated

        Returns:
            cdf (np.ndarray): cdf at evaluated positions
        """
        raise NotImplementedError(
            "CDF not implemented yet for generalized mean-field gamma distribution."
        )

    def draw(self, num_draws=1):
        """Draw samples.

        Args:
            num_draws (int, optional): Number of draws

        Returns:
            samples (np.ndarray): Drawn samples from the distribution
        """
        raise NotImplementedError(
            "Sampling not implemeted for mean-field generalized Gamma distribution."
        )

    def logpdf(self, x):
        """Log of the probability density function.

        Args:
            x (np.ndarray): Positions at which the log pdf is evaluated

        Returns:
            logpdf (np.ndarray): log pdf at evaluated positions
        """
        logpdf = np.sum(
            self.alpha * np.log(self.beta)
            - gammaln(self.alpha)
            + (self.alpha - 1) * x
            - self.beta * np.exp(x)
            + x,
            axis=1,
        )

        return logpdf

    def grad_logpdf(self, x, _extended_x):
        """Gradient of the log pdf with respect to x.

        Args:
            x (np.ndarray): Positions at which the gradient of log pdf is evaluated

        Returns:
            grad_logpdf (np.ndarray): Gradient of the log pdf evaluated at positions
        """
        gradients_batch = (self.alpha - 1) - self.beta * np.exp(x) + 1
        # gradients_batch = (self.alpha - 1)/np.exp(x) - self.beta
        # gradients_batch = factor * ((self.alpha - 1) - self.beta * np.exp(x))
        gradients_batch = gradients_batch.reshape(x.shape[0], -1)

        return gradients_batch

    def pdf(self, x):
        """Probability density function.

        Args:
            x (np.ndarray): Positions at which the pdf is evaluated

        Returns:
            pdf (np.ndarray): pdf at evaluated positions
        """
        pdf = np.exp(self.logpdf(x))
        return pdf

    def ppf(self, q):
        """Percent point function (inverse of cdf — quantiles).

        Args:
            q (np.ndarray): Quantiles at which the ppf is evaluated
        """
        raise NotImplementedError(
            "The function ppf is not implemented for the mean-field Gamma distribution."
        )
