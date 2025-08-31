"""Adjoint model with a slight custom modification for BMFIA."""

from queens.models.adjoint import Adjoint
from queens.utils.config_directories import current_job_directory
import numpy as np
import os


class AdjointBMFIA(Adjoint):
    """Adjoint model for BMFIA"""

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
        num_samples = samples.shape[0]

        # get last job_ids
        last_job_ids = [
            self.scheduler.next_job_id - num_samples + i for i in range(num_samples)
        ]
        experiment_dir = self.scheduler.experiment_dir

        # write adjoint data for each sample to adjoint files in old job directories
        for job_id, grad_objective in zip(last_job_ids, upstream_gradient, strict=True):
            job_dir = current_job_directory(experiment_dir, job_id)
            adjoint_file_path = job_dir.joinpath(self.adjoint_file)

            # Use os.makedirs to create the subdirectory if it does not exist
            # The exist_ok=True parameter prevents an error if the directory already exists
            os.makedirs(os.path.dirname(adjoint_file_path), exist_ok=True)
            if len(grad_objective.shape) == 3:

                grad_objective = np.rot90(grad_objective, k=-1, axes=[0, 1])  # was k=-1
            else:
                grad_objective = grad_objective.transpose(3, 0, 1, 2)
            grad_objective = grad_objective.flatten(order="F")  # this worked for 2D
            np.save(adjoint_file_path, grad_objective)

        # evaluate the adjoint model
        gradient = self.scheduler.evaluate(
            samples, driver=self.gradient_driver, job_ids=last_job_ids
        )["result"]
        return gradient
