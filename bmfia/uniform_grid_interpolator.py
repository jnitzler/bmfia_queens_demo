"""A very simple grid interpolator for rectangular meshes."""

import logging

import numpy as np
from scipy.interpolate import griddata

_logger = logging.getLogger(__name__)


class UniformGridInterpolator:
    """Simple grid interpolator for rectangular meshes.

    The class takes in two numpy files that contain the coordinates
    of the associated degrees of freedom (dofs) of the low-fidelity (LF)
    and high-fidelity (HF) mesh. The dof coordinates are used to interpolate
    spatial fields from the LF to the HF mesh using linear interpolation. The
    order of the dofs in the numpy files must correspond to the order of the global
    dofs in the simulation models. Currently, we assume that the meshes are uniform
    and rectangular (simplest case). We write the numpy files directly in the models
    itself, e.g., in Deal.II using the DataOut class, for this simple example.

    For more complex mesh interpolation, e.g., unstructured meshes, one could
    consider using vtk mesh interpolation using the python package `pyvista`.
    """

    def __init__(self, lf_dofs_coords_path, hf_dofs_coords_path):
        """Initialize grid interpolator."""
        self.lf_dofs_coords, self.hf_dofs_coords = self._read_dof_coords(
            lf_dofs_coords_path, hf_dofs_coords_path
        )

    @staticmethod
    def _read_dof_coords(lf_dofs_path, hf_dofs_path):
        """Load the LF and HF dof coordinates with numpy."""
        lf_dof_coords = np.load(lf_dofs_path)
        hf_dof_coords = np.load(hf_dofs_path)
        _logger.info("Read LF and HF dofs from %s and %s", lf_dofs_path, hf_dofs_path)
        return lf_dof_coords, hf_dof_coords

    def transfer_lf_to_hf_grid(self, x_lf):
        """Transfer spatial fields from one mesh to another.

        Args:
            x_lf (np.array): Value of LF dofs.

        Returns:
            x_hf (np.array): Interpolated values of HF dofs.
        """
        x_hf = griddata(
            self.lf_dofs_coords, x_lf.T, self.hf_dofs_coords, method="linear"
        ).T

        return x_hf
