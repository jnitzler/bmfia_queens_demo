#
# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (c) 2024-2025, QUEENS contributors.
#
# This file is part of QUEENS.
#
# QUEENS is free software: you can redistribute it and/or modify it under the terms of the GNU
# Lesser General Public License as published by the Free Software Foundation, either version 3 of
# the License, or (at your option) any later version. QUEENS is distributed in the hope that it will
# be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details. You
# should have received a copy of the GNU Lesser General Public License along with QUEENS. If not,
# see <https://www.gnu.org/licenses/>.
#
"""Convenience wrapper around Jobscript Driver."""

import logging
from pathlib import Path

import numpy as np
from queens.drivers.jobscript import Jobscript
from queens.utils.injector import inject
from queens.utils.logger_settings import log_init_args

_logger = logging.getLogger(__name__)

_JOBSCRIPT_TEMPLATE = (
    "{{ mpi_cmd }} -np {{ num_procs }} {{ executable }} {{ input_file }}"
)


class Deal(Jobscript):
    """Driver to run a generic MPI run."""

    @log_init_args
    def __init__(
        self,
        parameters,
        input_templates,
        executable,
        files_to_copy=None,
        data_processor=None,
        gradient_data_processor=None,
        feature_data_processor=None,
        mpi_cmd="/usr/bin/mpirun --bind-to none",
    ):
        """Initialize MPI object.

        Args:
            parameters (Parameters): Parameters object
            input_templates (str, Path, dict): path to simulation input template
            executable (str, Path): path to main executable of respective software
            files_to_copy (list, opt): files or directories to copy to experiment_dir
            data_processor (obj, opt): instance of data processor class
            gradient_data_processor (obj, opt): instance of data processor class for gradient data
            mpi_cmd (str, opt): mpi command
        """
        extra_options = {
            "mpi_cmd": mpi_cmd,
        }
        super().__init__(
            parameters=parameters,
            input_templates=input_templates,
            jobscript_template=_JOBSCRIPT_TEMPLATE,
            executable=executable,
            files_to_copy=files_to_copy,
            data_processor=data_processor,
            gradient_data_processor=gradient_data_processor,
            extra_options=extra_options,
        )
        self.feature_data_processor = feature_data_processor


    def prepare_input_files(self, sample_dict, experiment_dir, input_files):
        """Prepare and parse data to input files.

        Args:
            sample_dict (dict): Dict containing sample.
            experiment_dir (Path): Path to QUEENS experiment directory.
            input_files (dict): Dict with name and path of the input file(s).
        """
        for input_template_name, input_template_path in self.input_templates.items():
            numpy_input_path = input_files[input_template_name].with_suffix(".npy")
            output_dir = numpy_input_path.parent / "output"
            output_dir = output_dir / "bmfia"
            sample_dict["numpy_input_path"] = numpy_input_path
            sample_dict["output_dir"] = output_dir
            inject(
                sample_dict,
                experiment_dir / input_template_path.name,
                input_files[input_template_name],
            )
            # filter a list of floats from sample dict values
            samples = [v for v in sample_dict.values() if isinstance(v, float)]
            np.save(Path(input_files[input_template_name].with_suffix("")), np.array(samples))

    def _get_results(self, output_dir):
        """Get results from driver run.

        Args:
            output_dir (Path): Path to output directory.

        Returns:
            result (np.array): Result from the driver run.
            gradient (np.array, None): Gradient from the driver run (potentially None).
        """
        result = None
        if self.data_processor:
            result = self.data_processor.get_data_from_file(output_dir)
            _logger.debug("Got result: %s", result)
        
        features = None
        if self.feature_data_processor:
            features = self.feature_data_processor.get_data_from_file(output_dir)
            _logger.debug("Got features: %s", features)

        gradient = None
        if self.gradient_data_processor:
            gradient = self.gradient_data_processor.get_data_from_file(output_dir)
            _logger.debug("Got gradient: %s", gradient)
        return result, gradient, features