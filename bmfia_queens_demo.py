# Paths to external computational models and directories
from pathlib import Path

## Paths to model input files
lf_input_file_template = Path("./external_models/lf_input_template.json")
lf_adjoint_input_file_template = Path(
    "./external_models/lf_adjoint_input_template.json"
)
hf_input_file_template = Path("./external_models/hf_input_template.json")

## Paths to model executables
lf_model_path = Path("./external_models/darcy")
lf_adjoint_model_path = Path("./external_models/darcy_adjoint")
hf_model_path = Path("./external_models/darcy")

## Path to output directory
output_dir_path = Path("./output")
output_dir_path_initial_training = output_dir_path / "initial_training_phase"
output_dir_path_inference = output_dir_path / "inference_phase"

## Paths to the LF and HF mesh dof coordinates
lf_dofs_coords_path = Path("./data/dof_coords_5.npy")  # refinement level 5
hf_dofs_coords_path = Path("./data/dof_coords_6.npy")  # refinement level 6

## quick check if all these paths exist
assert (
    lf_input_file_template.exists()
), "Low-fidelity input file template does not exist."
assert (
    lf_adjoint_input_file_template.exists()
), "Low-fidelity adjoint input file template does not exist."
assert (
    hf_input_file_template.exists()
), "High-fidelity input file template does not exist."
assert lf_model_path.exists(), "Low-fidelity model executable does not exist."
assert (
    lf_adjoint_model_path.exists()
), "Low-fidelity adjoint model executable does not exist."
assert hf_model_path.exists(), "High-fidelity model executable does not exist."
assert output_dir_path.exists(), "Output directory does not exist."
assert (
    output_dir_path_initial_training.exists()
), "Output directory for initial training phase does not exist."
assert (
    output_dir_path_inference.exists()
), "Output directory for inference phase does not exist."
assert lf_dofs_coords_path.exists(), "LF mesh dof coordinates file does not exist."
assert hf_dofs_coords_path.exists(), "HF mesh dof coordinates file does not exist."

# -------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------

# Initial training phase of BMFIA
## Import all necessary modules from QUEENS to setup models, drivers and schedulers
from queens.global_settings import GlobalSettings
from queens.data_processors.numpy_file import NumpyFile as NumpyDataProc
from queens.drivers.mpi import Mpi as MpiDriver
from queens.schedulers.local import Local as LocalScheduler
from queens.models.simulation import Simulation as SimulationModel
from queens.parameters import Parameters
from queens.distributions.mean_field_normal import MeanFieldNormal
from bmfia.bmfia_iterator import BmfiaIterator
from bmfia.uniform_grid_interpolator import UniformGridInterpolator
from queens.main import run_iterator


# define the main execution (necessary for multiprocessing on some systems)
if __name__ == "__main__":
    ## Setup global settings
    experiment_name = "bmfia_initial_training_phase"
    global_settings_initial = GlobalSettings(
        experiment_name=experiment_name, output_dir=output_dir_path_initial_training
    )

    ## Setup the parameter definition for the models
    x_vec = MeanFieldNormal(
        0, 1, 1000
    )  # TODO: exchange this for actual GMRF distribution
    parameters = Parameters(x_vec=x_vec)

    ## Setup data processors, the velocity field is here additionally stored in a numpy file
    ## with file breakpoint()name ending "_sol.npy"
    lf_data_processor = NumpyDataProc(
        file_name_identifier="_sol.npy", file_options_dict={"delete_field_data": False}
    )
    hf_data_processor = NumpyDataProc(
        file_name_identifier="_sol.npy", file_options_dict={"delete_field_data": False}
    )

    ## Setup drivers
    mpi_driver_lf = MpiDriver(
        parameters,
        lf_input_file_template,
        lf_model_path,
        files_to_copy=None,
        data_processor=lf_data_processor,
        gradient_data_processor=None,
        mpi_cmd="/usr/bin/mpirun --bind-to none",
    )
    mpi_driver_hf = MpiDriver(
        parameters,
        hf_input_file_template,
        hf_model_path,
        files_to_copy=None,
        data_processor=hf_data_processor,
        gradient_data_processor=None,
        mpi_cmd="/usr/bin/mpirun --bind-to none",
    )

    ## setup the spatial fields transfer from LF to HF grid
    ## a super simple rectangular grid interpolator given dof coordinates
    ## should be replaced by actual (e.g., vtk, vtu) mesh interpolation
    uniform_grid_interpolator = UniformGridInterpolator(
        lf_dofs_coords_path, hf_dofs_coords_path
    )

    ## setup initial design configuration for training data points
    initial_design = {
        "num_HF_eval": 100,
        "seed": 1,
        "num_hyper_params_steps": 10,
        "delta_range": [5.0e-3, 1.0e-1],
        "mean_range": [1.0, 1.0],
        "L": 1.0,
    }

    # -------------------------------------------------------------------------------------------------------
    # -------------------------------------------------------------------------------------------------------

    # build the schedulers and the rest of the QUEENS model in context
    # this makes sure that everything is closed properly at the end
    # here we actually run the QUEENS run / the initial training phase of BMFIA
    with global_settings_initial:
        ## Setup schedulers with local MPI scheduler; allow 6 jobs to run in parallel on one processor each
        local_scheduler_lf = LocalScheduler(
            experiment_name,
            num_jobs=6,
            num_procs=1,
            restart_workers=False,
            verbose=True,
        )
        local_scheduler_hf = LocalScheduler(
            experiment_name,
            num_jobs=6,
            num_procs=1,
            restart_workers=False,
            verbose=True,
        )
        print("schedulers set up")
        ## Setup the QUEENS simulation models
        lf_model = SimulationModel(scheduler=local_scheduler_lf, driver=mpi_driver_lf)
        hf_model = SimulationModel(scheduler=local_scheduler_hf, driver=mpi_driver_hf)
        print("models set up")
        ## Setup the BMFIA iterator -> here we sample randomly from the conditional prior
        ## The prior being a GMRF, conditioned on a specific mean and precision factor
        bmfia_iterator = BmfiaIterator(
            parameters,
            hf_model,
            lf_model,
            initial_design,
            grid_interpolator=uniform_grid_interpolator,
            global_settings=global_settings_initial,
        )

        ## finally run the BMFIA iterator (the initial training phase) / start the QUEENS run
        run_iterator(bmfia_iterator, global_settings=global_settings_initial)
