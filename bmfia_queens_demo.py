# Paths to external computational models and directories
from pathlib import Path
import numpy as np

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

## Path to the neighbor mapping file for the GMRF prior
neighbor_mapping_path = Path("./data/neighbors_5.npy")  # refinement level 5

## Path to the training data file (will be written once initial training is done)
path_trainings_data = output_dir_path_initial_training / "bmfia_training_data.npz"

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
assert neighbor_mapping_path.exists(), "Neighbor mapping file does not exist."

# -------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------

from queens.data_processors.numpy_file import NumpyFile as NumpyDataProc
from queens.data_processors.csv_file import CsvFile as CsvDataProc

## Import all necessary modules from QUEENS to setup models, drivers and schedulers
from queens.global_settings import GlobalSettings
from queens.main import run_iterator
from queens.models.simulation import Simulation as SimulationModel
from queens.models.adjoint import Adjoint as AdjointSimulationModel
from queens.parameters import Parameters

from bmfia.bmfia_iterator import BmfiaIterator
from bmfia.deal_driver import Deal as DealDriver
from bmfia.gaussian_markov_random_field import GaussianMarkovRandomField
from bmfia.local_scheduler_w_features import LocalWFeatures
from bmfia.uniform_grid_interpolator import UniformGridInterpolator
from bmfia.gaussian_cnn import GaussianCNN
from bmfia.bayesian_mf_gaussian_likelihood import BMFGaussianModel

# define the main execution (necessary for multiprocessing on some systems)
if __name__ == "__main__":
    ## Setup global settings
    experiment_name = "bmfia_initial_training_phase"
    global_settings_initial = GlobalSettings(
        experiment_name=experiment_name, output_dir=output_dir_path_initial_training
    )

    ## Setup the parameter definition for the models
    mean = 1.0  # mean value of the latent GMRF field before exp transform
    dimension = 1089  # number of dofs in the LF model (refinement level 5)
    L = 1.0  # length of the square domain
    num_obs = 2500  # number of observation points in experimental data
    x_vec = GaussianMarkovRandomField(
        mean,
        dimension,
        L,
        neighbor_mapping_path,
        field_dimension=1,
        spatial_dimension=2,
    )
    parameters = Parameters(x_vec=x_vec)

    ## Setup data processors, the velocity field is here additionally stored in a numpy file
    lf_data_processor = NumpyDataProc(
        file_name_identifier="bmfia_sol.npy",
        file_options_dict={"delete_field_data": False},
    )
    lf_features_data_processor = NumpyDataProc(
        file_name_identifier="bmfia_features.npy",
        file_options_dict={"delete_field_data": False},
    )
    lf_gradient_data_processor = NumpyDataProc(
        file_name_identifier="bmfia_grad_solution.npy",
        file_options_dict={"delete_field_data": False},
    )

    hf_data_processor = NumpyDataProc(
        file_name_identifier="bmfia_sol.npy",
        file_options_dict={"delete_field_data": False},
    )

    ## Setup drivers
    # lf forward model driver
    mpi_driver_lf = DealDriver(
        parameters,
        lf_input_file_template,
        lf_model_path,
        files_to_copy=None,
        data_processor=lf_data_processor,
        gradient_data_processor=None,  # only needed for automated differentiation, not for adjoint
        feature_data_processor=lf_features_data_processor,
        mpi_cmd="/usr/bin/mpirun --bind-to none",
    )
    # lf adjoint model driver (for gradients)
    mpi_driver_lf_gradient = DealDriver(
        parameters,
        lf_input_file_template,
        lf_model_path,
        files_to_copy=None,
        data_processor=lf_gradient_data_processor,
        gradient_data_processor=None,  # only needed for automated differentiation, not for adjoint
        feature_data_processor=None,
        mpi_cmd="/usr/bin/mpirun --bind-to none",
    )
    # hf forward model driver
    mpi_driver_hf = DealDriver(
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

    # build the schedulers and the rest of the QUEENS model in context
    # this makes sure that everything is closed properly at the end
    # here we actually run the QUEENS run / the initial training phase of BMFIA
    with global_settings_initial:
        ## Setup schedulers with local MPI scheduler; allow 6 jobs to run in parallel on one processor each
        # we use a custom scheduler that supports feature output
        # alternatively, one could use the standard Local scheduler from QUEENS
        # and instead write a custom npy_data_processor that also returns features and reads in multiple files
        local_scheduler_lf = LocalWFeatures(
            experiment_name,
            num_jobs=6,
            num_procs=1,
            restart_workers=False,
            verbose=True,
        )
        local_scheduler_hf = LocalWFeatures(
            experiment_name,
            num_jobs=6,
            num_procs=1,
            restart_workers=False,
            verbose=True,
        )
        print("schedulers set up")

        ## Setup the QUEENS simulation models
        ## As we will later need gradients of the LF model, we set it up as an adjoint model
        ## in the initial training phase, we do not need gradients of LF
        lf_model = AdjointSimulationModel(
            scheduler=local_scheduler_lf,
            driver=mpi_driver_lf,
            gradient_driver=mpi_driver_lf_gradient,
        )
        hf_model = SimulationModel(scheduler=local_scheduler_hf, driver=mpi_driver_hf)
        print("models set up")

        ## Setup the BMFIA iterator -> here we sample randomly from the conditional prior
        ## The prior being a GMRF, conditioned on a specific mean and precision factor
        # -------------------------------------------------------------------------------------------------------
        # ------------------------ BMFIA: initial training phase ------------------------------------------------
        # -------------------------------------------------------------------------------------------------------
        bmfia_iterator = BmfiaIterator(
            parameters,
            hf_model,
            lf_model,
            initial_design,
            num_obs,
            grid_interpolator=uniform_grid_interpolator,
            global_settings=global_settings_initial,
            path_trainings_data=path_trainings_data,
        )

        ## finally run the BMFIA iterator (the initial training phase) / start the QUEENS run
        # run_iterator(bmfia_iterator, global_settings=global_settings_initial)
        print("Finished initial training phase of BMFIA.")

        # -------------------------------------------------------------------------------------------------------
        # ------------------------ BMFIA: inference phase -------------------------------------------------------
        # -------------------------------------------------------------------------------------------------------
        # get experimental data
        csv_data_processor = CsvDataProc(
            file_name_identifier="observations.csv",
            file_options_dict={
                "header_row": 0,
                "index_column": False,
                "returned_filter_format": "dict",
                "filter": {"type": "entire_file"},
            },
        )

        experimental_data = csv_data_processor.get_data_from_file(
            Path("./data/").resolve()
        )
        coordinate_labels = ["x1", "x2"]
        output_labels = ["u_1", "u_2"]
        experimental_coordinates = np.array(
            [experimental_data[coordinate] for coordinate in coordinate_labels]
        ).T
        y_obs_vec = np.array(
            [experimental_data[output_label] for output_label in output_labels]
        ).T
        # we reshape the array such that it matches the the image that will
        # be handed to the CNN (here: 50 x 50 grid with 2 channels) in its order
        # this is currently a bit hacky an can be resolved in the future by
        # by e.g. storing geometric information in a graph structure
        y_obs_vec = y_obs_vec.reshape(-1, order="C")
        y_obs_vec = np.rot90(y_obs_vec.reshape(50, 50, 2, order="F"), k=1).reshape(
            -1, order="F"
        )

        # now we add artificial noise to the observations (before they were noise-free)
        mean_y_obs_var = np.sum(y_obs_vec**2, axis=0) / y_obs_vec.size
        np.random.seed(42)
        percent_value = 0.02  # 2 % NSR (noise to signal ratio)
        noise_var = percent_value * mean_y_obs_var
        epsilon_noise = np.random.normal(0, np.sqrt(noise_var), y_obs_vec.shape[0])
        y_obs_vec = y_obs_vec + epsilon_noise

        # setup the multi-fidelity conditional approximation
        mf_conditional_approx = GaussianCNN(
            num_epochs=5 #5000,
            batch_size=64,
            training_rate=0.005,
            optimizer_seed=42,
            verbosity_on=True,
            nugget_std=1.0e-4,
            loss_plot_path_dir=output_dir_path_inference,
            cnn_grid_input=[
                50,
                50,
                2,
            ],  # a 50 x 50 observation grid with 2 channels (u_1, u_2)
            latent_dim=100,
            num_validation_data=20,
            feature_list=[16, 32, 64],
        )

        # setup multi-fidelity likelihood (BMFIA likelihood) model
        initial_noise_var = 1.0  # dummy value that will be overwritten EM
        mf_likelihood = BMFGaussianModel(
            lf_model,
            path_trainings_data,
            experimental_coordinates,
            coordinate_labels,
            output_labels,
            y_obs_vec,
            mf_conditional_approx,
            noise_variance=initial_noise_var,
        )
        breakpoint()
