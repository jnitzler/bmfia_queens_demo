# BMFIA demonstration with QUEENS

This demonstration shows the setup of BMFIA (Bayesian multi-fidelity inverse analysis) for the first
example in the [accompanying paper on BMFIA](https://arxiv.org/abs/2505.24708):
>Nitzler, J., Temür, B. Z., Koutsourelakis, P. S., & Wall, W. A. (2025). Efficient Bayesian multi-fidelity inverse analysis for expensive and non-differentiable physics-based simulations in high stochastic dimensions. arXiv preprint arXiv:2505.24708.

The example is a 2D porous media flow problem, where the goal is to infer the log-permeability field of a porous medium
given sparse and noisy measurements of the velocity field. We use our open-source library [QUEENS](https://github.com/queens-py/queens) to handle the simulation management and inference. The computational models of the porous media flow are efficiently implemented in [Deal.II](https://dealii.org/) and are freely available in [this GitHub repository](https://github.com/jnitzler/porous_media_flow_bmfia). 

QUEENS is an open-source Python framework for solver-independent analyses of large scale computational models. It provides a unified interface to various solvers and enables efficient management of simulations, data handling, and advanced inference techniques such as BMFIA. Please also see our [paper on QUEENS](https://www.arxiv.org/abs/2508.16316).

## Setup and installation

1. Get the computational models for the porous media flow from [this GitHub repository](https://github.com/jnitzler/porous_media_flow_bmfia) and follow the setup and compilation instructions.
2. Install and setup QUEENS by following the instructions in the [QUEENS GitHub repository](https://github.com/queens-py/queens) then activate the QUEENS environment. This demonstration was set up and tested with `commit aff4b70184525af0f22355146ade6f1e9a401b3d` on the main branch of QUEENS.
3. Clone this repository to your local machine (with the active QUEENS environment `queens`)

## Run the demonstration
You can run the main script via:

```bash
python bmfia_queens_demo.py
```

## Notes on the general idea and setup of this repository
The BMFIA implementation is in this demonstration is not a finished concept but will be improved in the future. Currently, the `bmfia` directory contains novel and custom implementations that were not yet present in QUEENS or were at the time faster implemented in a little specialized class or method for this example. The idea is, however, to get rid of such custom implementations and make them instead available in a more generalized manner in QUEENS directly. While this transition might take some time, we allow for this compromise solution, meanwhile.
In the future the code should become easier to use and adaptable to other problems.

We specifically want to point out the following aspects:
- custom `adjoint_bmfia` model: we communicate the partial derivatives of the adjoint via binary Numpy files for efficiency. Reading and writing these files is currently handled in a custom `adjoint_bmfia` model class. In the future, this should be replaced by a more general QUEENS functionality.
- `local_scheduler_w_features`: BMFIA requires the input features $X$ within $Z_{\mathrm{LF}}$. These are currently additionally written out by our Deal.II model and read in by a custom scheduler class, besides the usual model output. Alternatively we could have created a custom `data_processor` but we found it more intuitive to handle this in the scheduler. In the future, this should be replaced by a more general QUEENS functionality.
- `GaussianCNN` and partially incomprehensible reshaping and rotation of model output data: The Gaussian CNN that is used to approximate the conditional multi-fidelity density contains currently several array reshaping and rotation operations that are rather specific for the 2D porous media flow example. We are aware that this is currently not ideal and will refactor this in the future and on the long run switch to graph-based geometry representations that avoid such operations. At the moment, the reshaping ensures that the observation data fits to the arrays as written out by the Deal.II model and to the reconstruction of the fields for the CNN. The CNN itself is currently also rather specifically designed for 2D spatial data and will be generalized in the future. A more modern implementation in PyTorch is also planned.
- `uniform_grid_interpolator`: This is a quick and rather custom way to interpolate the solution fields of the LF and HF models to the HF grid. In the future we will use standard formats such as VTK and use established libraries for interpolation. Here we wanted to avoid additional dependencies and keep the setup simple (the interpolation requires the coordinates of the HF grid points, which are currently read in from a Numpy file and were once written out by the Deal.II model).
- `gaussian_markov_random_field`: This implementation of a Gaussian Markov random field is currently rather limited and depends on neighborhood information of DoFs that is read in from a Numpy file. We are aware that this is only a quick solution for this simple example. In the future it would be advisable to implement such random fields directly in the adjoint model and add its gradient to the adjoint computation. This way we could access finite element information directly and, e.g., use the Laplace matrix of the mesh for the covariance structure. This would also avoid the need to read in neighborhood information from a file.
- In the paper we used further visualization steps and Deal.II post-processing to generate the posterior plots. Here we currently only write the data out to a `pickle` file. We will add a matplotlib-based visualization in the near future.