# BMFIA demonstration with QUEENS

This demonstration shows the setup of BMFIA (Bayesian multi-fidelity inverse analysis) for the first
example in the [accompanying paper on BMFIA](https://arxiv.org/abs/2505.24708):
>Nitzler, J., Temür, B. Z., Koutsourelakis, P. S., & Wall, W. A. (2025). Efficient Bayesian multi-fidelity inverse analysis for expensive and non-differentiable physics-based simulations in high stochastic dimensions. arXiv preprint arXiv:2505.24708.

The example is a 2D porous media flow problem, where the goal is to infer the log-permeability field of a porous medium
given sparse and noisy measurements of the velocity field. We use our open-source library [QUEENS](https://github.com/queens-py/queens) to handle the simulation management and inference. The computational models of the porous media flow are efficiently implemented in [Deal.II](https://dealii.org/) and are freely available in [this GitHub repository](https://github.com/jnitzler/porous_media_flow_bmfia). 

QUEENS is an open-source Python framework for solver-independent analyses of large scale computational models. It provides a unified interface to various solvers and enables efficient management of simulations, data handling, and advanced inference techniques such as BMFIA. Please also see our [paper on QUEENS](https://www.arxiv.org/abs/2508.16316).

## Setup and installation

1. Get the computational models for the porous media flow from [this GitHub repository](https://github.com/jnitzler/porous_media_flow_bmfia) and follow the setup and compilation instructions.
2. Install and setup QUEENS by following the instructions in the [QUEENS GitHub repository](https://github.com/queens-py/queens) then activate the QUEENS environment.
3. Clone this repository to your local machine (with the active QUEENS environment `queens`) and run:

    ```bash
    pip install jupyter ipykernel
    ``` 

Afterwards, select the `queens` environment as your Jupyter kernel.

## Run the demonstration / the jupyter notebook
