# closed-form-discrete-diffusion-guidance-investigation

This repository contains the code used to run the experiments in the paper: `How
Reliable are Intermediate Rewards for SMC-based Guidance in Masked Discrete Diffusion
Models? A Closed-Form Analysis`. Code for running experiments can be found in `notebooks/experiments.py`.

### Getting Started

We used [uv](https://docs.astral.sh/uv/) for running experiments. In order to set up a virtual environment with all necessary dependencies installed, run the command:

```
uv sync --all-groups
```

We use [jupytext](https://jupytext.org/) for storing notebooks. In order to convert
`notebooks/experiments.py` to `.ipynb` run the command:

```
uv run jupytext --to ipynb notebooks/experiments.py
```
