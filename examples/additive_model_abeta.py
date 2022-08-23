from dataclasses import dataclass

import numpyro

from mavebay import model, utils


# Define the fitting arguments
@dataclass
class fit_args:
    num_chains = 4
    num_samples = 10
    num_warmup = 10
    learning_rate = 1e-3
    device = "cpu"
    # if jax is installed with GPU support can be either cpu and gpu
    method = "svi"
    # svi: Stochastic variational inference
    # mcmc: MonteCarlo with NUTS
    progress_bar = True


# Set the device and number of chains (CPUs) in case of the mcmc inference
numpyro.set_platform(fit_args.device)
numpyro.set_host_device_count(fit_args.num_chains)

# Read the pandas dataframe of the datasets: MAVENN builtin dataset
x, y, L, C = utils.load_dataset(
    filename="./datasets/amyloid_data.csv.gz", alphabet="protein*"
)

# Define the Model
mavebay_model = model.Model(
    L,
    C,
    regression_type="GE",
    gpmap_type="additive",
    ge_hidden_nodes=20,
    ge_nonlinearity_type="nonlinear",
    ge_noise_model_type="Gaussian",
    seed=1234,
)

# Fit the model
svi_guide, svi_results = mavebay_model.fit(fit_args, x=x, y=y)

print(svi_results)
