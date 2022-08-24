from dataclasses import dataclass

import numpy as np
import numpyro

import mavebay


# Fitting method and parameters
@dataclass
class fit_args:
    num_samples = 10_000
    learning_rate = 1e-2
    device = "cpu"
    progress_bar = True
    method = "svi"


# Set numpyro number of cpus and devices
if hasattr(fit_args, "num_chains"):
    numpyro.set_host_device_count(fit_args.num_chains)
if hasattr(fit_args, "device"):
    numpyro.set_platform(fit_args.device)
else:
    numpyro.set_platform("cpu")


# Get the dataset
x, y, L, C, alphabet, cons_seq = mavebay.utils.load_dataset(
    filename="./datasets/amyloid_data.csv.gz", alphabet="protein*"
)

# Define the Model
model = mavebay.model.Model(L=L, C=C, alphabet="protein*")

# fit the model
model.fit(fit_args, x=x, y=y)

# Get the model posterior prediction mean and hdi for x sequences
yhat, phi = model.ppc(num_samples=100, x=x, prob=0.95)

# Get the smooth measurement process for range of phi
phi_r = np.linspace(-10, 10, 1000)
prob = 0.95
phi_yhat = model.phi_to_yhat(phi=phi_r, prob=prob)
