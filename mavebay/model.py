from typing import Optional

import jax.numpy as jnp
import jax.random as random
import numpyro
import numpyro.distributions as dist
from jax.numpy import DeviceArray

from .gpmaps import additive_gp_map
from .measurements import ge_measurement
from .infer import fit


class Model:
    """
    Represents a numpyro model instance.
    Parameters
    ----------
    L: (int)
        Length of each training sequence. Must be ``>= 1``.
    C: (int)
        Length of the alphabet in the sequence.
    regression_type: (str)
        Type of the regression.
        The possible choice is 'GE' for now.
    gpmap_type: (str)
        Type of the gpmap.
        The possible choice is 'additive' for now.
    gpmap_kwargs: dict():
        The GP-map keywords input as dictionary.
    ge_noise_model_type: (str)
        Specifies the type of global epistasis noise model.
        The possible choice for now: 'Gaussian'
    """

    def __init__(
        self,
        L: int,
        C: int,
        regression_type: Optional[str] = "GE",
        gpmap_type: Optional[str] = "additive",
        gpmap_kwargs: Optional[dict] = None,
        ge_hidden_nodes: Optional[int] = 20,
        ge_nonlinearity_type: Optional[str] = "nonlinear",
        ge_noise_model_type: Optional[str] = "Gaussian",
        seed: Optional[int] = 1234,
        inference_method: Optional[str] = "svi",
    ):

        # Assign the sequence length.
        self.L = L
        # Assign the alphabet length.
        self.C = C
        # Assign the regression type
        self.regression_type = regression_type
        # Assign the gpmap type
        self.gpmap_type = gpmap_type
        # Assign the gpmap_kwargs
        self.gpmap_kwargs = gpmap_kwargs
        # Assign the number of nodes for the GE layer
        self.ge_hidden_nodes = ge_hidden_nodes
        # Assign the nonlinearity type of the GE
        self.ge_nonlinearity_type = ge_nonlinearity_type
        # Assign the ge noise model type.
        self.ge_noise_model_type = ge_noise_model_type
        # Random seed
        self.seed = seed
        # Inference Method
        self.inference_method = inference_method

    def set_gp_params(self, x):
        """
        Set the GP map parameters.
        """
        if self.gpmap_type == "additive":
            theta_0, theta_lc, phi = additive_gp_map(self.L, self.C, x)
        return theta_0, theta_lc, phi

    def set_ge_params(self):
        if self.regression_type == "GE":
            g = ge_measurement(
                self.ge_hidden_nodes, self.phi, self.ge_nonlinearity_type
            )
        return g

    def model(self, x: DeviceArray = None, y: DeviceArray = None):
        self.theta_0, self.theta_lc, self.phi = self.set_gp_params(x)
        if y is not None:
            assert (
                self.phi.shape == y.shape
            ), f"phi has shape {self.phi.shape}, y has shape {y.shape}"
        self.g = self.set_ge_params()

        noise = numpyro.sample("noise", dist.Gamma(3.0, 1.0))
        # self.alpha, self.beta, noise = self.noise_model(self.ge_noise_model_type)
        sigma_obs = 1.0 / jnp.sqrt(noise)
        return numpyro.sample("yhat", dist.Normal(self.g, sigma_obs).to_event(1), obs=y)

    def fit(
        self, fit_args, x: DeviceArray, y: DeviceArray, rng_key: Optional[int] = None
    ):
        rng_key, _ = random.split(random.PRNGKey(self.seed))
        if self.inference_method == "svi":
            self.guide, self.svi_results = fit(
                rng_key=rng_key, args=fit_args, model=self.model
            ).svi(x=x, y=y)
            return self.guide, self.svi_results
        # if args.method == "mcmc":
        # self.trace = fit(args=args, rng_key=rng_key, model=self.model).mcmc(
        # x=x, y=y
        # )
        # return self.trace
