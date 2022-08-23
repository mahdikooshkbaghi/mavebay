from typing import Optional

import jax.random as random
import numpyro
import numpyro.distributions as dist
from jax.numpy import DeviceArray
from numpyro.infer import Predictive

from .gpmaps import additive_gp_map
from .infer import fit
from .measurements import ge_measurement
from .utils import summary


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
        # Random number generator for jax. inference and prediction seeds
        self.rng_infer, self.rng_predict = random.split(random.PRNGKey(self.seed))

    def set_gp_params(self, x: Optional[DeviceArray] = None):
        """
        Set the GP map parameters.
        Parameters
        ----------
        x: (jax.numpy.DeviceArray)
            One-hot encoded input sequences
        """
        gp_params = {}
        if self.gpmap_type == "additive":
            theta_0, theta_lc, phi = additive_gp_map(self.L, self.C, x)
            gp_params["theta_0"] = theta_0
            gp_params["theta_lc"] = theta_lc
            gp_params["phi"] = phi

        return gp_params

    def set_mp_params(self):
        """
        Set measurement process parameters.
        Returns
        ----------
        g: (jax.numpy.DeviceArray):
            The noiseless predictions of the model. g=MP(phi)
        """
        if self.regression_type == "GE":
            g = ge_measurement(
                self.ge_hidden_nodes, self.phi, self.ge_nonlinearity_type
            )
        return g

    def model(self, x: DeviceArray = None, y: DeviceArray = None):

        # Get the gp parameters
        self.gp_params = self.set_gp_params(x)
        self.phi = self.gp_params["phi"]
        if y is not None:
            assert (
                self.phi.shape == y.shape
            ), f"phi has shape {self.phi.shape}, y has shape {y.shape}"
        self.g = self.set_mp_params()
        self.sigma = numpyro.sample("sigma", dist.HalfNormal())

        return numpyro.sample(
            "yhat", dist.Normal(self.g, self.sigma).to_event(1), obs=y
        )

    def fit(self, fit_args, x: DeviceArray, y: DeviceArray):
        # Assign the fitting method to the model
        self.fit_method = fit_args.method
        # Variational inference
        if self.fit_method == "svi":
            self.guide, self.svi_results = fit(
                rng_key=self.rng_infer, args=fit_args, model=self.model
            ).svi(x=x, y=y)
        # MCMC with NUTS inference
        if fit_args.method == "mcmc":
            self.trace = fit(
                args=fit_args, rng_key=self.rng_infer, model=self.model
            ).mcmc(x=x, y=y)

    def sample_posterior(self, num_samples: Optional[int] = 1000):
        """
        Sample from posterior.
        """
        if self.fit_method == "svi":
            self.posteriors = self.guide.sample_posterior(
                self.rng_predict, self.svi_results.params, sample_shape=(num_samples,)
            )

    def ppc(
        self,
        num_samples: Optional[int] = 1000,
        x: DeviceArray = None,
        prob: Optional[float] = 0.95,
    ):
        """
        Assign the posterior predictive object to the model
        """
        self.posterior_predictive = Predictive(
            model=self.model,
            guide=self.guide,
            params=self.svi_results.params,
            num_samples=num_samples,
        )

        posterior_predictions = self.posterior_predictive(self.rng_predict, x=x)
        yhat = summary(posterior_predictions["yhat"], prob)
        return yhat
