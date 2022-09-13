from typing import Optional

import jax.numpy as jnp
import jax.random as random
import numpyro
import numpyro.distributions as dist
from jax.numpy import DeviceArray
from numpyro.infer import Predictive

from .gpmaps import KOrderGPMap, additive_gp_map
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
        Specifies the type of noise prior.
        The possible choice for now: 'Gaussian'
    """

    def __init__(
        self,
        L: int,
        C: int,
        alphabet: str,
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
        # Assign the alphabet
        self.alphabet = alphabet
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
            theta_dict, phi = additive_gp_map(self.L, self.C, x)
        if self.gpmap_type == "kth_order":
            theta_dict, phi = KOrderGPMap(self.L, self.C, x, **self.gpmap_kwargs)

        gp_params["theta_dict"] = theta_dict
        gp_params["phi"] = phi

        return gp_params

    def set_mp_params(self, phi):
        """
        Set measurement process parameters.
        Returns
        ----------
        g: (jax.numpy.DeviceArray):
            The noiseless predictions of the model. g=MP(phi)
        """
        if self.regression_type == "GE":
            g = ge_measurement(self.ge_hidden_nodes, phi, self.ge_nonlinearity_type)
        return g

    def model(
        self,
        x: DeviceArray = None,
        y: DeviceArray = None,
        phi: DeviceArray = None,
        batch_size: int = None,
    ):

        # Get the gp parameters
        if x is not None:
            self.gp_params = self.set_gp_params(x)
        if phi is None:
            phi = self.gp_params["phi"]
            self.phi = phi
        else:
            phi = phi[..., jnp.newaxis]
        if y is not None:
            assert (
                self.phi.shape == y.shape
            ), f"phi has shape {self.phi.shape}, y has shape {y.shape}"
        g = self.set_mp_params(phi)

        if y is not None:
            assert g.shape == y.shape, f"g has shape {g.shape}, y has shape {y.shape}"

        if self.ge_noise_model_type == "Gaussian":
            self.sigma = numpyro.sample("sigma", dist.HalfNormal())

        return numpyro.sample("yhat", dist.Normal(g, self.sigma).to_event(1), obs=y)

    def fit(self, fit_args=None, x: DeviceArray = None, y: DeviceArray = None):
        # Assign the fitting method to the model
        if fit_args is None:
            # Default method is stochastic variational inference
            self.fit_method = "svi"
        else:
            if hasattr(fit_args, "method"):
                self.fit_method = fit_args.method
            else:
                self.fit_method = "svi"
        # Variational inference
        if self.fit_method == "svi":
            self.guide, self.svi_results = fit(
                rng_key=self.rng_infer, args=fit_args, model=self.model
            ).svi(x=x, y=y)
        # MCMC with NUTS inference
        if self.fit_method == "mcmc":
            self.trace = fit(
                args=fit_args, rng_key=self.rng_infer, model=self.model
            ).mcmc(x=x, y=y)

    def sample_posterior(self, num_samples: Optional[int] = 1000):
        """
        Sample from parameters posteriors.
        Assign the posterior attribute to the model.
        Parameters
        ----------
        num_samples: Optional (int)
            number of samples draws from posterior for the infer parameters.
            Default = 1000
        """
        if self.fit_method == "svi":
            self.posteriors = self.guide.sample_posterior(
                self.rng_predict, self.svi_results.params, sample_shape=(num_samples,)
            )
        if self.fit_method == "mcmc":
            self.posteriors = Predictive(
                model=self.model,
                posterior_samples=self.trace.get_samples(),
            ).posterior_samples

    def ppc(
        self,
        x: DeviceArray = None,
        num_samples: Optional[int] = 1000,
        prob: Optional[float] = 0.95,
    ):
        """
        Sample from predictions posteriors.

        Parameters
        ----------
        x: (jax.numpy.DeviceArray)
            the one-hot encoded sequences
        num_samples: Optional (int)
            number of samples draws from posterior for the prediction.
            Default = 1000
        prob: (float)
            the confidence interval probability.
            Default = 0.95: 95% hdpi.
        Returns
        ----------
        yhat: (jax.numpy.DeviceArray)
            The mean and hdpi model predictions for the measurements for input x.
        phi: (jax.numpy.DeviceArray)
            The mean and hdpi of latent phenotype values for input x.
        """
        if self.fit_method == "svi":
            self.posterior_predictive = Predictive(
                model=self.model,
                guide=self.guide,
                params=self.svi_results.params,
                num_samples=num_samples,
            )
        elif self.fit_method == "mcmc":
            self.posterior_predictive = Predictive(
                model=self.model,
                posterior_samples=self.trace.get_samples(),
            )

        posterior_predictions = self.posterior_predictive(self.rng_predict, x=x)
        yhat = summary(posterior_predictions["yhat"], prob)
        phi = summary(posterior_predictions["phi"], prob)
        return yhat, phi

    def phi_to_yhat(self, phi: DeviceArray = None, prob: float = 0.95):
        """
        get the model prediction yhat for the given phi.
        good for plot the smooth measurement process.

        Parameters
        ----------
        phi: (jax.numpy.DeviceArray)
        """

        return summary(
            self.posterior_predictive(self.rng_predict, phi=phi)["yhat"], prob
        )
