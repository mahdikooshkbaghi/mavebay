from typing import Optional

import jax
import jax.numpy as jnp
import jax.random as random
import numpy as np
import numpyro
import numpyro.distributions as dist
from jax.numpy import DeviceArray
from numpyro.infer import Predictive

from .entropy import mi_continuous
from .gpmaps import KOrderGPMap, additive_gp_map
from .infer import fit
from .measurements import ge_measurement, multi_head_measurement
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
    alphabet: (str)
        Alphabet used for one-hot encoding.
    D_Y: (int)
        Dimension of the Measurement
    regression_type: (str)
        Type of the regression.
        The possible choices are `GE` and `blackbox` for now.
    gpmap_type: (str)
        Type of the gpmap.
        The possible choices are 'additive' and 'kth_order' for now.
    gpmap_kwargs: dict()
        The GP-map keywords input as dictionary.
    hidden_nodes: (int)
        The number of hidden nodes in the `GE` regression or nodes in each layer
        in `blackbox` model.
    ge_noise_model_type: (str)
        Specifies the type of noise prior.
        The possible choice for now: 'Gaussian'.
    num_layers: (int)
        Number of hidden layer in `blackbox` measurement process.
    seed: (int)
        seed for random number generators.
    """

    def __init__(
        self,
        L: int,
        C: int,
        alphabet: str,
        D_Y: Optional[int] = 1,
        regression_type: Optional[str] = "GE",
        gpmap_type: Optional[str] = "additive",
        gpmap_kwargs: Optional[dict] = None,
        hidden_nodes: Optional[int] = 20,
        ge_nonlinearity_type: Optional[str] = "nonlinear",
        ge_noise_model_type: Optional[str] = "Gaussian",
        num_layers: Optional[int] = 2,
        seed: Optional[int] = 1234,
    ):

        # Assign the sequence length.
        self.L = L
        # Assign the alphabet length.
        self.C = C
        # Assign the alphabet
        self.alphabet = alphabet
        # Dimension of the measurement
        self.D_Y = D_Y
        # Assign the regression type
        self.regression_type = regression_type
        # Assign the gpmap type
        self.gpmap_type = gpmap_type
        # Assign the gpmap_kwargs
        self.gpmap_kwargs = gpmap_kwargs
        # Assign the number of nodes for the GE layer
        self.hidden_nodes = hidden_nodes
        # Assign the nonlinearity type of the GE
        self.ge_nonlinearity_type = ge_nonlinearity_type
        # Assign the ge noise model type.
        self.ge_noise_model_type = ge_noise_model_type
        # number of layers in blackbox measurement
        # it will not be used in ge measurement.
        self.num_layers = num_layers
        # Random seed
        self.seed = seed
        # Random number generator for jax. inference and prediction seeds
        self.rng_infer, self.rng_predict = random.split(random.PRNGKey(self.seed))
        np.random.seed(self.seed)

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
            g = ge_measurement(self.hidden_nodes, phi, self.ge_nonlinearity_type)
        if self.regression_type == "blackbox":
            g = multi_head_measurement(
                D_Y=self.D_Y, phi=phi, num_layers=self.num_layers, D_H=self.hidden_nodes
            )

        return g

    def model(
        self, x: DeviceArray = None, y: DeviceArray = None, phi: DeviceArray = None
    ):

        # Get the gp parameters
        if x is not None:
            self.gp_params = self.set_gp_params(x)
        if phi is None:
            phi = self.gp_params["phi"]
            self.phi = phi
        else:
            phi = phi[..., jnp.newaxis]
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

    def sample_params_posterior(self, num_samples: Optional[int] = 1000):
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
        num_samples: Optional[int] = 10_000,
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

        posterior_predictions = jax.jit(self.posterior_predictive)(
            self.rng_predict, x=x
        )
        yhat = jax.jit(summary)(posterior_predictions["yhat"], prob)
        phi = jax.jit(summary)(posterior_predictions["phi"], prob)
        return yhat, phi

    def phi_to_yhat(
        self,
        phi: DeviceArray = None,
        prob: Optional[float] = 0.95,
        num_samples: Optional[int] = 10000,
    ):
        """
        get the model prediction yhat for the given phi.
        good for plot the smooth measurement process.

        Parameters
        ----------
        phi: (jax.numpy.DeviceArray)
        """
        if self.fit_method == "svi":
            posterior_predictive = Predictive(
                model=self.model,
                guide=self.guide,
                params=self.svi_results.params,
                num_samples=num_samples,
            )
        elif self.fit_method == "mcmc":
            posterior_predictive = Predictive(
                model=self.model,
                posterior_samples=self.trace.get_samples(),
            )

        phi_yhat = jax.jit(posterior_predictive)(self.rng_predict, phi=phi)["yhat"]
        return jax.jit(summary)(samples=phi_yhat, prob=prob)

    def I_predictive(
        self,
        x: Optional[DeviceArray] = None,
        y: Optional[DeviceArray] = None,
        knn: Optional[int] = 5,
        knn_fuzz: Optional[float] = 0.01,
        uncertainty: Optional[bool] = False,
        num_subsamples: Optional[int] = 25,
        use_LNC: Optional[bool] = False,
        alpha_LNC: Optional[float] = 0.5,
        verbose: Optional[bool] = False,
    ):
        """
        Predictive Information.
        Parameters
        ----------
        x: (jax.numpy.DeviceArray)
            One-hot encoded input sequences
        y: (jax.numpy.DeviceArray)
            Array of measurements.
            For GE models, ``y`` must be a 1D array of ``N`` floats.
        knn: (int>0)
            Number of nearest neighbors to use in the entropy estimators from
            the NPEET package.
        knn_fuzz: (float)
            Amount of noise to add to ``phi`` values before passing them
            to the KNN estimators. Specifically, Gaussian noise with standard deviation
            ``knn_fuzz * np.std(phi)`` is added to ``phi`` values. This is a
            hack and is not ideal, but is needed to get the KNN estimates to
            behave well on real MAVE data.
        uncertainty: (bool)
            Whether to estimate the uncertainty in ``I_pred``.
            Substantially increases runtime if ``True``.
        num_subsamples: (int)
            Number of subsamples to use when estimating the uncertainty in
            ``I_pred``.
        use_LNC: (bool)
            Whether to use the Local Nonuniform Correction (LNC) of
            Gao et al., 2015 when computing ``I_pred`` for GE models.
            Substantially increases runtime set to ``True``.
        alpha_LNC: (float in (0,1))
            Value of ``alpha`` to use when computing the LNC correction.
            See Gao et al., 2015 for details. Used only for GE models.
        verbose: (bool)
            Whether to print results and execution time.
        Returns
        -------
        I_pred: (float)
            Estimated variational information, in bits.
        dI_pred: (float)
            Standard error for ``I_pred``. Is ``0`` if ``uncertainty=False``
            is used.
        """

        # Calculate the mean phi for given x
        posterior_predictions = jax.jit(self.posterior_predictive)(
            self.rng_predict, x=x
        )
        # TODO: do we need this even if we are doing fully bayesian inference?
        phi_mean = jnp.mean(posterior_predictions["phi"], axis=0).ravel()
        phi = phi_mean + knn_fuzz * phi_mean.std(ddof=1) * np.random.randn(
            len(phi_mean)
        )

        # Compute mi estimate
        return mi_continuous(
            phi,
            y,
            knn=knn,
            uncertainty=uncertainty,
            use_LNC=use_LNC,
            alpha_LNC=alpha_LNC,
            verbose=verbose,
        )
