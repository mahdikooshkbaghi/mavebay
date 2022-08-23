import time

import numpyro.optim as optim

# jax imports
from jax.numpy import DeviceArray
from numpyro.infer import MCMC, NUTS, SVI, Trace_ELBO, autoguide, init_to_sample


class fit:
    def __init__(self, args, rng_key, model: DeviceArray = None):

        self.rng_key = rng_key
        self.model = model

        error_samp = (
            "number of samples for mcmc/svi inference should be provided as args"
        )
        # Assign number of samples for inference.
        assert args.num_samples is not None, error_samp
        self.num_samples = args.num_samples

        # Assign parameters for the MCMC inference
        if args.method == "mcmc":
            error_warm = (
                "number of warmup steps for mcmc inference should be provided as args"
            )
            error_chain = (
                "number of chains for mcmc inference should be provided as args"
            )

            # Assign number of warmup steps for MCMC inference.
            assert args.num_warmup is not None, error_warm
            self.num_warmup = args.num_warmup

            # Assign number of chains for MCMC inference.
            assert args.num_chains is not None, error_chain
            self.num_chains = args.num_chains

        # Assign parameters for the SVI inference.
        if args.method == "svi":
            # Assign the learning rate
            learning_rate = getattr(args, "learning_rate", None)
            if learning_rate is None:
                self.learning_rate = 1e-1
                print("\nDefault Learning rate = 1e-1 is used")
            else:
                self.learning_rate = learning_rate

        # Progress bar. For large data, gpu, multiple chains sometimes bar make it slow
        progress_bar = getattr(args, "progress_bar", None)
        if progress_bar is None:
            self.progress_bar = True
        else:
            self.progress_bar = args.progress_bar

    def svi(self, x: DeviceArray, y: DeviceArray):
        """
        Stochastic Variational Inference.

        Parameters
        ----------
        x: (DeviceArray):
            Input features for training.
        y: (DeviceArray):
            Output (measurements) for training.

        Returns
        -------
        guide: (numpyro.infer.autoguide.AutoDelta):
            Numpyro Automatic Guide Generated.

        svi_results: (SVIRunResult)
            Stochastic Variational Inference Result.
        """

        print("\nTraining using Stochastic Variational Inference\n")
        start = time.time()

        # Assign AutoNormal guide.
        guide = autoguide.AutoNormal(self.model, init_loc_fn=init_to_sample)

        # Define the optimizer
        optimizer = optim.Adam(self.learning_rate)

        # Loss function is the ELBO
        svi = SVI(self.model, guide, optimizer, loss=Trace_ELBO())

        svi_results = svi.run(
            rng_key=self.rng_key,
            num_steps=self.num_samples,
            x=x,
            y=y,
            progress_bar=self.progress_bar,
        )

        print("\nVariational inference elapsed time:", time.time() - start)
        return guide, svi_results

    def mcmc(self, x, y):
        print("\nTraining using MCMC\n")
        start = time.time()
        # define kernel
        kernel = NUTS(model=self.model)
        # setup mcmc
        mcmc = MCMC(
            kernel,
            num_warmup=self.num_warmup,
            num_samples=self.num_samples,
            num_chains=self.num_chains,
            progress_bar=self.progress_bar,
        )
        # run mcmc inference
        mcmc.run(self.rng_key, x=x, y=y)
        print("\nMCMC elapsed time:", time.time() - start)
        return mcmc
