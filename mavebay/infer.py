import time

import numpyro.optim as optim

# jax imports
from jax.numpy import DeviceArray
from numpyro.infer import MCMC, NUTS, SVI, Trace_ELBO, autoguide, init_to_sample


class fit:
    def __init__(self, rng_key, args=None, model: DeviceArray = None):

        default_dict = {
            "method": "svi",
            "num_samples": 1000,
            "learning_rate": 1e-2,
            "progress_bar": True,
        }

        self.rng_key = rng_key
        self.model = model

        # Assign the inference method
        if hasattr(args, "method"):
            self.method = args.method
        else:
            self.method = default_dict["method"]
            print("User did not provide the inference method in fitting args")
            print(f"Default method of inference is set to {self.method}")

        # Assign number of samples for inference.
        if hasattr(args, "num_samples"):
            self.num_samples = args.num_samples
        else:
            self.num_samples = default_dict["num_samples"]
            print("User did not provide num_samples in fitting args")
            print(f"Default Number of samples is set to {self.num_samples}")

        # Assign the learning rate for SVI
        if self.method == "svi":
            if hasattr(args, "learning_rate"):
                self.learning_rate = args.learning_rate
            else:
                self.learning_rate = default_dict["learning_rate"]
                print("User did not provide learning rate for SVI in fitting args")
                print(f"Default learning rate is set to {self.learning_rate}")

        # Progress bar. For large data, gpu, multiple chains sometimes bar make it slow
        if hasattr(args, "progress_bar"):
            self.progress_bar = args.progress_bar
        else:
            self.progress_bar = default_dict["progress_bar"]
            print("User did not provide progress_bar status in fitting args")
            print(f"Default progress bar is set to {self.progress_bar}")

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

        print("\nTraining using Stochastic Variational Inference")
        print("Training parameters are:")
        print(f"Learning Rate: {self.learning_rate}")
        print(f"Number of SVI steps: {self.num_samples}")
        print(f"Progress bar: {self.progress_bar}\n")

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
