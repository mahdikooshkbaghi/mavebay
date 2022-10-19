import argparse

import matplotlib.pyplot as plt
import numpy as np
import numpyro

import mavebay


def main(args):

    print(f"Reading {args.data} dataset from MAVENN repo\n")
    if args.data == "mpsa":
        filename = "https://github.com/jbkinney/mavenn/raw/master/mavenn/examples/datasets/mpsa_data.csv.gz"  # noqa
        alphabet = "rna"
    if args.data == "abeta":
        filename = "https://github.com/jbkinney/mavenn/raw/master/mavenn/examples/datasets/amyloid_data.csv.gz"  # noqa
        alphabet = "protein*"
    if args.data == "tdp43":
        filename = "https://github.com/jbkinney/mavenn/raw/master/mavenn/examples/datasets/tdp43_data.csv.gz"  # noqa
        alphabet = "protein*"
    # Get the dataset
    x, y, L, C, alphabet, cons_seq = mavebay.utils.load_dataset(
        filename=filename, alphabet=alphabet
    )
    # Define the GP map kwargs
    gpmap_kwargs = {}
    gpmap_kwargs["interaction_order"] = args.interaction_order
    # Define the Model
    model = mavebay.model.Model(
        L=L,
        C=C,
        alphabet=alphabet,
        gpmap_type="kth_order",
        gpmap_kwargs=gpmap_kwargs,
    )
    # fit the model
    model.fit(args, x=x, y=y)

    # Get the model posterior prediction mean and hdi for x sequences
    yhat, phi = model.ppc(num_samples=100, x=x, prob=0.95)

    # Variational Information calculation
    I_var, dI_var = model.I_predictive(x=x, y=y, uncertainty=True)

    # ELBO loss, measurements vs predictions, y-phi space plots
    fig, axs = plt.subplots(1, 4, figsize=(12, 3))
    ax = axs[0]
    ax.plot(model.svi_results.losses)
    ax.set_ylabel("ELBO loss")
    ax.set_xlabel("num SVI steps")

    ax = axs[1]
    ax.plot(y, yhat["mean"], "o", alpha=0.01)
    Rsq = np.corrcoef(y.ravel(), np.array(yhat["mean"]).ravel())[0, 1] ** 2
    ax.plot(y, y, c="r")
    ax.set_title(rf"$R^2$ = {Rsq:.3f}")
    ax.set_xlabel("y measurements")
    ax.set_ylabel("mean y predictions")

    ax = axs[2]
    ax.scatter(phi["mean"], yhat["mean"], alpha=0.01, s=1, label=r"$\hat{y}$")
    ax.scatter(phi["mean"], yhat["hdpi"][0], alpha=0.01, s=1, label=r"$\hat{y}$ 2.5%")
    ax.scatter(phi["mean"], yhat["hdpi"][1], alpha=0.01, s=1, label=r"$\hat{y}$ 97.5%")
    ax.legend(fontsize=8)
    ax.set_xlabel(r"mean $\phi$")  # noqa
    ax.set_ylabel(r"$y$ predictions")

    # # Get the smooth measurement process for range of phi
    phi_r = np.linspace(np.amin(phi["mean"]), np.amax(phi["mean"]), 1000)
    prob = 0.95
    phi_yhat = model.phi_to_yhat(phi=phi_r, prob=prob, num_samples=10_000)
    # Plot the smooth measurement process for range of phi
    ax = axs[3]
    ax.plot(phi_r, phi_yhat["mean"], label=r"$\hat{y}$")
    ax.fill_between(
        phi_r,
        phi_yhat["hdpi"][0].ravel(),
        phi_yhat["hdpi"][1].ravel(),
        interpolate=True,
        color="darkorange",
        alpha=0.5,
        label=f"{100*prob}% hdi",
    )
    ax.scatter(phi["mean"], y, s=1, c="k", label="data", alpha=0.1)
    ax.legend(fontsize=7)
    ax.set_xlabel(r"$\phi$")  # noqa
    ax.set_ylabel(r"$y$ predictions")
    ax.set_title(rf"I_var={I_var:.4f}$\pm${dI_var:4f}")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="parse args")

    # inference method
    parser.add_argument(
        "-m",
        "--method",
        default="svi",
        type=str,
        help="Inference method: (variational) svi or mcmc",
    )

    # GP map interaction order
    parser.add_argument(
        "-k",
        "--interaction_order",
        default=1,
        type=int,
        help="GP map interaction order",
    )

    # Number of samples
    parser.add_argument(
        "-n",
        "--num_samples",
        default=1000,
        type=int,
        help="Number of training epochs for SVI or MCMC samples.",
    )

    # Learning rate
    parser.add_argument(
        "-lr",
        "--learning_rate",
        default=1.0e-2,
        type=float,
        help="Learning rate for SVI inference optimizer",
    )

    # Device to run the inference
    parser.add_argument(
        "-d",
        "--device",
        default="cpu",
        type=str,
        help="Device to run the inference, cpu, gpu or tpu.",
    )

    # Initialization for SVI
    parser.add_argument(
        "-i",
        "--init_loc_fn",
        default="feasible",
        type=str,
        help="initialization for the SVI",
    )

    # Progress Bar
    parser.add_argument(
        "-p",
        "--progress_bar",
        default=True,
        type=bool,
        help="Device to run the inference",
    )

    # Dataset to read
    parser.add_argument(
        "-ds",
        "--data",
        default="abeta",
        type=str,
        help="Dataset to read from online MAVENN repo",
    )

    args = parser.parse_args()

    # Set device for numpyro
    numpyro.set_platform(args.device)

    # Set numpyro number of cpus in case of MCMC
    if hasattr(args, "num_chains"):
        numpyro.set_host_device_count(args.num_chains)
    main(args)
