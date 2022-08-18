import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist


def additive_gp_map(
    L,
    C,
    x_lc,
):
    # Additive parameters
    # Normal prior on theta_0
    theta_0 = numpyro.sample("theta_0", dist.Normal(loc=0, scale=1))
    # Prior on the theta_lc
    theta_lc = numpyro.sample(
        "theta_lc", dist.Normal(loc=jnp.zeros((L, C)), scale=jnp.ones((L, C)))
    )
    phi = numpyro.deterministic(
        "phi", theta_0 + jnp.einsum("ij,kij->k", theta_lc, x_lc)
    )
    phi = phi[..., jnp.newaxis]

    return theta_0, theta_lc, phi
