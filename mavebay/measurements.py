from typing import Optional

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from jax.numpy import DeviceArray


def nonlinear_f(ge_nonlinearity_type: Optional[str] = "nonlinear", x=None):
    """
    Set the nonlinearity function in the measurement process
    """
    if ge_nonlinearity_type == "nonlinear":
        return jnp.tanh(x)


def ge_measurement(
    D_H: int,
    phi: DeviceArray,
    ge_nonlinearity_type: Optional[str] = "nonlinear",
):
    # GE parameters
    a = numpyro.sample("a", dist.Normal(loc=0, scale=1))
    b = numpyro.sample("b", dist.Normal(jnp.zeros((1, D_H)), jnp.ones((D_H,))))
    c = numpyro.sample("c", dist.Normal(jnp.zeros((1, D_H)), jnp.ones((1, D_H))))
    d = numpyro.sample("d", dist.Normal(jnp.zeros((1, D_H)), jnp.ones((1, D_H))))

    # GE regression
    g = a + jnp.sum(b * nonlinear_f(ge_nonlinearity_type, c * phi + d), axis=1)
    g = g[..., jnp.newaxis]

    return g
