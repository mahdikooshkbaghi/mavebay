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
    if ge_nonlinearity_type == "linear":
        return x


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


def multi_head_measurement(
    D_Y: int,
    phi: DeviceArray,
    num_layers: Optional[int] = 2,
    D_H: Optional[int] = 10,
):
    """
    Represent blackbox measurement process from latent phenotype to the observation.

    Parameters
    ----------
    D_Y (int):
        The dimension of the observation.
    phi (jax.numpy.DeviceArray):
        The latent phenotype.
    num_layers (int):
        Number of hidden layers in neural network. Default = 2.
    D_H (int):
        Number of hidden nodes in the neural network layer. Default = 10.

    Returns
    -------
    g: (jax.numpy.DeviceArray)
        p(yhat|phi): the model prediction without noise.
    """

    # Find the dimension of the latent phenotype
    D_phi = phi.shape[1]
    D_X = D_phi

    # Dictionary for the measurement process neural network weights.
    w_dict = {}

    # neural network layers
    x = phi
    for i in range(num_layers - 1):
        w_name = f"w_{i}"
        w_dict[w_name] = numpyro.sample(
            w_name, dist.Normal(jnp.zeros((D_X, D_H)), jnp.ones((D_X, D_H)))
        )
        z = nonlinear_f("nonlinear", jnp.matmul(x, w_dict[w_name]))
        x = z
        D_X = x.shape[1]

    # last layer
    w_name = f"w_{num_layers}"
    w_dict[w_name] = numpyro.sample(
        w_name, dist.Normal(jnp.zeros((D_X, D_Y)), jnp.ones((D_X, D_Y)))
    )
    # last layer has linear activation function
    g = jnp.matmul(x, w_dict[w_name])

    return g


# def mpa_measurement(Y, K, phi):
#     """
#     Y (int):
#         number of bins.
#     K
#     """

#     # MPA parameters
#     a = numpyro.sample("a", dist.Normal(jnp.zeros((Y, K)), jnp.ones((Y, K))))
#     b = numpyro.sample("b", dist.Normal(jnp.zeros((Y, K)), jnp.ones((Y, K))))
#     c = numpyro.sample("c", dist.Normal(jnp.zeros((Y, K)), jnp.ones((Y, K))))
#     d = numpyro.sample("d", dist.Normal(jnp.zeros((Y, K)), jnp.ones((Y, K))))

#     ct_my = inputs[:, 1:]
#     # Compute p(y|phi)
#     # Compute weights
#     psi_my = a + jnp.sum(b * tanh(c_yk * phi + d_yk), axis=2)
#     psi_my = tf.reshape(psi_my, [-1, self.Y])
#     w_my = Exp(psi_my)

#         # Compute and return distribution
#         p_my = w_my / tf.reshape(K.sum(w_my, axis=1), [-1, 1])


#     p_my = self.p_of_all_y_given_phi(phi)
#     # Compute negative log likelihood
#     negative_log_likelihood = -K.sum(ct_my * jnp.log(p_my), axis=1)
