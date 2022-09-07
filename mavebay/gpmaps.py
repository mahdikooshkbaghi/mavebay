import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist


def additive_gp_map(L, C, x_lc):
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

    theta_dict = {}
    theta_dict["theta_0"] = theta_0
    theta_dict["theta_lc"] = theta_lc

    return theta_dict, phi


def KOrderGPMap(L, C, x_lc, K=1):
    """
    Kth order GPmap
    """
    assert L >= K, f"Interaction order {K}, should be less than seq length {L}"

    # Initialize the GP params dictionary
    theta_dict = {}

    # Constant parameter: theta_0
    theta_dict["theta_0"] = numpyro.sample("theta_0", dist.Normal(loc=0, scale=1))
    theta_shape = (1,)
    seq_len_arrange = jnp.arange(L, dtype=int)
    mask_dict = {}

    # theta_0 contribution to the latent phenotype
    phi_val = theta_dict["theta_0"]

    # Loop over interaction order
    for k in range(K):
        theta_name = f"theta_{k+1}"
        theta_shape = theta_shape + (L, C)
        # Sample the Kth interaction theta values
        theta_dict[theta_name] = numpyro.sample(
            f"{theta_name}",
            dist.Normal(loc=jnp.zeros(theta_shape), scale=jnp.ones(theta_shape)),
        )

        # Compute the contribution of k-th order interaction to
        # the latent phenotype phi.

        # Create list of indices
        # which should be False or True based on level of interactions.
        # The ls_dict is analogous to l, l', l'', ... in MAVE-NN paper
        ls_dict = {}
        # Find the axis shape (order) which we should sum the array
        axis_shape = np.arange(1, len(theta_shape), dtype=int)
        # starting location of L,C characters in the shape lists
        lc_loc = 1
        x_mult = 1
        for w in range(k + 1):
            ls_part_shape = [1] * len(theta_shape)
            ls_tile_shape = list(theta_shape)
            ls_part_shape[lc_loc] = L
            ls_tile_shape[lc_loc] = 1
            ls = jnp.tile(seq_len_arrange.reshape(ls_part_shape), ls_tile_shape)
            ls_dict[f"ls_{w}"] = ls

            x_shape_k_order = [1] * len(theta_shape)
            x_shape_k_order[0] = -1
            x_shape_k_order[lc_loc] = L
            x_shape_k_order[lc_loc + 1] = C
            x_mult = x_mult * jnp.reshape(x_lc, x_shape_k_order)

            lc_loc = lc_loc + 2

        m_dict = {}
        for w in range(k):
            m_dict[f"m_{w}"] = ls_dict[f"ls_{w+1}"] > ls_dict[f"ls_{w}"]

        mask = True
        for key in m_dict.keys():
            mask = m_dict[key] * mask
        # Create mask array
        mask_dict[theta_name] = mask

        phi_val = phi_val + jnp.reshape(
            jnp.sum(
                theta_dict[theta_name] * mask_dict[theta_name] * x_mult, axis=axis_shape
            ),
            [-1, 1],
        )
    phi = numpyro.deterministic("phi", phi_val)

    return theta_dict, phi
