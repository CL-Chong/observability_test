import jax
import jax.numpy as jnp

from . import common


def numlog(sys, x0, u, dt, eps, perturb_axis=None):
    if perturb_axis is None:
        perturb_axis = jnp.arange(0, sys.nx)
    perturb_axis = jnp.asarray(perturb_axis)

    observation = jax.vmap(sys.observation)

    dt = jnp.broadcast_to(dt, u.shape[0])

    def _perturb(x0_plus, x0_minus):
        yi_plus = observation(common.forward_dynamics(sys.dynamics, x0_plus, u, dt))
        yi_minus = observation(common.forward_dynamics(sys.dynamics, x0_minus, u, dt))
        return yi_plus - yi_minus

    perturb_bases = jnp.eye(x0.size)[perturb_axis]
    x0_plus = x0 + eps * perturb_bases
    x0_minus = x0 - eps * perturb_bases
    y_all = jax.vmap(_perturb, out_axes=2)(x0_plus, x0_minus) / (2.0 * eps)

    coord_vec = jnp.arange(0, perturb_axis.size)
    xm, ym = jnp.meshgrid(coord_vec, coord_vec)

    dt = dt[..., None, None, None]

    return jnp.sum(dt * y_all[:, :, xm] * y_all[:, :, ym], axis=(0, 1))
