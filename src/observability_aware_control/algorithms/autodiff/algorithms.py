import math

import jax
import jax.numpy as jnp
import functools

NX = 3
NU = 2


@functools.partial(jax.jit, static_argnames=("sys",))
def numsolve_sigma(sys, x0, u, dt):
    def _update(x_op, tup):
        u = tup[:-1]
        dt = tup[-1]
        return x_op + dt * sys.dynamics(x_op, u), x_op

    _, x = jax.lax.scan(_update, init=x0, xs=jnp.vstack([u, dt]).T)

    return x.T, sys.observation(x.T)


@functools.partial(jax.jit, static_argnames=("sys",))
@functools.partial(jax.vmap, in_axes=(None, 1, 1, None, None), out_axes=2)
def _perturb(sys, x0_plus, x0_minus, u, dts):
    _, yi_plus = numsolve_sigma(sys, x0_plus, u, dts)
    _, yi_minus = numsolve_sigma(sys, x0_minus, u, dts)
    return yi_plus - yi_minus


@functools.partial(jax.jit, static_argnames=("sys", "eps"))
def _numlog(sys, x0, u, dt, eps, perturb_axis):
    if perturb_axis is None:
        perturb_axis = jnp.arange(0, sys.nx)
    perturb_axis = jnp.asarray(perturb_axis)

    perturb_bases = jnp.eye(x0.size)[:, perturb_axis]
    x0_plus = x0[..., None] + eps * perturb_bases
    x0_minus = x0[..., None] - eps * perturb_bases
    y_all = _perturb(sys, x0_plus, x0_minus, u, dt) / (2.0 * eps)

    [xm, ym] = jnp.meshgrid(perturb_axis, perturb_axis)

    return jnp.sum(dt[:, None, None] * y_all[:, :, xm] * y_all[:, :, ym], axis=(0, 1))


def numlog(sys, x0, u, dt, eps, perturb_axis=None):
    x0 = jnp.asarray(x0)
    u = jnp.asarray(u)
    dt = jnp.asarray(dt)
    n_steps = dt.size
    if x0.shape != (sys.nx,):
        raise ValueError(f"Expected x0 with shape ({sys.nx}), got {x0.shape}")

    if u.shape != (sys.nu, n_steps):
        raise ValueError(
            f"Expected matrix with shape ({sys.nu - sys.NU}, {n_steps}), got {u.shape}"
        )
    if jnp.any(dt <= 0):
        raise ValueError("Discrete time-step is not positive.")
    return _numlog(sys, x0, u, dt, eps, perturb_axis)
