import math

import jax
import jax.numpy as jnp
import functools

NX = 3
NU = 2


@functools.partial(jax.jit, static_argnames=["sys"])
def numsolve_sigma(sys, x0, u, dts):
    x = [x0]

    for i, dt in enumerate(dts, 1):
        dx = sys.dynamics(x[-1], u[:, i - 1])
        x.append(x[-1] + dt * dx)

    x = jnp.array(x).T

    return x, sys.observation(x)


@functools.partial(jax.jit, static_argnames=("sys"))
@functools.partial(jax.vmap, in_axes=(None, 1, 1, None, None), out_axes=2)
def _perturb(sys, x0_plus, x0_minus, u, dts):
    _, yi_plus = numsolve_sigma(sys, x0_plus, u, dts)
    _, yi_minus = numsolve_sigma(sys, x0_minus, u, dts)
    return yi_plus - yi_minus


@functools.partial(jax.jit, static_argnames=("sys", "eps"))
def _numlog(sys, x0, u, dt, eps, perturb_axis):
    if perturb_axis is None:
        perturb_axis = jnp.arange(0, sys.nx)
    n_perturbed_x = len(perturb_axis)

    perturb_bases = jnp.eye(x0.size)[:, perturb_axis]
    x0_plus = x0[..., None] + eps * perturb_bases
    x0_minus = x0[..., None] - eps * perturb_bases
    y_all = _perturb(sys, x0_plus, x0_minus, u, dt[1:])

    gramian = jnp.zeros((n_perturbed_x, n_perturbed_x))
    for i in range(0, n_perturbed_x):
        for j in range(0, i + 1):
            gramian.at[i, j].set((dt * y_all[:, :, i] * y_all[:, :, j]).sum())
    gramian = jnp.tril(gramian, -1) + jnp.tril(gramian)

    return gramian


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
