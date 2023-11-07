import functools

import jax
import jax.numpy as jnp

NX = 3
NU = 2


@functools.partial(jax.jit, static_argnames=("sys", "without_observation"))
def numsolve_sigma(sys, x0, u, dt, f_args=None, h_args=None, without_observation=False):
    """Run forward simulation of a dynamical system

    Details
    -------
    This function enables time-varying parameters (known beforehand) to be passed in,
    whereas constant parameters are intended to be put into the sys object

    Parameters
    ----------
    sys : ModelBase
        An object satisfying the ModelBase Interface
    x0 : ArrayLike
        Initial state
    u : ArrayLike
        A sys.nu-by-len(dt) array of control inputs
    dt : ArrayLike
        An array of time steps
    f_args : ArrayLike, optional
        Additional (time-dependent) vector-valued inputs for sys.dynamics specified by a
        *-by-len(dt) array, by default None
    h_args : ArrayLike, optional
        Additional (time-dependent) vector-valued inputs for sys.observation specified
        by a *-by-len(dt) array, by default None
    without_observation: bool, optional
        If true, the observation equation will not be run on the states, and only the
        states trajectory will be returned

    Returns
    -------
    Tuple[jnp.array, jnp.array] | jnp.array
        The state and optionally observation trajectory
    """

    def _update(x_op, tup):
        u = tup[: sys.nu]
        dt = tup[sys.nu]
        if f_args is not None:
            param = tup[sys.nu + 1 :]
            dx = sys.dynamics(x_op, u, param)
        else:
            dx = sys.dynamics(x_op, u)
        return x_op + dt * dx, x_op

    if f_args is None:
        xs = jnp.vstack([u, dt]).T
    else:
        f_args = jnp.asarray(f_args)
        xs = jnp.vstack([u, dt, f_args]).T

    _, x = jax.lax.scan(_update, init=x0, xs=xs)

    if without_observation:
        return x.T

    if h_args is None:
        y = sys.observation(x.T)
    else:
        h_args = jnp.asarray(h_args)
        y = sys.observation(x.T, h_args)

    return x.T, y


@functools.partial(jax.jit, static_argnames=("sys",))
@functools.partial(jax.vmap, in_axes=(None, 1, 1, None, None, None, None), out_axes=2)
def _perturb(sys, x0_plus, x0_minus, u, dts, f_args, h_args):
    _, yi_plus = numsolve_sigma(sys, x0_plus, u, dts, f_args, h_args)
    _, yi_minus = numsolve_sigma(sys, x0_minus, u, dts, f_args, h_args)
    return yi_plus - yi_minus


@functools.partial(jax.jit, static_argnames=("sys", "eps"))
def _numlog(sys, x0, u, dt, eps, perturb_axis, f_args, h_args):
    if perturb_axis is None:
        perturb_axis = jnp.arange(0, sys.nx)
    perturb_axis = jnp.asarray(perturb_axis)

    perturb_bases = jnp.eye(x0.size)[:, perturb_axis]
    x0_plus = x0[..., None] + eps * perturb_bases
    x0_minus = x0[..., None] - eps * perturb_bases
    y_all = _perturb(sys, x0_plus, x0_minus, u, dt, f_args, h_args) / (2.0 * eps)

    coord_vec = jnp.arange(0, perturb_axis.size)
    [xm, ym] = jnp.meshgrid(coord_vec, coord_vec)

    return jnp.sum(dt[:, None, None] * y_all[:, :, xm] * y_all[:, :, ym], axis=(0, 1))


def numlog(sys, x0, u, dt, eps, perturb_axis=None, f_args=None, h_args=None):
    x0 = jnp.asarray(x0)
    u = jnp.asarray(u)
    dt = jnp.asarray(dt)
    n_steps = dt.size
    if x0.shape != (sys.nx,):
        raise ValueError(f"Expected x0 with shape ({sys.nx}), got {x0.shape}")

    if u.shape != (sys.nu, n_steps):
        raise ValueError(
            f"Expected matrix with shape ({sys.nu}, {n_steps}), got {u.shape}"
        )
    if jnp.any(dt <= 0):
        raise ValueError("Discrete time-step is not positive.")
    return _numlog(sys, x0, u, dt, eps, perturb_axis, f_args, h_args)
