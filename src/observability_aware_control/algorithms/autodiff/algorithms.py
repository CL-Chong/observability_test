import functools

import jax
import jax.numpy as jnp

NX = 3
NU = 2


@functools.partial(jax.jit, static_argnames=("sys", "without_observation", "axis"))
def numsolve_sigma(
    sys,
    x0,
    u,
    dt,
    axis=None,
    f_args=None,
    h_args=None,
    without_observation=False,
):
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
    axis: int
        Axis that defines each vector-valued input
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

    def _process_in_axis(var):
        return jnp.moveaxis(var, axis, 0) if axis is not None else var

    def _process_out_axis(var):
        return jnp.moveaxis(var, 0, axis) if axis is not None else var

    def _update(x_op, tup):
        dt, *rem = tup
        dx = sys.dynamics(x_op, *rem)
        return x_op + dt * dx, x_op

    u = _process_in_axis(u)
    if f_args is None:
        xs = (dt, u)
    else:
        xs = (dt, u, _process_in_axis(f_args))

    _, x = jax.lax.scan(_update, init=x0, xs=xs)

    if without_observation:
        return _process_out_axis(x)

    if h_args is None:
        y = sys.observation(x)
    else:
        y = sys.observation(x, _process_in_axis(h_args))

    return _process_out_axis(x), _process_out_axis(y)


@functools.partial(jax.jit, static_argnames=("sys", "eps", "axis"))
def _numlog(sys, x0, u, dt, eps, axis, perturb_axis, f_args, h_args):
    if perturb_axis is None:
        perturb_axis = jnp.arange(0, sys.nx)
    perturb_axis = jnp.asarray(perturb_axis)

    @functools.partial(jax.vmap, out_axes=2)
    def _perturb(x0_plus, x0_minus):
        _, yi_plus = numsolve_sigma(sys, x0_plus, u, dt, axis, f_args, h_args)
        _, yi_minus = numsolve_sigma(sys, x0_minus, u, dt, axis, f_args, h_args)
        return yi_plus - yi_minus

    perturb_bases = jnp.eye(x0.size)[perturb_axis]
    x0_plus = x0 + eps * perturb_bases
    x0_minus = x0 - eps * perturb_bases
    y_all = _perturb(x0_plus, x0_minus) / (2.0 * eps)

    coord_vec = jnp.arange(0, perturb_axis.size)
    xm, ym = jnp.meshgrid(coord_vec, coord_vec)

    dt = dt[..., None, None, None]
    if axis is not None:
        dt = jnp.moveaxis(dt, 0, axis)

    return jnp.sum(dt * y_all[:, :, xm] * y_all[:, :, ym], axis=(0, 1))


def numlog(sys, x0, u, dt, eps, axis=None, perturb_axis=None, f_args=None, h_args=None):
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
    return _numlog(sys, x0, u, dt, eps, axis, perturb_axis, f_args, h_args)
