import functools

import jax
import jax.numpy as jnp


@functools.partial(
    jax.jit, static_argnames=("dynamics", "method", "return_derivatives")
)
def forward_dynamics(dynamics, x0, u, dt, method="euler", return_derivatives=False):
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
    method: Literal["RK4"] | Literal["euler"]
        Specifies the integration method
    Returns
    -------
    Tuple[jnp.array, jnp.array] | jnp.array
        The state and optionally observation trajectory
    """

    def _update(x_op, tup):
        u, dt = tup
        dx = dynamics(x_op, u)
        if method == "RK4":
            k = jnp.empty((4, x_op.size))
            k = k.at[0, :].set(dx)
            k = k.at[1, :].set(dynamics(x_op + dt / 2 * k[0, :], u))
            k = k.at[2, :].set(dynamics(x_op + dt / 2 * k[1, :], u))
            k = k.at[3, :].set(dynamics(x_op + dt * k[2, :], u))
            increment = jnp.array([1, 2, 2, 1]) @ k / 6
        elif method == "euler":
            increment = dx
        else:
            raise NotImplementedError(f"{method} is not a valid integration method")
        x_new = x_op + dt * increment
        if return_derivatives:
            return x_new, (x_new, dx)
        return x_new, x_new

    if u.ndim == 1:
        _, x = _update(x0, (u, dt))
        return x

    _, x = jax.lax.scan(_update, init=x0, xs=(u, dt))
    return x
