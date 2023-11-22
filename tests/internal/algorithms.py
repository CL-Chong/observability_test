import joblib
import numpy as np


def _scan(f, init, xs):
    carry = init
    ys = []
    for x in xs:
        carry, y = f(carry, x)
        ys.append(y)
    return np.stack(ys)


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
    Tuple[np.array, np.array] | np.array
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
        xs = np.vstack([u, dt]).T
    else:
        f_args = np.asarray(f_args)
        xs = np.vstack([u, dt, f_args]).T

    x = _scan(_update, init=x0, xs=xs)

    if without_observation:
        return x.T

    if h_args is None:
        y = np.apply_along_axis(sys.observation, 1, x).T
    else:
        h_args = np.asarray(h_args)

        xs = np.hstack([x, h_args.T])
        y = np.apply_along_axis(
            lambda _x: sys.observation(_x[: sys.nx], _x[sys.nx :]), 1, xs
        ).T

    return x.T, y


def numlog(sys, x0, u, dt, eps, perturb_axis=None, f_args=None, h_args=None):
    dt = np.asarray(dt)
    n_steps = dt.size
    if x0.shape != (sys.nx,):
        raise ValueError(f"Expected x0 with shape ({sys.nx}), got {x0.shape}")
    if u.shape != (sys.nu, n_steps):
        raise ValueError(
            f"Expected matrix with shape ({sys.NU}, {n_steps}), got {u.shape}"
        )

    if np.any(dt) <= 0:
        raise ValueError("Discrete time-step is not positive.")

    if perturb_axis is None:
        perturb_axis = np.arange(0, sys.nx, dtype=np.int64)
    perturb_axis = np.asarray(perturb_axis)

    n_perturbed_x = perturb_axis.size

    @joblib.delayed
    def _perturb(i):
        x0_plus = np.array(x0, copy=True)
        x0_plus[i] += eps
        x0_minus = np.array(x0, copy=True)
        x0_minus[i] -= eps
        _, yi_plus = numsolve_sigma(sys, x0_plus, u, dt, f_args, h_args)
        _, yi_minus = numsolve_sigma(sys, x0_minus, u, dt, f_args, h_args)
        return (yi_plus - yi_minus) / (2 * eps)

    par_evaluator = joblib.Parallel(12)
    y_all = par_evaluator(_perturb(i) for i in perturb_axis)
    if y_all is None:
        raise RuntimeError("Perturbation failed")

    gramian = np.zeros((n_perturbed_x, n_perturbed_x))
    for i in range(0, n_perturbed_x):
        for j in range(0, i + 1):
            gramian[i, j] = (dt * y_all[i] * y_all[j]).sum()

    gramian = np.tril(gramian, -1).T + np.tril(gramian)
    return gramian
