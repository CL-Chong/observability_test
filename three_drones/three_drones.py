import math

import casadi
import numpy as np


def numsolve_sigma(sys, x0, u_control, u_drone, dt, n_steps):
    x = np.zeros((sys.nx, n_steps))
    x[:, 0] = np.array(x0)

    y = np.zeros((sys.ny, n_steps))
    y[:, 0] = sys.observation(x0)
    for i in range(1, n_steps):
        dx = sys.dynamics(x[:, i - 1], u_control[:, i - 1], u_drone[:, i - 1]).ravel()
        x[:, i] = x[:, i - 1] + dt * dx
        y[:, i] = sys.observation(x[:, i])

    return x, y


def numlog(sys, x0, u_control, u_drone, dt, n_steps, eps):
    if x0.shape != (sys.nx,):
        raise ValueError(f"Expected x0 with shape ({sys.nx}), got {x0.shape}")
    if u_control.shape != (sys.NU, n_steps):
        raise ValueError(
            f"Expected matrix with shape ({sys.NU}, {n_steps}), got {u_control.shape}"
        )
    if u_drone.shape != (sys.nu - sys.NU, n_steps):
        raise ValueError(
            f"Expected matrix with shape ({sys.nu - sys.NU}, {n_steps}), got {u_drone.shape}"
        )
    if dt <= 0:
        raise ValueError("Discrete time-step is not positive.")

    y_all = np.zeros(
        (sys.nx, sys.ny, n_steps)
    )  # y_all[i,k,:] gives the y^k(t) of the normalised x0-perturbations in the i direction

    for i in range(0, sys.nx):
        x0_plus = np.array(x0, copy=True)
        x0_plus[i] += eps
        x0_minus = np.array(x0, copy=True)
        x0_minus[i] -= eps
        _, yi_plus = numsolve_sigma(sys, x0_plus, u_control, u_drone, dt, n_steps)
        _, yi_minus = numsolve_sigma(sys, x0_minus, u_control, u_drone, dt, n_steps)
        y_all[i, :, :] = (yi_plus - yi_minus) / (2 * eps)

    gramian = np.zeros((sys.nx, sys.nx))
    for i in range(0, sys.nx):
        for j in range(0, i + 1):
            gramian[i, j] = dt * np.tensordot(y_all[i, :, :], y_all[j, :, :], axes=2)
            gramian[j, i] = gramian[i, j]

    return gramian


def stlog_symbolic(sys, order):
    x = casadi.MX.sym("x", sys.nx)
    u = casadi.MX.sym("u", sys.nu - sys.NU)
    v = casadi.MX.sym("v", sys.NU)
    T = casadi.MX.sym("T")
    stlog = casadi.MX.zeros(sys.nx, sys.nx)
    lh_store = casadi.MX.zeros(sys.ny, order + 1)
    lh_store[:, 0] = sys.observation(x)

    for l in range(0, order):
        lh_store[:, l + 1] = casadi.jtimes(lh_store[:, l], x, sys.dynamics(x, v, u))

    for l in range(0, order):
        for k in range(0, l + 1):
            stlog += (
                (T ** (l + 1)) / ((l + 1) * math.factorial(k) * math.factorial(l))
            ) * casadi.mtimes(
                casadi.jacobian(lh_store[:, k], x).T,
                casadi.jacobian(lh_store[:, l - k], x),
            )

    stlog_fun = casadi.Function("stlog_fun", [x, v, u, T], [stlog])

    return stlog_fun
