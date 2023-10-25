import math

import casadi
import numpy as np


def numsolve_sigma(sys, x0, u_control, u_drone, dt, n_steps):
    x = np.zeros((9, n_steps))
    x[:, 0] = np.array(x0)
    for i in range(1, n_steps):
        dx = sys.dynamics(x[:, i - 1], u_control[:, i - 1], u_drone[:, i - 1]).ravel()
        x[:, i] = x[:, i - 1] + dt * dx

    return x


def obs_sigma(x, n_steps):
    y = np.zeros((7, n_steps))
    y[0, :] = x[0, :]
    y[1, :] = x[1, :]
    y[2, :] = x[2, :]
    y[3, :] = x[5, :]
    y[4, :] = x[8, :]
    rho1_h = np.cos(x[5, :]) * (x[3, :] - x[0, :]) - np.sin(x[5, :]) * (
        x[4, :] - x[1, :]
    )
    rho1_v = np.sin(x[5, :]) * (x[3, :] - x[0, :]) + np.cos(x[5, :]) * (
        x[4, :] - x[1, :]
    )
    y[5, :] = np.arctan2(rho1_v, rho1_h)
    rho2_h = np.cos(x[8, :]) * (x[6, :] - x[0, :]) - np.sin(x[8, :]) * (
        x[7, :] - x[1, :]
    )
    rho2_v = np.sin(x[8, :]) * (x[6, :] - x[0, :]) + np.cos(x[8, :]) * (
        x[7, :] - x[1, :]
    )
    y[6, :] = np.arctan2(rho2_v, rho2_h)
    return y


def numlog(sys, x0, u_control, u_drone, dt, n_steps, eps):
    if x0.shape != (9,):
        raise ValueError("Expected x0 with shape (9), got " + str(x0.shape))
    if u_control.shape != (2, n_steps):
        raise ValueError(
            "Expected matrix with shape ( 2, "
            + str(n_steps)
            + "); got "
            + str(u_control.shape)
        )
    if u_drone.shape != (4, n_steps):
        raise ValueError(
            "Expected matrix with shape ( 4, "
            + str(n_steps)
            + "); got "
            + str(u_drone.shape)
        )
    if dt <= 0:
        raise ValueError("Discrete time-step is not positive.")

    y_all = np.zeros(
        (9, 7, n_steps)
    )  # y_all[i,k,:] gives the y^k(t) of the normalised x0-perturbations in the i direction

    for i in range(0, 9):
        x0_plus = np.array(x0, copy=True)
        x0_plus[i] += eps
        x0_minus = np.array(x0, copy=True)
        x0_minus[i] -= eps
        yi_plus = obs_sigma(
            numsolve_sigma(sys, x0_plus, u_control, u_drone, dt, n_steps), n_steps
        )
        yi_minus = obs_sigma(
            numsolve_sigma(sys, x0_minus, u_control, u_drone, dt, n_steps), n_steps
        )
        y_all[i, :, :] = (yi_plus - yi_minus) / (2 * eps)

    gramian = np.zeros((9, 9))
    for i in range(0, 9):
        for j in range(0, i + 1):
            gramian[i, j] = dt * np.tensordot(y_all[i, :, :], y_all[j, :, :], axes=2)
            gramian[j, i] = gramian[i, j]

    return gramian


def stlog_symbolic(sys, order):
    x = casadi.MX.sym("x", sys.NX * sys.n_robots)
    u = casadi.MX.sym("u", sys.NU * (sys.n_robots - 1))
    v = casadi.MX.sym("v", sys.NU)
    T = casadi.MX.sym("T")
    stlog = casadi.MX.zeros(9, 9)
    lh_store = casadi.MX.zeros(7, order + 1)
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
