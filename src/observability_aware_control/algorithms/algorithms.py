import numpy as np
import joblib


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


def numlog(sys, x0, u_control, u_drone, dt, n_steps, eps, perturb_axis=None):
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

    if perturb_axis is None:
        perturb_axis = np.arange(0, sys.nx, dtype=np.int64)

    n_perturbed_x = perturb_axis.size

    @joblib.delayed
    def _perturb(i):
        x0_plus = np.array(x0, copy=True)
        x0_plus[i] += eps
        x0_minus = np.array(x0, copy=True)
        x0_minus[i] -= eps
        _, yi_plus = numsolve_sigma(sys, x0_plus, u_control, u_drone, dt, n_steps)
        _, yi_minus = numsolve_sigma(sys, x0_minus, u_control, u_drone, dt, n_steps)
        return (yi_plus - yi_minus) / (2 * eps)

    par_evaluator = joblib.Parallel(12)
    y_all = par_evaluator(_perturb(i) for i in perturb_axis)
    if y_all is None:
        raise RuntimeError("Perturbation failed")

    gramian = np.zeros((n_perturbed_x, n_perturbed_x))
    for i in range(0, n_perturbed_x):
        for j in range(0, i + 1):
            gramian[i, j] = dt * (y_all[i] * y_all[j]).sum()

    gramian = np.tril(gramian, -1).T + np.tril(gramian)
    return gramian
