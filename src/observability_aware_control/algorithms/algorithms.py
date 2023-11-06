import numpy as np
import joblib


def numsolve_sigma(sys, x0, u, dt):
    dt = np.asarray(dt)
    n_steps = dt.size
    x = np.zeros((sys.nx, n_steps))
    x[:, 0] = np.array(x0)

    y = np.zeros((sys.ny, n_steps))
    y[:, 0] = sys.observation(x0)
    for k, dt_k in enumerate(dt[1:], 1):
        dx = sys.dynamics(x[:, k - 1], u[:, k - 1]).ravel()
        x[:, k] = x[:, k - 1] + dt_k * dx
        y[:, k] = sys.observation(x[:, k])

    return x, y


def numlog(sys, x0, u, dt, eps, perturb_axis=None):
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
        _, yi_plus = numsolve_sigma(sys, x0_plus, u, dt)
        _, yi_minus = numsolve_sigma(sys, x0_minus, u, dt)
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
