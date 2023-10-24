import numpy as np


def compute_gramian(x0, u_all, dt, n_timesteps, eps):
    if x0.shape != (3,):
        raise ValueError("Expected x0 with shape (3), got " + str(x0.shape))
    if u_all.shape != (2, n_timesteps):
        raise ValueError(
            "Expected matrix with shape ( 2, "
            + str(n_timesteps)
            + "); got "
            + str(u_all.shape)
        )
    if dt <= 0:
        raise ValueError("Discrete time-step is not positive.")

    y_all = np.zeros(
        (3, 2, n_timesteps)
    )  # y_all[i,k,:] gives the y^k(t) of the normalised x0-perturbations in the i direction

    for i in range(0, 3):
        x0_plus = np.array(x0, copy=True)
        x0_plus[i] += eps
        x0_minus = np.array(x0, copy=True)
        x0_minus[i] -= eps
        yi_plus = obs_system(solve_system(x0_plus, u_all, dt, n_timesteps), n_timesteps)
        yi_minus = obs_system(
            solve_system(x0_minus, u_all, dt, n_timesteps), n_timesteps
        )
        y_all[i, :, :] = (yi_plus - yi_minus) / (2 * eps)

    gramian = np.zeros((3, 3))
    for i in range(0, 3):
        for j in range(0, 3):
            gramian[i, j] = dt * np.tensordot(y_all[i, :, :], y_all[j, :, :], axes=2)

    return gramian


def solve_system(x0, u_all, dt, n_timesteps):
    x = np.zeros((3, n_timesteps))
    x[:, 0] = np.array(x0)
    for i in range(1, n_timesteps):
        x[0, i] = x[0, i - 1] + dt * (u_all[0, i - 1] * np.cos(x[2, i - 1]))
        x[1, i] = x[1, i - 1] + dt * (u_all[0, i - 1] * np.sin(x[2, i - 1]))
        x[2, i] = x[2, i - 1] + dt * (u_all[1, i - 1])

    return x


def obs_system(x, n_timesteps):
    y = np.zeros((2, n_timesteps))
    y[0, :] = np.arctan2(x[1, :], x[0, :])
    y[1, :] = (x[0, :] ** 2 + x[1, :] ** 2) ** (1 / 2)
    return y


# dt = 0.01
# eps = 1e-3
# n_timesteps = 1000
# x0 = np.array([1.0, 0.0, 0.0])
# u_all = 1.0*np.ones((2, n_timesteps))

# gramian = compute_gramian(x0, u_all, dt, n_timesteps, eps)
# condition_number = np.linalg.cond(gramian)
# print(gramian)
# print(condition_number)
