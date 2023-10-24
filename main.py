import numpy as np
import casadi
import math

# from observability_test_small import compute_gramian
from three_drones import stlog_symbolic, numlog


def main():
    dt = 0.01
    eps = 1e-3
    n_steps = 50
    x0 = np.array([0.0, 0.0, 0.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0])
    [v0, v1, u0, u1, u2, u3] = [1.0, -0.5, 0.3, -0.4, 0.5, 0.6]
    u_control = np.array([v0 * np.ones(n_steps), v1 * np.ones(n_steps)])
    u_drone = np.array(
        [
            u0 * np.ones(n_steps),
            u1 * np.ones(n_steps),
            u2 * np.ones(n_steps),
            u3 * np.ones(n_steps),
        ]
    )
    numlog_x = numlog(x0, u_control, u_drone, dt, n_steps, eps)
    print("numlog diagonstics:")
    print(numlog_x)
    print("condition number = " + str(np.linalg.cond(numlog_x)))

    stlog_fun = stlog_symbolic(2)
    stlog_x = stlog_fun(x0, [v0, v1], [u0, u1, u2, u3], dt * n_steps)
    print("stlog diagonstics:")
    print(stlog_x)
    print("condition number = " + str(np.linalg.cond(stlog_x)))

    err_mat = numlog_x - stlog_x
    err_rms = np.sum(err_mat**2) / 81.0
    print("matrix of differences:")
    print(err_mat)
    print("rms error = " + str(err_rms))


# def main_old():
#    dt = 0.01
#    eps = 1e-3
#    n_timesteps = 1000
#    x0 = np.array([1.0, 0.0, 0.0])
#    u_all = 1.0 * np.ones((2, n_timesteps))
#
#    gramian = compute_gramian(x0, u_all, dt, n_timesteps, eps)
#    condition_number = np.linalg.cond(gramian)
#    print(gramian)
#    print(condition_number)


if __name__ == "__main__":
    main()
