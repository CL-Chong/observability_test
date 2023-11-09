import functools

import casadi as cs
import matplotlib.pyplot as plt
import numpy as np

from src.models import models
from src.observability_aware_control import algorithms
from src.observability_aware_control.algorithms.misc import simple_ekf
from src.utils import utils


def main():
    # Load data from optimization session
    data = np.load("data/optimization_results.npz")
    data = utils.expand_dict(data)

    x0_follower = data["params"]["x0_follower"]
    u_follower = data["params"]["u_follower"]
    dt = data["params"]["dt"]

    # Build estimator
    model = models.ReferenceSensingRobots(n_robots=2, is_symbolic=True)
    kf = simple_ekf.SimpleEKF(
        model,
        x0=np.array(x0_follower),
        kf_cov_0=0.1 * np.eye(model.nx),
        in_cov=0.1 * np.eye(model.nu),
        obs_cov=0.05 * np.eye(model.ny),
    )

    # Setup leader trajectory to be used as observation model parameter
    x_leader = algorithms.numsolve_sigma(
        models.Robot(),
        data["params"]["x0_leader"],
        data["params"]["u_leader"],
        dt,
        without_observation=True,
    )

    assert isinstance(x_leader, np.ndarray)
    pos_leader = x_leader[0:2, :]

    # Using loaded data and parameters, bind several arguments to estimation and
    # simulation functions
    forward_simulate_followers = functools.partial(
        algorithms.numsolve_sigma,
        sys=models.ReferenceSensingRobots(n_robots=2),
        x0=x0_follower,
        dt=dt,
        h_args=pos_leader,
        without_observation=True,
    )

    rng = np.random.default_rng(281919)

    def generate_obs(s):
        return kf.hfcn(s["x"], s["p"]) + rng.normal(0, 0.05)

    estimate = functools.partial(
        kf.estimate, dt=dt, params=pos_leader, meas=generate_obs
    )

    reshape = functools.partial(
        np.reshape, newshape=(models.Robot.NX, model.n_robots, -1), order="F"
    )

    hist = {"raw": {}, "soln": {}}
    # Generate true and estimated trajectories using unoptimized inputs
    hist["raw"]["pos_true"] = reshape(forward_simulate_followers(u=u_follower))
    x_hist, cov_hist, hist["raw"]["t"] = estimate(u_follower)
    hist["raw"]["pos_est"] = reshape(x_hist)
    hist["raw"]["pos_cov"] = reshape(cov_hist.diagonal(axis1=1, axis2=2).T)[0:2, ...]

    # Generate true and estimated trajectories using optimized inputs
    u_optimized = data["soln"]["x"].reshape(model.nu, -1, order="F")
    hist["soln"]["pos_true"] = reshape(forward_simulate_followers(u=u_optimized))
    x_hist, cov_hist, hist["soln"]["t"] = estimate(u_optimized)
    hist["soln"]["pos_est"] = reshape(x_hist)
    hist["soln"]["pos_cov"] = reshape(cov_hist.diagonal(axis1=1, axis2=2).T)[0:2, ...]

    plotting(model, x_leader, hist)


def plotting(model, x_leader, hist):
    desc = {"raw": "Initial Guess", "soln": "Optimized"}

    figs = {}
    figs[0], ax = plt.subplots()
    ax.plot(x_leader[0, :], x_leader[1, :], "C9")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    for idx in range(model.n_robots):  # Vary vehicles
        for key, style in zip(("raw", "soln"), ("-", "--")):  # Toggle raw/soln
            for k1, alpha in zip(("pos_est", "pos_true"), (1, 0.2)):
                ax.plot(
                    hist[key][k1][0, idx, :],
                    hist[key][k1][1, idx, :],
                    style,
                    color=f"C{idx}",
                    alpha=alpha,
                )

    figs[0].savefig("data/trajectories.png")

    pos_err = {
        key: np.linalg.norm(
            hist[key]["pos_est"] - hist[key]["pos_true"], axis=0
        ).squeeze()
        for key in ("raw", "soln")
    }
    sigmas = {
        key: 3 * np.sqrt(np.linalg.norm(hist[key]["pos_cov"], axis=0))
        for key in ("raw", "soln")
    }
    figs[1], axs = plt.subplots(model.n_robots)
    for idx, ax in enumerate(axs):
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Pos. Est. Error (m)")
        for key in ("raw", "soln"):
            ax.plot(
                hist[key]["t"],
                pos_err[key][idx, :],
                label=(r"$\tilde{\mathbf{p}}_{x, y}$" f" ({desc[key]})"),
            )
            ax.fill_between(
                hist[key]["t"],
                sigmas[key][idx, :],
                alpha=0.2,
                label=(r"$3\sigma(\mathbf{p}_{x, y})$" f" ({desc[key]})"),
            )
        ax.legend(fontsize=8)
    figs[1].tight_layout()
    figs[1].savefig("data/estimation_performance.png")

    plt.show()


if __name__ == "__main__":
    main()
