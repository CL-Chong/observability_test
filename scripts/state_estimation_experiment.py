import argparse
import pickle
import tomllib

import exlib
import jax
import jax.experimental.compilation_cache.compilation_cache as cc
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

jax.config.update("jax_platform_name", "cpu")

import observability_aware_control.algorithms.misc.simple_ekf as ekf
from observability_aware_control.algorithms.common import forward_dynamics
from observability_aware_control.models import multi_quadrotor

cc.initialize_cache("./.cache")
jax.config.update("jax_enable_x64", True)


def main():
    parser = argparse.ArgumentParser("state_estimation_experiment")
    parser.add_argument(
        "config", type=str, help="Configuration file for the experiment"
    )
    parser.add_argument(
        "--load", type=str, default="", help="Pickle file for experimental data"
    )

    args = parser.parse_args()
    with open(str(args.config), "rb") as fp:
        cfg = tomllib.load(fp)

    n_robots = cfg["model"]["n_robots"]
    interrobot_observation_kind = cfg["model"]["interrobot_observation_kind"]
    interrobot_observation_dim = 2 if interrobot_observation_kind == "bearings" else 1
    cov = np.diag(
        np.r_[
            np.full(multi_quadrotor.DIM_LEADER_POS_OBS, 1e-2),
            np.full(multi_quadrotor.DIM_ATT_OBS * n_robots, 1e-2),
            np.full(interrobot_observation_dim * (n_robots - 1), 1e-2),
            np.full(multi_quadrotor.DIM_VEL_OBS * n_robots, 1e-2),
        ]
    )

    mdl = multi_quadrotor.MultiQuadrotor(
        n_robots,
        cfg["model"]["robot_mass"],
        stlog_order=cfg["stlog"]["order"],
        has_odom=True,
        stlog_cov=cov,
        interrobot_observation_kind=interrobot_observation_kind,
    )

    n_robots = mdl.n_robots
    if args.load:
        print(f"Loading state estimation data from {args.load}")
        with open(args.load, "rb") as fp:
            trial = pickle.load(fp)
    else:
        trial = run_experiment(mdl, cfg)
    run_plot(n_robots, trial, cfg)


def run_plot(n_robots, trial, config):
    fig, ax1 = plt.subplots(nrows=n_robots - 1)
    for id_trial, (trial_name, trial_it) in enumerate(trial.items()):
        time, cov_hist, x_err = trial_it

        for idx in range(1, n_robots):
            err_mag = np.linalg.norm(x_err[:, idx, :], axis=-1)
            ax1[idx - 1].plot(time, err_mag)
            cov_mag = np.linalg.norm(cov_hist[:, idx, :], axis=-1)
            rmse = exlib.rms(err_mag)
            ax1[idx - 1].axhline(y=rmse, linestyle="--", color=f"C{id_trial}")
            l, r = time[0], time[-1]

            text_x_pos = l + (0.05 + 0.33 * id_trial) * (r - l)

            ax1[idx - 1].annotate(
                (f"{trial_name}\n" r"RMS($||\hat{\mathbf{e}}_p||$) = " f"{rmse:.4}m"),
                (text_x_pos, rmse),
                (text_x_pos, 0.5),
                arrowprops={
                    "width": 1,
                    "facecolor": "k",
                    "edgecolor": "None",
                    "alpha": 0.6,
                },
                fontsize=8,
            )
            ax1[idx - 1].set_ylabel(
                r"$\overset{\mathrm{Follower\ %d}}{||\hat{\mathbf{e}}_p||}$ (m)" % idx,
                fontsize=14,
            )
            ax1[idx - 1].fill_between(time, cov_mag, alpha=0.2)
            ax1[idx - 1].set_ylim(0, 1)
            ax1[idx - 1].set_xlim(time[0], time[-1])
    fig.supxlabel("Time (s)", fontsize=14)
    new_var = config["session"].get("image_save", "state_estimation_results.png")
    fig.tight_layout()
    fig.savefig(new_var)
    plt.show()


def run_experiment(mdl, config):
    kf = ekf.SimpleEKF(
        jax.jit(
            lambda x, u, dt: forward_dynamics(mdl.dynamics, x, u, dt, method="euler")
        ),
        jax.jit(mdl.observation),
        jnp.diag(jnp.tile(jnp.r_[1, jnp.full(3, 1)] / 20, mdl.n_robots)),
        mdl.cov,
    )
    seed = config["session"].get("seed", 100)
    key = jax.random.PRNGKey(seed)
    experimental_data = config["session"]["experimental_data"]
    n_samples = config["session"]["n_samples"]
    n_experiments = len(experimental_data)
    print(
        f"Running new experiment with seed {seed}, repeating {n_experiments} cases"
        f" {n_samples} times each"
    )
    keys = jax.random.split(key, (n_experiments, n_samples))

    trial = {}
    for k, v in experimental_data.items():
        results = jnp.load(v["file"])
        states = results["states"]
        inputs = results["inputs"]
        time = results["time"]

        trial[k] = exlib.evaluate_state_estimation(
            kf, states, inputs, time, jnp.eye(mdl.nx) / 30, mdl, keys[2, :]
        )

    with open(config["session"].get("save", "state_estimation_data.pkl"), "wb") as fp:
        pickle.dump(trial, fp)
    return trial


if __name__ == "__main__":
    main()
