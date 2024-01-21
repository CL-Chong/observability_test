import tomllib
import pickle
import numpy as np
import matplotlib.pyplot as plt
import jax
import tqdm
import jax.numpy as jnp
import argparse
import jax.experimental.compilation_cache.compilation_cache as cc
from observability_aware_control.models import multi_quadrotor

from observability_aware_control.algorithms import forward_dynamics
import observability_aware_control.algorithms.misc.simple_ekf as ekf


def gaussian_noise(key, cov, shape):
    noise = jax.random.multivariate_normal(key, jnp.zeros(cov.shape[0]), cov, shape)
    return jnp.diagonal(noise, axis1=1, axis2=2)


@jax.tree_util.Partial(jax.jit, static_argnums=[0])
def run_state_est(kf, xs, us, dt, cov_op_init, key):
    f_key, h_key = jax.random.split(key)

    def ekf_update(x_tup, u_tup):
        x_op, cov_op = x_tup
        u, y = u_tup

        x_pred, cov_pred = kf.predict(x_op, cov_op, u, dt)
        res = kf.update(x_pred, cov_pred, y)

        return res, res

    xs_tup = (jnp.array(xs[0, ...]), jnp.array(cov_op_init))
    u_noise = gaussian_noise(f_key, kf.in_cov, us.shape) / 4
    ys = jax.vmap(kf.hfcn)(xs)
    y_noise = gaussian_noise(h_key, kf.obs_cov, ys.shape)
    us_tup = (us + u_noise, ys + y_noise)
    _, (x_hist, cov_hist) = jax.lax.scan(ekf_update, xs_tup, us_tup)

    return x_hist, cov_hist


cc.initialize_cache("./.cache")

jax.config.update("jax_enable_x64", True)


def rms(data):
    return jnp.sqrt((data**2).mean())


def evaluate_state_estimation(kf, states, inputs, time, init_cov, mdl, keys):
    x_errs = []
    cov_hists = []
    dt = time[1] - time[0]

    for seed in tqdm.tqdm(keys):
        x_hist, cov_hist = run_state_est(kf, states, inputs, dt, init_cov, seed)

        x_err = x_hist - states
        x_err = x_err.reshape(x_err.shape[0], mdl.n_robots, -1)[:, :, 0:3]
        x_errs.append(x_err)

        cov_hist = 3 * jnp.sqrt(jnp.diagonal(cov_hist, axis1=1, axis2=2))
        cov_hist = cov_hist.reshape(cov_hist.shape[0], mdl.n_robots, -1)[:, :, 0:3]
        cov_hists.append(cov_hist)

    x_err = jnp.array(x_errs).mean(axis=0)
    cov_hist = jnp.array(cov_hists).mean(axis=0)
    return time, cov_hist, x_err


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

    mdl = multi_quadrotor.MultiQuadrotor(
        cfg["model"]["n_robots"], cfg["model"]["robot_mass"], has_odom=True
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
            rmse = rms(err_mag)
            ax1[idx - 1].axhline(y=rmse, linestyle="--", color=f"C{id_trial}")
            l, r = time[0], time[-1]

            text_x_pos = l + (0.05 + 0.33 * id_trial) * (r - l)

            ax1[idx - 1].annotate(
                (f"{trial_name}\n" r"RMS($||\hat{\mathbf{e}}_p||$) = " f"{rmse:.4}m"),
                (text_x_pos, rmse),
                (text_x_pos, 7),
                arrowprops={
                    "width": 1,
                    "facecolor": "k",
                    "edgecolor": "None",
                    "alpha": 0.6,
                },
                fontsize=8,
            )
            ax1[idx - 1].set_ylabel(r"\hat{\mathbf{e}}_p (m)")
            ax1[idx - 1].set_xlabel("Time (s)")
            ax1[idx - 1].set_title(f"Follower {idx}")
            ax1[idx - 1].fill_between(time, cov_mag, alpha=0.2)
            ax1[idx - 1].set_ylim(top=9)
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
        jnp.diag(
            jnp.r_[
                jnp.full(mdl.DIM_LEADER_POS_OBS, 1e-2),
                jnp.full(mdl.DIM_ATT_OBS * mdl.n_robots, 1e-2),
                jnp.full(mdl.DIM_BRNG_OBS * (mdl.n_robots - 1), 1e-2),
                jnp.full(mdl.DIM_VEL_OBS * mdl.n_robots, 1e-2),
            ]
        ),
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

        trial[k] = evaluate_state_estimation(
            kf, states, inputs, time, 3 * jnp.eye(mdl.nx), mdl, keys[2, :]
        )

    with open(config["session"].get("save", "state_estimation_data.pkl"), "wb") as fp:
        pickle.dump(trial, fp)
    return trial


if __name__ == "__main__":
    main()
