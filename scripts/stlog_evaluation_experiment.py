import pickle
import tomllib

import jax
import jax.experimental.compilation_cache.compilation_cache as cc
import jax.numpy as jnp
import matplotlib.pyplot as plt
import tqdm
from matplotlib import ticker

from observability_aware_control.algorithms import cooperative_localization, stlog
from observability_aware_control.algorithms.misc import (
    simple_ekf,
    trajectory_generation,
)
from observability_aware_control.models import multi_quadrotor


def run_state_est(kf, x0, us, dt, cov_op_init):
    cov_num = []
    cov_op = jnp.array(cov_op_init)
    x = jnp.array(x0)
    for u in us:
        x_pred, cov_pred = kf.predict(x, cov_op, u, dt)
        x, cov_op = kf.update(x_pred, cov_pred, kf.hfcn(x_pred))
        cov_num.append(jnp.array(cov_op))
    return jnp.stack(cov_num)


def evaluate_trajectory(min_problem, kf, us, x0, dt, i_stlog):
    cost, stlog, xs = min_problem.opc(us, x0, dt, return_stlog=True, return_traj=True)
    cov = run_state_est(kf, x0, us, dt, jnp.eye(30) / 10)[i_stlog]

    stddev = 3 * jnp.sqrt(jnp.diagonal(cov, axis1=1, axis2=2))
    return dict(
        zip(["cost", "stlog", "xs", "cov", "stddev"], [cost, stlog, xs, cov, stddev])
    )


cc.set_cache_dir("./.cache")


def main():
    with open("./config/stlog_evaluation_experiment.toml", "rb") as fp:
        cfg = tomllib.load(fp)

    mdl = multi_quadrotor.MultiQuadrotor(
        cfg["model"]["n_robots"],
        cfg["model"]["robot_mass"],
        has_odom=True,
        has_baro=False,
        stlog_order=1,
    )

    ms = trajectory_generation.MinimumSnap(
        cfg["initial_path"]["degree"],
        cfg["initial_path"]["derivative_weights"],
        trajectory_generation.MinimumSnapAlgorithm.CONSTRAINED,
    )

    p_refs = jnp.array(cfg["initial_path"]["position_reference"])
    t_ref = jnp.array(cfg["initial_path"]["time_reference"])
    n_steps = cfg["initial_path"]["n_timesteps"]
    dt = jnp.diff(t_ref) / n_steps
    t_sample = jnp.arange(0, n_steps) * dt
    states_traj, inputs_traj = ms.generate_trajectories(
        jnp.tile(t_ref, [cfg["model"]["n_robots"], 1]),
        p_refs,
        jnp.tile(t_sample, [cfg["model"]["n_robots"], 1]),
    )

    # ---------------------------Setup the Optimizer----------------------------
    window = cfg["opc"]["window_size"]
    try:
        with open("stlogdata.pkl", "rb") as fp:
            trials = pickle.load(fp)
    except FileNotFoundError:

        u_lb = jnp.tile(jnp.array(cfg["optim"]["lb"]), (window, mdl.n_robots))
        u_ub = jnp.tile(jnp.array(cfg["optim"]["ub"]), (window, mdl.n_robots))
        opts = cfg["optim"]["options"]
        opts = cooperative_localization.CooperativeLocalizationOptions(
            window=window,
            id_leader=0,
            lb=u_lb,
            ub=u_ub,
            obs_comps=cfg["opc"]["observed_components"],
            method=cfg["optim"]["method"],
            optim_options=opts,
            min_v2v_dist=-1,
            max_v2v_dist=-1,
        )
        min_problem = cooperative_localization.CooperativeLocalizingOPC(mdl, opts)

        obs_comps = jnp.array(cfg["opc"]["observed_components"])
        init_index = cfg["trials"]["init_timestep"]
        x0 = states_traj[:, :, init_index].ravel()
        us = (
            inputs_traj[:, :, init_index : init_index + window]
            .swapaxes(0, 1)
            .reshape(-1, window, order="F")
            .T
        )

        i_stlog = (...,) + jnp.ix_(obs_comps, obs_comps)
        kf = simple_ekf.SimpleEKF(
            lambda x, u, dt: x + dt * mdl.dynamics(x, u),
            mdl.observation,
            jnp.diag(jnp.tile(jnp.r_[0.1, jnp.full(3, 0.01)], mdl.n_robots)),
            jnp.diag(
                jnp.r_[
                    jnp.full(multi_quadrotor.DIM_LEADER_POS_OBS, 1e-1),
                    jnp.full(multi_quadrotor.DIM_ATT_OBS * mdl.n_robots, 1e-3),
                    jnp.full(multi_quadrotor.DIM_BRNG_OBS * (mdl.n_robots - 1), 1e-3),
                    jnp.full(multi_quadrotor.DIM_VEL_OBS * mdl.n_robots, 1e-3),
                ]
            ),
        )

        trials = {}
        trials["init"] = evaluate_trajectory(min_problem, kf, us, x0, dt, i_stlog)
        soln = min_problem.minimize(x0, us, dt)
        us = soln.x
        trials["opt"] = evaluate_trajectory(min_problem, kf, us, x0, dt, i_stlog)
        key = jax.random.PRNGKey(1000)
        n_rand = cfg["trials"]["random_trials"]

        rand_us = jax.random.uniform(
            key,
            shape=(n_rand,) + u_lb.shape,
            minval=jnp.stack([u_lb] * n_rand),
            maxval=jnp.stack([u_ub] * n_rand),
        )

        trials["rand"] = [
            evaluate_trajectory(min_problem, kf, us, x0, dt, i_stlog)
            for us in tqdm.tqdm(rand_us)
        ]

        with open("stlogdata.pkl", "wb") as fp:
            pickle.dump(trials, fp)

    rand_stddev = jnp.stack([it["stddev"] for it in trials["rand"]])

    time = jnp.r_[0:window] * dt
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    ax.set_xlim(-50, 50)
    ax.set_ylim(-50, 50)
    ax.set_zlim(-50, 50)
    for it in trials["rand"]:
        xs_all = it["xs"].reshape(40, 3, 10)
        for idx in range(3):
            xs = xs_all[:, idx, :]
            ax.plot(xs[:, 0], xs[:, 1], xs[:, 2], alpha=0.1, color="k")

    xs_all = trials["init"]["xs"].reshape(40, 3, 10)
    for idx in range(3):
        xs = xs_all[:, idx, :]
        ax.plot(xs[:, 0], xs[:, 1], xs[:, 2])

    xs_all = trials["opt"]["xs"].reshape(40, 3, 10)
    for idx in range(3):
        xs = xs_all[:, idx, :]
        ax.plot(xs[:, 0], xs[:, 1], xs[:, 2], linewidth=2)

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    fig.tight_layout()
    fig.savefig("data/stlog_evaluation_trajectories.png")
    for idv in range(2):
        fig, axs = plt.subplots(3)
        for idx, it in enumerate("xyz"):
            ax = axs[idx]
            ax.boxplot(
                rand_stddev[:, :, idv * 3 + idx].swapaxes(0, 1) ** 2 / 3,
                positions=time,
                manage_ticks=False,
                patch_artist=True,
                medianprops={"color": "white", "linewidth": 0.5},
                boxprops={"linewidth": 0, "facecolor": "C1", "alpha": 0.5},
                flierprops={
                    "markersize": 5,
                    "markerfacecolor": "C2",
                    "markeredgecolor": "None",
                    "alpha": 0.5,
                },
                widths=0.1,
            )
            ax.plot(
                time,
                trials["init"]["stddev"][:, idv * 3 + idx] ** 2 / 3,
                "--",
                color="r",
                # label=rf"$\sigma(\mathbf{{p}}_{it})$ on initial trajectory",
                alpha=0.5,
                linewidth=2,
            )
            ax.plot(
                time,
                trials["opt"]["stddev"][:, idv * 3 + idx] ** 2 / 3,
                color="b",
                # label=rf"$\sigma(\mathbf{{p}}_{it})$ on optimized trajectory",
                linewidth=2,
            )
            # ax.legend()

            # ax.set_xticks(jnp.round(time[::2], 2))
            ax.xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.1f}"))
            ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.2f}"))
            ax.set_ylabel(rf"$\sigma^2(\mathbf{{p}}_{it})$ (m)", fontsize=15)
        fig.supxlabel("Time (s)", fontsize=15)
        fig.savefig(f"data/stlog_evaluation_results_v{idv}.png", bbox_inches="tight")
        fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
