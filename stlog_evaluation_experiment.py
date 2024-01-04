import tomllib
import tqdm
from matplotlib import ticker
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import jax.experimental.compilation_cache.compilation_cache as cc
from observability_aware_control.models import multi_quadrotor
from observability_aware_control.algorithms import (
    STLOG,
    CooperativeOPCProblem,
    CooperativeLocalizationOptions,
)
import observability_aware_control.algorithms.misc.trajectory_generation as planning
import observability_aware_control.algorithms.misc.simple_ekf as ekf


def run_state_est(kf, xs, us, dt, cov_op_init):
    cov_num = []
    cov_op = jnp.array(cov_op_init)
    for x, u in zip(xs, us):
        x_pred, cov_pred = kf.predict(x, cov_op, u, dt)
        x, cov_op = kf.update(x_pred, cov_pred, kf.hfcn(x_pred))
        cov_num.append(jnp.array(cov_op))
    return jnp.stack(cov_num)


def evaluate_trajectory(min_problem, kf, us, x0, dt, i_stlog):
    cost, stlog, xs = min_problem.opc(us, x0, dt, return_stlog=True, return_traj=True)
    cov = run_state_est(kf, xs, us, dt, 0.01 * jnp.eye(30))[i_stlog]

    stddev = 3 * jnp.sqrt(jnp.diagonal(cov, axis1=1, axis2=2))
    return dict(
        zip(["cost", "stlog", "xs", "cov", "stddev"], [cost, stlog, xs, cov, stddev])
    )


cc.initialize_cache("./.cache")


def main():
    with open("./config/stlog_evaluation_experiment.toml", "rb") as fp:
        cfg = tomllib.load(fp)

    mdl = multi_quadrotor.MultiQuadrotor(
        cfg["model"]["n_robots"], cfg["model"]["robot_mass"]
    )

    ms = planning.MinimumSnap(
        cfg["initial_path"]["degree"],
        cfg["initial_path"]["derivative_weights"],
        planning.MinimumSnapAlgorithm.CONSTRAINED,
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
    stlog = STLOG(mdl, cfg["stlog"]["order"])

    window = cfg["opc"]["window_size"]
    u_lb = jnp.tile(jnp.array(cfg["optim"]["lb"]), (window, mdl.n_robots))
    u_ub = jnp.tile(jnp.array(cfg["optim"]["ub"]), (window, mdl.n_robots))
    opts = cfg["optim"]["options"]
    opts = CooperativeLocalizationOptions(
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
    min_problem = CooperativeOPCProblem(stlog, opts)

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
    kf = ekf.SimpleEKF(
        lambda x, u, dt: x + dt * mdl.dynamics(x, u),
        mdl.observation,
        jnp.diag(jnp.tile(jnp.r_[0.1, jnp.full(3, 0.01)], mdl.n_robots)),
        jnp.diag(
            jnp.r_[
                jnp.full(mdl.DIM_LEADER_POS_OBS, 1e-3),
                jnp.full(mdl.DIM_ALT_OBS * (mdl.n_robots - 1), 1e-3),
                jnp.full(mdl.DIM_ATT_OBS * mdl.n_robots, 1e-3),
                jnp.full(mdl.DIM_BRNG_OBS * (mdl.n_robots - 1), 1e-3),
            ]
        ),
    )

    # Use the .opc method since the full .objective method contains logic separating
    # independent from controlled variables

    init = evaluate_trajectory(min_problem, kf, us, x0, dt, i_stlog)
    soln = min_problem.minimize(x0, us, dt)
    us = soln.x
    opt = evaluate_trajectory(min_problem, kf, us, x0, dt, i_stlog)
    key = jax.random.PRNGKey(1000)
    n_rand = cfg["trials"]["random_trials"]

    rand_us = jax.random.uniform(
        key,
        shape=(n_rand,) + u_lb.shape,
        minval=jnp.stack([u_lb] * n_rand),
        maxval=jnp.stack([u_ub] * n_rand),
    )

    rand = [
        evaluate_trajectory(min_problem, kf, us, x0, dt, i_stlog)
        for us in tqdm.tqdm(rand_us)
    ]
    rand_stddev = jnp.stack([it["stddev"] for it in rand])

    fig, axs = plt.subplots(2, 3, figsize=(16, 8))
    time = jnp.r_[0:window] * dt
    for idv in range(2):
        for idx, it in enumerate("xyz"):
            ax = axs[idv, idx]
            ax.boxplot(
                rand_stddev[:, :, idv * 3 + idx].swapaxes(0, 1),
                positions=time,
                patch_artist=True,
                medianprops={"color": "white", "linewidth": 0.5},
                boxprops={"linewidth": 0, "facecolor": "C1", "alpha": 0.5},
                flierprops={
                    "markersize": 5,
                    "markerfacecolor": "C2",
                    "markeredgecolor": "None",
                    "alpha": 0.5,
                },
                widths=0.25,
            )
            ax.plot(
                time,
                init["stddev"][:, idv * 3 + idx],
                "--",
                color="r",
                label=f"Initial v{idv}, axis {idx}",
                alpha=0.5,
                linewidth=2,
            )
            ax.plot(
                time,
                opt["stddev"][:, idv * 3 + idx],
                color="b",
                label=f"Optimized v{idv}, axis {idx}",
                linewidth=2,
            )
            ax.legend()

            ax.set_xlabel("Time (s)")
            ax.set_xticks(jnp.round(time[::2], 2))
            ax.xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.1f}"))
            ax.set_ylabel("Position Standard Deviation (m)")
            ax.set_title(f"Vehicle {idv}, {it}-axis")
    fig.tight_layout()
    fig.savefig("data/stlog_evaluation_results.png")
    plt.show()


if __name__ == "__main__":
    main()
