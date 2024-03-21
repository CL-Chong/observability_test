import sys
import tomllib

import jax
import jax.experimental.compilation_cache.compilation_cache as cc
import matplotlib.pyplot as plt
import numpy as np
import tqdm

from observability_aware_control import utils
from observability_aware_control.algorithms import common, cooperative_localization
from observability_aware_control.models import multi_quadrotor
from observability_aware_control.utils.utils import generate_leader_trajectory

# testing list: (X) = bad, (1/2) = not sure, (O) = good
# nonlinear constraints (X)
# log-scaling (1/2)
# det and tr replacements for min eig (X)
# waypoints (1/2 - doesn't break, but doesn't achieve much)
# adaptive dt_stlog (O - continue to refine)
# psd mod (O - less transients than base, but less turning too)

cc.set_cache_dir("./.cache")

jax.config.update("jax_enable_x64", True)


u_eqm = np.r_[9.81, 0.0, 0.0, 0.0]
q_eqm = np.r_[np.zeros(3), 1.0]
v_eqm = np.zeros(3)


def main():
    with open("./config/quadrotor_control_experiment.toml", "rb") as fp:
        cfg = tomllib.load(fp)

    n_robots = cfg["model"]["n_robots"]
    cov = np.diag(
        np.r_[
            np.full(multi_quadrotor.DIM_LEADER_POS_OBS, 1e-2),
            np.full(multi_quadrotor.DIM_ATT_OBS * n_robots, 1e-2),
            np.full(multi_quadrotor.DIM_BRNG_OBS * (n_robots - 1), 1e-2),
            np.full(multi_quadrotor.DIM_VEL_OBS * n_robots, 1e-2),
        ]
    )

    mdl = multi_quadrotor.MultiQuadrotor(
        n_robots,
        cfg["model"]["robot_mass"],
        stlog_order=cfg["stlog"]["order"],
        has_odom=True,
        stlog_cov=cov,
    )
    window = cfg["opc"]["window_size"]
    u_lb = np.tile(np.array(cfg["optim"]["lb"]), (window, mdl.n_robots))
    u_ub = np.tile(np.array(cfg["optim"]["ub"]), (window, mdl.n_robots))
    opts = cooperative_localization.CooperativeLocalizationOptions(
        window=window,
        id_leader=0,
        lb=u_lb,
        ub=u_ub,
        obs_comps=cfg["opc"].get("observed_components", ()),
        method=cfg["optim"]["method"],
        optim_options=cfg["optim"]["options"],
        min_v2v_dist=cfg["opc"]["min_inter_vehicle_distance"],
        max_v2v_dist=cfg["opc"]["max_inter_vehicle_distance"],
    )

    min_problem = cooperative_localization.CooperativeLocalizingOPC(mdl, opts)
    anim = utils.anim_utils.Animated3DTrajectory(mdl.n_robots)

    # -----------------------Generate initial trajectory------------------------
    leader_trajectory = cfg["session"]["leader_trajectory"]
    dt = leader_trajectory["sample_period"]
    timestamps = leader_trajectory["timestamps"]
    t_sample = np.arange(0, timestamps[-1], dt)
    waypoints = np.asarray(leader_trajectory["waypoints"])
    init_positions = np.asarray(cfg["session"]["initial_positions"])
    quadrotor_mass = cfg["model"]["robot_mass"]

    x_leader, u_leader = generate_leader_trajectory(
        timestamps, waypoints, t_sample, quadrotor_mass, dt
    )

    # -----------------Setup initial conditions and data saving-----------------
    sim_steps = cfg["sim"]["steps"]
    time = t_sample[0:sim_steps]
    x = np.zeros((sim_steps, mdl.nx))
    u = np.zeros((sim_steps, mdl.nu))

    n_ic = len(init_positions)
    if n_ic != mdl.n_robots:
        print(f"Incorrect number of initial positions {n_ic} for {mdl.n_robots} robots")
        sys.exit(1)

    x[0, :] = np.concatenate(
        [x_leader[0, :]] + [np.r_[it, q_eqm, v_eqm] for it in init_positions[1:]]
    )

    input_steps, nu = u_leader.shape
    if nu != mdl.robot_nu:
        print(f"Incorrect quad trajectory input size {nu} vs {mdl.robot_nu}")
        sys.exit(1)
    if sim_steps + window > input_steps:
        u_leader_tmp = np.array(u_leader)
        u_leader = np.zeros((sim_steps + window, mdl.robot_nu))
        u_leader[:input_steps, :] = u_leader_tmp
        u_leader[input_steps:, :] = u_leader[-1, :]

    u0 = np.r_[u_leader[0, :], np.tile(u_eqm, mdl.n_robots - 1)]
    u[0, :] = u0

    soln_stats = {
        "status": [],
        "nit": [],
        "fun_hist": [],
        "execution_time": [],
        "constr_violation": [],
        "optimality": [],
    }

    # ----------------------------Run the Simulation----------------------------
    success = False
    try:
        with plt.ion():
            for i in tqdm.trange(1, sim_steps):
                u_leader_0 = u_leader[i : i + window, :]

                u0 = np.hstack([u_leader_0, np.tile(u_eqm, (window, 2))])
                soln = min_problem.minimize(x[i - 1, :], u0, dt)
                soln_u = np.concatenate([u_leader[i, :], soln.x[0, mdl.robot_nu :]])
                u[i, :] = soln_u
                x[i, :] = common.forward_dynamics(
                    mdl.dynamics, x[i - 1, :], soln_u, dt, "euler"
                )

                fun = soln.fun
                status = soln.get("status", -1)
                nit = soln.get("nit", np.nan)
                execution_time = soln.get("execution_time", np.nan)
                constr_violation = float(soln.get("constr_violation", np.nan))
                optimality = soln.get("optimality", np.nan)

                soln_stats["status"].append(status)
                soln_stats["nit"].append(nit)
                soln_stats["execution_time"].append(execution_time)
                soln_stats["constr_violation"].append(constr_violation)
                soln_stats["optimality"].append(optimality)

                fun_hist = np.full(cfg["optim"]["options"]["maxiter"], fun)
                fun_hist[0 : len(soln.fun_hist)] = np.asarray(soln.fun_hist)
                soln_stats["fun_hist"].append(fun_hist)

                anim.annotation = (
                    f"nit: {nit} f(x): {fun:.4}\n $\\Delta$ f(x):"
                    f" {(fun - fun_hist[0]):4g}\nOptimality:"
                    f" {optimality:.4}\nviolation: {constr_violation:.4}"
                )
                plt_data = np.reshape(x[0:i, :], (i, mdl.n_robots, mdl.robot_nx))

                anim.t = time[0:i]
                for idx in range(mdl.n_robots):
                    anim.x[idx] = plt_data[:, idx, 0]
                    anim.y[idx] = plt_data[:, idx, 1]
                    anim.z[idx] = plt_data[:, idx, 2]
                plt.pause(1e-3)
            success = True
    finally:  # Save the data at all costs
        anim.anim.save(cfg["session"].get("video_name", "optimization.mp4"))
        soln_stats = {k: np.asarray(v) for k, v in soln_stats.items()}
        save_name = str(cfg["session"].get("save_name", "optimization_results.npz"))
        if not success:
            save_name = save_name.replace(".npz", ".failed.npz")
        np.savez(save_name, states=x, inputs=u, time=time, **soln_stats)

    figs = {}
    figs[0], ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")

    plt_data = np.reshape(x, (-1, mdl.n_robots, mdl.robot_nx))
    for idx in range(mdl.n_robots):  # Vary vehicles
        ax.plot(
            plt_data[:, idx, 0], plt_data[:, idx, 1], plt_data[:, idx, 2], f"C{idx}"
        )

    plt.show()


if __name__ == "__main__":
    main()
