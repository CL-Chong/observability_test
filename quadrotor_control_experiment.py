import math
import sys

import jax
import jax.numpy as jnp
import jax.experimental.compilation_cache.compilation_cache as cc
import matplotlib.pyplot as plt
import numpy as np
import tqdm
import tomllib

import observability_aware_control.algorithms.misc.trajectory_generation as planning
from observability_aware_control.algorithms import (
    STLOG,
    CooperativeLocalizationOptions,
    CooperativeOPCProblem,
    forward_dynamics,
)
from observability_aware_control.models import multi_quadrotor
from observability_aware_control import utils


# testing list: (X) = bad, (1/2) = not sure, (O) = good
# nonlinear constraints (X)
# log-scaling (1/2)
# det and tr replacements for min eig (X)
# waypoints (1/2 - doesn't break, but doesn't achieve much)
# adaptive dt_stlog (O - continue to refine)
# psd mod (O - less transients than base, but less turning too)

cc.initialize_cache("./.cache")

jax.config.update("jax_enable_x64", True)


u_eqm = np.r_[9.81, 0.0, 0.0, 0.0]
q_eqm = np.r_[np.zeros(3), 1.0]
v_eqm = np.zeros(3)


def main():
    with open("./config/quadrotor_control_experiment.toml", "rb") as fp:
        cfg = tomllib.load(fp)

    mdl = multi_quadrotor.MultiQuadrotor(
        cfg["model"]["n_robots"], cfg["model"]["robot_mass"], has_odom=True
    )

    # ---------------------------Setup the Optimizer----------------------------
    stlog = STLOG(mdl, cfg["stlog"]["order"])

    window = cfg["opc"]["window_size"]
    u_lb = jnp.tile(jnp.array(cfg["optim"]["lb"]), (window, mdl.n_robots))
    u_ub = jnp.tile(jnp.array(cfg["optim"]["ub"]), (window, mdl.n_robots))
    opts = CooperativeLocalizationOptions(
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

    min_problem = CooperativeOPCProblem(stlog, opts)
    anim = utils.anim_utils.Animated3DTrajectory(mdl.n_robots)

    # -----------------------Generate initial trajectory------------------------
    trajectory_data = np.load(cfg["session"]["leader_trajectory"])
    u_leader = trajectory_data["inputs"]
    t_sample = trajectory_data["time"]
    dt = np.diff(t_sample)[0]

    # -----------------Setup initial conditions and data saving-----------------
    sim_steps = cfg["sim"]["steps"]
    time = t_sample[0:sim_steps]
    x = np.zeros((sim_steps, mdl.nx))
    u = np.zeros((sim_steps, mdl.nu))

    n_ic = len(cfg["session"]["initial_positions"])
    if n_ic != mdl.n_robots:
        print(f"Incorrect number of initial positions {n_ic} for {mdl.n_robots} robots")
        sys.exit(1)

    x[0, :] = np.concatenate(
        [np.r_[it, q_eqm, v_eqm] for it in cfg["session"]["initial_positions"]]
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

    u0 = np.r_[u_leader[0, :], np.tile(u_eqm, 2)]
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
                u_leader_0 = u_leader[i - 1 : i - 1 + window, :]

                # u0 = jnp.hstack([u_leader_0, np.tile(u_eqm, (window, 2))])
                u0 = jnp.tile(u_leader_0, (1, 3))
                soln = min_problem.minimize(x[i - 1, :], u0, dt)
                soln_u = np.concatenate([u_leader[i - 1, :], soln.x[0, mdl.robot_nu :]])
                u[i, :] = soln_u
                x[i, :] = forward_dynamics(
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

                anim.annotation = f"nit: {nit} f(x): {fun:.4}\n $\\Delta$ f(x): {(fun - fun_hist[0]):4g}\nOptimality: {optimality:.4}\nviolation: {constr_violation:.4}"
                plt_data = np.reshape(x[0:i, :], (i, mdl.n_robots, mdl.robot_nx))

                anim.t = time[0:i]
                for idx in range(mdl.n_robots):
                    anim.x[idx] = plt_data[:, idx, 0]
                    anim.y[idx] = plt_data[:, idx, 1]
                    anim.z[idx] = plt_data[:, idx, 2]
                plt.pause(1e-3)
            success = True
    finally:  # Save the data at all costs
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
