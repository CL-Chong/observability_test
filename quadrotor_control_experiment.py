import math

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


def main():
    with open("./config/quadrotor_control_experiment.toml", "rb") as fp:
        cfg = tomllib.load(fp)

    mdl = multi_quadrotor.MultiQuadrotor(
        cfg["model"]["n_robots"], cfg["model"]["robot_mass"]
    )

    # -----------------------Generate initial trajectory------------------------
    ms = planning.MinimumSnap(
        cfg["initial_path"]["degree"],
        cfg["initial_path"]["derivative_weights"],
        planning.MinimumSnapAlgorithm.CONSTRAINED,
    )

    p_refs = np.array(cfg["initial_path"]["position_reference"])
    t_ref = np.array(cfg["initial_path"]["time_reference"])
    n_steps = cfg["initial_path"]["n_timesteps"]
    dt = np.diff(t_ref) / n_steps
    t_sample = np.arange(0, n_steps) * dt
    states_traj, inputs_traj = ms.generate_trajectories(
        jnp.tile(t_ref, [cfg["model"]["n_robots"], 1]),
        p_refs,
        jnp.tile(t_sample, [cfg["model"]["n_robots"], 1]),
    )

    # -----------------Setup initial conditions and data saving-----------------
    sim_steps = cfg["sim"]["steps"]
    time = t_sample[0:sim_steps]
    x = np.zeros((sim_steps, mdl.nx))
    u = np.zeros((sim_steps, mdl.nu))
    x[0, :] = states_traj[:, :, 0].ravel()
    u[0, :] = inputs_traj[:, :, 0].ravel()
    u_leader = inputs_traj[0, :, :]
    status = []
    nit = []
    fun_hists = []

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
        obs_comps=cfg["opc"]["observed_components"],
        method=cfg["optim"]["method"],
        optim_options=cfg["optim"]["options"],
        min_v2v_dist=cfg["opc"]["min_inter_vehicle_distance"],
        max_v2v_dist=cfg["opc"]["max_inter_vehicle_distance"],
    )

    min_problem = CooperativeOPCProblem(stlog, opts)
    anim = utils.anim_utils.Animated3DTrajectory(mdl.n_robots)

    # ----------------------------Run the Simulation----------------------------
    with plt.ion():
        for i in tqdm.tqdm(range(1, sim_steps)):
            soln = min_problem.minimize(
                x[i - 1, :],
                jnp.broadcast_to(u[i - 1, :], (window, len(u[i - 1, :]))),
                dt,
            )
            soln_u = np.concatenate([u_leader[:, i], soln.x[0, mdl.robot_nu :]])

            status.append(soln.status)
            nit.append(soln.nit)
            fun_hist = np.full(cfg["optim"]["options"]["maxiter"], np.inf)
            fun_hist[0 : len(soln.fun_hist)] = np.asarray(soln.fun_hist)
            fun_hists.append(np.array(fun_hist))
            u[i, :] = soln_u
            x[i, :] = x[i - 1, :] + dt * min_problem.stlog.model.dynamics(
                x[i - 1, :], soln_u
            )

            plt_data = np.reshape(x[0:i, :], (i, mdl.n_robots, mdl.robot_nx))
            anim.annotation = (
                f"f(x): {soln.fun:.4}\nOptimality: {soln.optimality:.4}\ngnorm:"
                f" {np.linalg.norm(soln.grad):.4}\nviolation: {soln.constr_violation:.4}"
            )
            anim.t = time[0:i]
            for idx in range(mdl.n_robots):
                anim.x[idx] = plt_data[:, idx, 0]
                anim.y[idx] = plt_data[:, idx, 1]
                anim.z[idx] = plt_data[:, idx, 2]
            plt.pause(1e-3)

    np.savez(
        "data/optimization_results.npz",
        states=x,
        inputs=u,
        time=time,
        status=status,
        nit=nit,
        fun_hist=np.asarray(fun_hists),
    )

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
