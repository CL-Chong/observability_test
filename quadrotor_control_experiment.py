import math

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import tqdm

from src.observability_aware_control import planning
from src.observability_aware_control.algorithms import (
    STLOG,
    CooperativeLocalizationOptions,
    CooperativeOPCProblem,
)
from src.observability_aware_control.models import multi_quadrotor
from src.observability_aware_control import utils


# testing list: (X) = bad, (1/2) = not sure, (O) = good
# nonlinear constraints (X)
# log-scaling (1/2)
# det and tr replacements for min eig (X)
# waypoints (1/2 - doesn't break, but doesn't achieve much)
# adaptive dt_stlog (O - continue to refine)
# psd mod (O - less transients than base, but less turning too)

jax.config.update("jax_enable_x64", True)


def main():
    order = 1

    mdl = multi_quadrotor.MultiQuadrotor(3, 1.0)
    win_sz = 20
    dt = 0.2
    stlog = STLOG(mdl, order)
    n_steps = 500
    n_splits = 2
    # adaptive stlog - kicks up if min eig < min_tol, down if min eig > max_tol
    ms = planning.MinimumSnap(
        5, [0, 0, 1, 1], planning.MinimumSnapAlgorithm.CONSTRAINED
    )

    x0 = [np.r_[2.0, 1e-3, 10.0], np.r_[-1e-3, 1.0, 10.0], np.r_[2e-4, -1.0, 10.0]]
    states_traj = []
    inputs_traj = []
    for it in x0:
        p_ref = np.linspace(
            it, it + np.array([n_steps * 2.5 * dt, 0, 0]), n_splits, axis=1
        )
        t_ref = np.linspace(0, n_steps * dt, n_splits)
        pp = ms.generate(p_ref, t_ref)
        traj = pp.to_real_trajectory(1.0, np.r_[0:n_steps] * dt)
        states_traj.append(traj.states)
        inputs_traj.append(traj.inputs)

    states_traj = np.stack(states_traj)
    inputs_traj = np.stack(inputs_traj)

    u_lb = jnp.tile(jnp.r_[0.0, -0.4, -0.4, -2.0], mdl.n_robots)
    u_ub = jnp.tile(jnp.r_[11.0, 0.4, 0.4, 2.0], mdl.n_robots)
    x = np.zeros((n_steps, mdl.nx))
    u = np.zeros((n_steps, mdl.nu))
    x[0, :] = states_traj[:, :, 0].ravel()
    u[0, :] = inputs_traj[:, :, 0].ravel()

    u_leader = inputs_traj[0, :, :]

    obs_comps = (10, 11, 12, 20, 21, 22)
    opts = CooperativeLocalizationOptions(
        window=win_sz,
        id_leader=0,
        lb=u_lb,
        ub=u_ub,
        obs_comps=obs_comps,
        method="trust-constr",
        optim_options={
            "xtol": 1e-1,
            "gtol": 1e-4,
            "disp": False,
            "verbose": 0,
            "maxiter": 150,
        },
        min_v2v_dist=math.sqrt(0.2),
        max_v2v_dist=math.sqrt(10),
    )
    assert opts.optim_options is not None
    min_problem = CooperativeOPCProblem(stlog, opts)
    status = []
    nit = []
    fun_hists = []
    time = np.arange(0, n_steps) * dt

    anim = utils.anim_utils.Animated3DTrajectory(mdl.n_robots)

    with plt.ion():
        for i in tqdm.tqdm(range(1, n_steps)):
            soln = min_problem.minimize(
                x[i - 1, :],
                jnp.broadcast_to(u[i - 1, :], (win_sz, len(u[i - 1, :]))),
                dt,
            )
            soln_u = np.concatenate([u_leader[:, i], soln.x[0, mdl.robot_nu :]])

            status.append(soln.status)
            nit.append(soln.nit)
            fun_hist = np.full(opts.optim_options["maxiter"], np.inf)
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

    return


if __name__ == "__main__":
    main()
