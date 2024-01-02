import math

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import tqdm
from scipy import optimize
from scipy.optimize import NonlinearConstraint

import src.observability_aware_control.models.multi_quadrotor as nummodels
from src.observability_aware_control import planning
from src.observability_aware_control.algorithms import (
    STLOG,
    CooperativeOPCProblem,
    CooperativeLocalizationOptions,
)
from src.observability_aware_control.utils import utils

# testing list: (X) = bad, (1/2) = not sure, (O) = good
# nonlinear constraints (X)
# log-scaling (1/2)
# det and tr replacements for min eig (X)
# waypoints (1/2 - doesn't break, but doesn't achieve much)
# adaptive dt_stlog (O - continue to refine)
# psd mod (O - less transients than base, but less turning too)

jax.config.update("jax_enable_x64", True)


def test(anim=False):
    order_psd = 1

    num_mdl = nummodels.MultiQuadrotor(3, 1.0)
    win_sz = 20
    dt = 0.2
    stlog = STLOG(num_mdl, order_psd)
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
    x0 = states_traj[:, :, 0].ravel()

    u0 = inputs_traj[:, :, 0].ravel()
    u_lb = jnp.tile(jnp.r_[0.0, -0.4, -0.4, -2.0], num_mdl.n_robots)
    u_ub = jnp.tile(jnp.r_[11.0, 0.4, 0.4, 2.0], num_mdl.n_robots)
    x = np.zeros((num_mdl.nx, n_steps))
    x[:, 0] = x0

    u = np.zeros((num_mdl.nu, n_steps))
    u_leader = inputs_traj[0, :, :]
    u[0 : num_mdl.robot_nu, :] = u_leader
    u[:, 0] = u0

    if anim:
        anim, anim_ax = plt.subplots(1, 2)
        plt.ioff()

        anim_data = {
            idx: {"line": [a.plot([], [])[0] for a in anim_ax]}
            for idx in range(num_mdl.n_robots)
        }

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
        min_v2v_dist=np.sqrt(0.2),
        max_v2v_dist=np.sqrt(10),
    )
    assert opts.optim_options is not None
    min_problem = CooperativeOPCProblem(stlog, opts)
    t = 0
    status = []
    nit = []
    fun_hists = []
    time = [t]
    for i in tqdm.tqdm(range(1, n_steps)):
        soln = min_problem.minimize(
            x[:, i - 1],
            jnp.broadcast_to(u[:, i - 1], (win_sz, len(u[:, i - 1]))),
            dt,
        )
        soln_u = np.concatenate([u_leader[:, i], soln.x[0, num_mdl.robot_nu :]])
        tqdm.tqdm.write(
            f"f(x): {soln.fun} Optimality: {soln.optimality} gnorm: {np.linalg.norm(soln.grad)} violation: {soln.constr_violation}"
        )
        status.append(soln.status)
        nit.append(soln.nit)
        fun_hist = np.full(opts.optim_options["maxiter"], np.inf)
        fun_hist[0 : len(soln.fun_hist)] = np.asarray(soln.fun_hist)
        fun_hists.append(np.array(fun_hist))
        u[:, i] = soln_u
        x[:, i] = x[:, i - 1] + dt * min_problem.stlog.model.dynamics(
            x[:, i - 1], soln_u
        )
        t += dt
        time.append(t)

        if anim:
            x_drawable = np.reshape(
                x[:, 0:i], (num_mdl.robot_nx, num_mdl.n_robots, i), order="F"
            )
            for j in range(num_mdl.n_robots):
                anim_data[j]["x"] = x_drawable[0, j, :]
                anim_data[j]["y"] = x_drawable[1, j, :]
                anim_data[j]["z"] = x_drawable[2, j, :]
                anim_data[j].setdefault("t", []).append(t)
                anim_data[j]["line"][0].set_data(anim_data[j]["x"], anim_data[j]["y"])
                anim_data[j]["line"][1].set_data(anim_data[j]["t"], anim_data[j]["z"])
            for a in anim_ax:
                a.relim()
                a.autoscale_view(True, True)
            anim.canvas.draw_idle()
            plt.pause(0.01)

    np.savez(
        "data/optimization_results.npz",
        states=x,
        inputs=u,
        time=time,
        status=status,
        nit=nit,
        fun_hist=np.asarray(fun_hists),
    )

    def plotting_simple(model, x):
        figs = {}
        figs[0], ax = plt.subplots(subplot_kw={"projection": "3d"})
        ax.plot(x[0, :], x[1, :], x[2, :], "C9")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        for idx in range(1, model.n_robots):  # Vary vehicles
            ax.plot(
                x[0 + idx * model.robot_nx, :],
                x[1 + idx * model.robot_nx, :],
                x[2 + idx * model.robot_nx, :],
                f"C{idx}",
            )

        figs[0].savefig("data/stlog_planning.png")
        # figs[0].savefig("stlog_planning_for_go.png")

        # plt.show()

    plotting_simple(num_mdl, x)

    return


print(test(anim=True))
