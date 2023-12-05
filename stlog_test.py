import math

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import tqdm
from scipy import optimize
from scipy.optimize import NonlinearConstraint

import src.observability_aware_control.models.multi_robot as nummodels
from src.observability_aware_control.algorithms.algorithms import (
    STLOG,
    STLOGMinimizeProblem,
    STLOGOptions,
)
from src.observability_aware_control.optimize import minimize
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

    num_mdl = nummodels.LeaderFollowerRobots(3)
    win_sz = 5
    stlog = STLOG(num_mdl, order_psd, components=(4, 5, 6, 8, 9, 10))
    dt = 0.05
    dt_stlog = 0.2
    n_steps = 4000
    # adaptive stlog - kicks up if min eig < min_tol, down if min eig > max_tol
    x0 = np.array(
        [0.5, 0.0, 10.0, 0.0, 0.0, 5.0, 10.0, 0.0, 0.0, -5.0, 10.0, 0.0],
    )
    rot_magnitude = 2
    min_thurst = 0.0
    max_thrust = 4.0
    min_rise = -1
    max_rise = 1
    u_leader = np.array([1.0, 0.0, 0.0])
    u0 = np.concatenate((u_leader, [4.0, -1.0, 0.0, 4.0, 1.0, 0.0]))
    u_lb = np.concatenate(
        (
            u_leader,
            [
                min_thurst,
                min_rise,
                -rot_magnitude,
                min_thurst,
                min_rise,
                -rot_magnitude,
            ],
        )
    )
    u_ub = np.concatenate(
        (
            u_leader,
            [
                max_thrust,
                max_rise,
                rot_magnitude,
                max_thrust,
                max_rise,
                rot_magnitude,
            ],
        )
    )

    x = np.zeros((num_mdl.nx, n_steps))
    x[:, 0] = x0

    u = np.zeros((num_mdl.nu, n_steps))
    u[0 : num_mdl.NU, :] = u_leader[:, None]
    u[:, 0] = u0

    if anim:
        anim, anim_ax = plt.subplots()
        plt.ioff()

        anim_data = {
            idx: {"line": anim_ax.plot([], [])[0]} for idx in range(num_mdl.n_robots)
        }

    # optim_hist = {}
    # def con(a):
    #     return np.array([a[1] / a[0], a[3] / a[2]])

    # nlc = NonlinearConstraint(con, [-1e4, -1e4], [1e4, 1e4])

    u_leader = np.tile(u_leader[..., None], [1, win_sz])
    min_problem = STLOGMinimizeProblem(stlog, STLOGOptions(dt=dt_stlog, window=win_sz))
    for i in tqdm.tqdm(range(1, n_steps)):
        problem = min_problem.make_problem(
            x[:, i - 1],
            jnp.broadcast_to(u[:, i - 1], (win_sz, len(u[:, i - 1]))),
            dt,
            u_lb,
            u_ub,
            id_const=(0, 1, 2),
        )

        soln = minimize(problem)
        soln_u = soln.x[0, :]
        u[:, i] = soln_u
        x[:, i] = min_problem.forward_dynamics(x[:, i - 1], soln_u, dt)

        x[::-4, i] = utils.wrap_to_pi(x[::-4, i])

        if anim:
            x_drawable = np.reshape(
                x[:, 0:i], (num_mdl.NX, num_mdl.n_robots, i), order="F"
            )
            for j in range(num_mdl.n_robots):
                anim_data[j]["x"] = x_drawable[0, j, :]
                anim_data[j]["y"] = x_drawable[1, j, :]
                anim_data[j]["line"].set_data(anim_data[j]["x"], anim_data[j]["y"])
            anim_ax.relim()
            anim_ax.autoscale_view(True, True)
            anim.canvas.draw_idle()
            plt.pause(0.01)

    np.savez("data/optimization_results.npz", states=x, inputs=u)

    def plotting_simple(model, x):
        figs = {}
        figs[0], ax = plt.subplots(subplot_kw={"projection": "3d"})
        ax.plot(x[0, :], x[1, :], x[2, :], "C9")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        for idx in range(1, model.n_robots):  # Vary vehicles
            ax.plot(
                x[0 + idx * model.NX, :],
                x[1 + idx * model.NX, :],
                x[2 + idx * model.NX, :],
                f"C{idx}",
            )
        ax.set_zlim([8, 12])

        figs[0].savefig("data/stlog_planning.png")
        # figs[0].savefig("stlog_planning_for_go.png")

        # plt.show()

    plotting_simple(num_mdl, x)

    return


print(test(anim=True))
