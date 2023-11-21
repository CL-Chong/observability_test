import math

import casadi as cs
import matplotlib.pyplot as plt
import numpy as np
import tqdm
from scipy import optimize

import src.observability_aware_control.models.autodiff.multi_planar_robot as nummodels
import src.observability_aware_control.models.symbolic.multi_planar_robot as symmodels
from src.observability_aware_control.algorithms.symbolic.algorithms import STLOG
from src.observability_aware_control.utils import utils
from src.observability_aware_control.utils.minimize_problem import MinimizeProblem


def test(anim=False):
    # rng = np.random.default_rng(seed=1000)
    order = 5

    sym_mdl = symmodels.LeaderFollowerRobots(3)
    num_mdl = nummodels.LeaderFollowerRobots(3)
    stlog_cls = STLOG(sym_mdl, order)
    dt = 0.05
    dt_stlog = 0.2
    n_steps = 1900

    x0 = np.array(
        [0.5, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, -5.0, 0.0],
    )
    rot_magnitude = 2
    thrust = 4
    # u0 = np.concatenate(
    #     ([2.0, 0.0], rng.uniform(0, magnitude, (sym_mdl.nu - 2,)))
    # )
    u_leader = [1.0, 0.0]
    u0 = np.concatenate((u_leader, [4.0, -1.0, 4.0, 1.0]))
    u_lb = np.concatenate((u_leader, [0.0, -rot_magnitude, 0.0, -rot_magnitude]))
    u_ub = np.concatenate((u_leader, [thrust, rot_magnitude, thrust, rot_magnitude]))

    x = np.zeros((sym_mdl.nx, n_steps))
    u = np.zeros((sym_mdl.nu, n_steps))
    x[:, 0] = x0
    u[:, 0] = u0

    if anim:
        anim, anim_ax = plt.subplots()
        plt.ioff()

        anim_data = {
            idx: {"line": anim_ax.plot([], [])[0]} for idx in range(sym_mdl.n_robots)
        }

    optim_hist = {}
    for i in tqdm.tqdm(range(1, n_steps)):
        problem = stlog_cls.make_problem(
            x[:, i - 1],
            u[:, i - 1],
            dt_stlog,
            u_lb,
            u_ub,
            log_scale=False,
            omit_leader=True,
        )

        soln = optimize.minimize(**vars(problem))

        u[:, i] = np.concatenate((u_leader, soln.x))
        x[:, i] = x[:, i - 1] + dt * num_mdl.dynamics(x[:, i - 1], u[:, i])

        soln = utils.take_arrays(soln)
        for k, v in soln.items():
            optim_hist.setdefault(k, []).append(v)

        problem = utils.take_arrays(vars(problem))
        for k, v in problem.items():
            optim_hist.setdefault(k, []).append(v)

        if anim:
            x_drawable = np.reshape(
                x[:, 0:i], (sym_mdl.NX, sym_mdl.n_robots, i), order="F"
            )
            for j in range(sym_mdl.n_robots):
                anim_data[j]["x"] = x_drawable[0, j, :]
                anim_data[j]["y"] = x_drawable[1, j, :]
                anim_data[j]["line"].set_data(anim_data[j]["x"], anim_data[j]["y"])
            anim_ax.relim()
            anim_ax.autoscale_view(True, True)
            anim.canvas.draw_idle()
            plt.pause(0.01)

    optim_hist = {k: np.asarray(v) for k, v in optim_hist.items()}
    np.savez("data/optimization_results.npz", states=x, inputs=u, **optim_hist)

    def plotting_simple(model, x):
        figs = {}
        figs[0], ax = plt.subplots()
        ax.plot(x[0, :], x[1, :], "C9")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        for idx in range(1, model.n_robots):  # Vary vehicles
            ax.plot(x[0 + idx * model.NX, :], x[1 + idx * model.NX, :], f"C{idx}")

        figs[0].savefig("data/stlog_planning.png")
        # figs[0].savefig("stlog_planning_for_go.png")

        # plt.show()

    plotting_simple(sym_mdl, x)

    return


print(test(anim=False))
