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

# from scipy.optimize import NonlinearConstraint


def test(anim=False):
    rng = np.random.default_rng(seed=1000)
    order = 5

    sym_mdl = symmodels.LeaderFollowerRobots(3)
    num_mdl = nummodels.LeaderFollowerRobots(3)
    stlog_cls = STLOG(sym_mdl, order)
    dt = 0.05
    dt_stlog = 0.24
    n_steps = 2000
    opt_tol = 1e-7
    waypt_ratio = 1
    kick_eps = 0.1

    x0 = np.array(
        [0.5, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, -5.0, 0.0],
    )
    rot_magnitude = 2
    thrust = 4
    u_leader = [1.0, 0.0]
    u0 = np.concatenate((u_leader, [4.0, -1.0, 4.0, 1.0]))
    u_lb = np.concatenate((u_leader, [0.0, -rot_magnitude, 0.0, -rot_magnitude]))
    u_ub = np.concatenate((u_leader, [thrust, rot_magnitude, thrust, rot_magnitude]))
    # failed_dict = np.load("failed_optimization_results.npz")
    # pre_mortem_index = 2000
    # fatal_index = 2630
    # x0 = failed_dict["states"][:, pre_mortem_index]
    # u0 = failed_dict["inputs"][:, pre_mortem_index]
    # print(x0)
    # print(num_mdl.observation(x0))
    # print(np.linalg.eig(stlog_cls._fun(x0, u0, dt_stlog)))
    # return

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

    # optim_hist = {}
    # def con(a):
    #     return (a[1] - a[3]) ** 2

    # nlc = NonlinearConstraint(con, 1e-6, +np.inf)

    for i in tqdm.tqdm(range(1, n_steps)):
        if i % waypt_ratio == 0:
            problem = stlog_cls.make_problem(
                x[:, i - 1],
                u[:, i - 1]
                + np.concatenate(
                    ([0, 0], rng.uniform(low=-kick_eps, high=kick_eps, size=(4,)))
                ),
                dt_stlog,
                u_lb,
                u_ub,
                log_scale=False,
                omit_leader=True,
            )

            soln = optimize.minimize(**vars(problem))
            if abs(soln.fun) < opt_tol or soln.nit < 10:
                print(
                    f"Warning: Premature termination with min eigenvalue {abs(soln.fun)} at {soln.nit} iterations. Brute optimization initiated."
                )
                defn = vars(problem)
                x0, fval, grid, Jout = optimize.brute(
                    defn["fun"],
                    tuple(zip(u_lb[2:], u_ub[2:])),
                    Ns=5,
                    full_output=True,
                    finish=None,
                )

                u[:, i] = np.concatenate((u_leader, x0))
                print(f"Brute optimization returns min eigenvalue {abs(fval)}.")
            else:
                u[:, i] = np.concatenate((u_leader, soln.x))
        else:
            u[:, i] = u[:, i - 1]

        x[:, i] = x[:, i - 1] + dt * num_mdl.dynamics(x[:, i - 1], u[:, i])
        x[::-3, i] = np.arctan2(np.sin(x[::-3, i]), np.cos(x[::-3, i]))

        # soln = utils.take_arrays(soln)
        # for k, v in soln.items():
        #     optim_hist.setdefault(k, []).append(v)

        # problem = utils.take_arrays(vars(problem))
        # for k, v in problem.items():
        #     optim_hist.setdefault(k, []).append(v)

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

    # optim_hist = {k: np.asarray(v) for k, v in optim_hist.items()}
    # np.savez("data/optimization_results.npz", states=x, inputs=u, **optim_hist)

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


print(test(anim=True))
