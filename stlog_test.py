import math

import casadi as cs
import matplotlib.pyplot as plt
import numpy as np
import tqdm
from scipy import optimize
from scipy.optimize import NonlinearConstraint

import src.observability_aware_control.models.autodiff.multi_planar_robot as nummodels
import src.observability_aware_control.models.symbolic.multi_planar_robot as symmodels
from src.observability_aware_control.algorithms.symbolic.algorithms import STLOG
from src.observability_aware_control.utils import utils
from src.observability_aware_control.utils.minimize_problem import MinimizeProblem

# testing list: (X) = bad, (1/2) = not sure, (O) = good
# nonlinear constraints (X)
# log-scaling (1/2)
# det and tr replacements for min eig (X)
# waypoints (1/2 - doesn't break, but doesn't achieve much)
# adaptive dt_stlog (O - continue to refine)
# psd mod (O - less transients than base, but less turning too)


def test(anim=False):
    rng = np.random.default_rng(seed=1000)
    order = 5
    order_psd = 3

    sym_mdl = symmodels.LeaderFollowerRobots(3)
    num_mdl = nummodels.LeaderFollowerRobots(3)
    stlog_cls = STLOG(sym_mdl, order, is_psd=False)
    stlog_psd_cls = STLOG(sym_mdl, order_psd, is_psd=True)
    dt = 0.05
    dt_stlog = 0.2
    n_steps = 4000
    # adaptive stlog - kicks up if min eig < min_tol, down if min eig > max_tol
    min_tol = 1e-6
    max_tol = 1e-3
    stlog_kick = 0.001  # kick size - in testing
    switch_val = 1.01  # if dt_stlog * max(ub) > switch_val, switches to psd

    waypt_ratio = 1

    kick_eps = 0.1  # random kick to optimization IC

    x0 = np.array(
        [0.5, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, -5.0, 0.0],
    )
    rot_magnitude = 2
    min_thurst = 0.0
    max_thrust = 4.0
    u_leader = [1.0, 0.0]
    u0 = np.concatenate((u_leader, [4.0, -1.0, 4.0, 1.0]))
    u_lb = np.concatenate(
        (u_leader, [min_thurst, -rot_magnitude, min_thurst, -rot_magnitude])
    )
    u_ub = np.concatenate(
        (u_leader, [max_thrust, rot_magnitude, max_thrust, rot_magnitude])
    )
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
    #     return np.array([a[1] / a[0], a[3] / a[2]])

    # nlc = NonlinearConstraint(con, [-1e4, -1e4], [1e4, 1e4])
    dt_stlog_running = np.array(dt_stlog, copy=True)
    for i in tqdm.tqdm(range(1, n_steps)):
        if i % waypt_ratio == 0:
            while True:
                if dt_stlog * np.max(u_ub) < switch_val:
                    problem = stlog_cls.make_problem(
                        x[:, i - 1],
                        u[:, i - 1]
                        + np.concatenate(
                            (
                                [0, 0],
                                rng.uniform(low=-kick_eps, high=kick_eps, size=(4,)),
                            )
                        ),
                        dt_stlog_running,
                        u_lb,
                        u_ub,
                        log_scale=False,
                        omit_leader=True,
                    )
                else:
                    problem = stlog_psd_cls.make_problem(
                        x[:, i - 1],
                        u[:, i - 1]
                        + np.concatenate(
                            (
                                [0, 0],
                                rng.uniform(low=-kick_eps, high=kick_eps, size=(4,)),
                            )
                        ),
                        dt_stlog_running,
                        u_lb,
                        u_ub,
                        log_scale=False,
                        omit_leader=True,
                    )
                # problem.constraints = nlc
                # if i > 1:
                #     problem.options = {
                #         "xtol": 1e-3,
                #         "gtol": soln.fun * 1e-3,
                #         "disp": False,
                #         "verbose": 0,
                #         "maxiter": 100,
                #     }
                soln = optimize.minimize(**vars(problem))
                if abs(soln.fun) < min_tol:
                    print(
                        f"Reject solution. dt_stlog = {str(dt_stlog_running)}, u = {soln.x}, f = {abs(soln.fun)}. Increase dt_stlog."
                    )
                    dt_stlog_running += stlog_kick
                elif abs(soln.fun) > max_tol:
                    print(
                        f"Reject solution. dt_stlog = {str(dt_stlog_running)}, u = {soln.x}, f = {abs(soln.fun)}. Decrease dt_stlog."
                    )
                    dt_stlog_running -= stlog_kick
                else:
                    u[:, i] = np.concatenate((u_leader, soln.x))
                    break
        else:
            u[:, i] = u[:, i - 1]

        x[:, i] = (
            x[:, i - 1]
            + dt * num_mdl.dynamics(x[:, i - 1], u[:, i])
            # + np.sqrt(dt)
            # * np.concatenate(([0.0, 0.0, 0.0], rng.standard_normal(size=(6,))))
        )

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
