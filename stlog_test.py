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


def test():
    # rng = np.random.default_rng(seed=1000)
    order = 5

    sym_mdl = symmodels.LeaderFollowerRobots(3)
    num_mdl = nummodels.LeaderFollowerRobots(3)
    stlog_cls = STLOG(sym_mdl, order)
    dt = 0.01
    dt_stlog = 0.15
    n_steps = 10

    x0 = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, -1.0, 0.0])
    magnitude = 5
    # u0 = np.concatenate(
    #     ([2.0, 0.0], rng.uniform(0, magnitude, (sym_mdl.nu - 2,)))
    # )
    u_leader = [2.0, 0.0]
    u0 = np.concatenate((u_leader, [5, -5, 5, 5]))
    u_lb = np.concatenate((u_leader, [0.0, -magnitude, 0.0, -magnitude]))
    u_ub = np.concatenate((u_leader, magnitude * np.ones((sym_mdl.nu - 2,))))

    x = np.zeros((sym_mdl.nx, n_steps))
    u = np.zeros((sym_mdl.nu, n_steps))
    x[:, 0] = x0
    u[:, 0] = u0
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
        u[:, i] = np.concatenate((u_leader, optimize.minimize(**vars(problem)).x))
        x[:, i] = x[:, i - 1] + dt * num_mdl.dynamics(x[:, i - 1], u[:, i])

    def plotting_simple(model, x):
        figs = {}
        figs[0], ax = plt.subplots()
        ax.plot(x[0, :], x[1, :], "C9")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        for idx in range(1, model.n_robots):  # Vary vehicles
            ax.plot(x[0 + idx * model.NX, :], x[1 + idx * model.NX, :], f"C{idx}")

        figs[0].savefig("data/stlog_planning.png")

        plt.show()

    plotting_simple(sym_mdl, x)

    return


print(test())
