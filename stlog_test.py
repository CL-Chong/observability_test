import math

import casadi as cs
import numpy as np
from scipy import optimize

import src.observability_aware_control.models.symbolic.multi_planar_robot as models
from src.observability_aware_control.algorithms.symbolic.algorithms import STLOG, stlog
from src.observability_aware_control.utils import utils
from src.observability_aware_control.utils.minimize_problem import MinimizeProblem


def test():
    rng = np.random.default_rng(seed=1000)
    order = 5

    sym_mdl = models.LeaderFollowerRobots(3)
    stlog_cls = STLOG(sym_mdl, order)
    # stlog_old_fun = stlog(sym_mdl, order)

    x0 = np.array([0.0, 0.0, 0.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0])
    magnitude = 5
    u0 = np.concatenate(
        ([2.0, 0.0], rng.uniform(-magnitude, magnitude, (sym_mdl.nu - 2,)))
    )
    u_lb = np.concatenate(([2.0, 0.0], -magnitude * np.ones((sym_mdl.nu - 2,))))
    u_ub = np.concatenate(([2.0, 0.0], magnitude * np.ones((sym_mdl.nu - 2,))))
    t0 = 0.15

    # print(np.linalg.eig(stlog_cls.fun()(x0, u0, t0)))
    print(stlog_cls.objective(x0, t0)(u0))

    problem = stlog_cls.make_problem(x0, u0, t0, u_lb, u_ub)

    return optimize.minimize(**vars(problem))


print(test())
