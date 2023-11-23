import math

import casadi as cs
import numpy as np
from scipy import optimize

from src.observability_aware_control.utils import utils
from src.observability_aware_control.utils.minimize_problem import MinimizeProblem


class STLOG:
    # Constructor: construct STLOG as cs MX given mdl and order (in t)
    def __init__(self, mdl, order, is_psd=False):
        self._nx = mdl.nx
        self._nu = mdl.nu
        self._order = order
        self._NX = mdl.NX
        self._NU = mdl.NU

        self._symbols = {
            "x": cs.MX.sym("x", mdl.nx),
            "u": cs.MX.sym("u", mdl.nu),
            "t": cs.MX.sym("t"),
        }

        # calculate self._stlog
        self._stlog = cs.MX.zeros(mdl.nx, mdl.nx)
        # calculate L_f^k h and D L_f^k h for k = 0, ..., order
        lh_store = []
        lh_store.append(mdl.observation(self._symbols["x"]))
        dlh_store = []
        dlh_store.append(cs.jacobian(lh_store[0], self._symbols["x"]))
        for l in range(0, order):
            lh_store.append(
                cs.jtimes(
                    lh_store[l],
                    self._symbols["x"],
                    mdl.dynamics(self._symbols["x"], self._symbols["u"]),
                )
            )
            dlh_store.append(cs.jacobian(lh_store[l + 1], self._symbols["x"]))

        # summing D L_f^k terms to obtain STLOG
        if is_psd:
            for a in range(0, order + 1):
                for b in range(0, order + 1):
                    self._stlog += (
                        (self._symbols["t"] ** (a + b + 1))
                        / (math.factorial(a) * math.factorial(b) * (a + b + 1))
                    ) * cs.mtimes(
                        dlh_store[a].T,
                        dlh_store[b],
                    )
        else:
            for l in range(0, order):
                for k in range(0, l + 1):
                    self._stlog += (
                        (self._symbols["t"] ** (l + 1))
                        / ((l + 1) * math.factorial(k) * math.factorial(l - k))
                    ) * cs.mtimes(
                        dlh_store[k].T,
                        dlh_store[l - k],
                    )

        # self._fun computes STLOG as a cs function object, which can be called with self._fun()(x,u,t)
        self._fun = cs.Function(
            "stlog_fun",
            [self._symbols["x"], self._symbols["u"], self._symbols["t"]],
            [self._stlog],
            # {
            #     "compiler": "shell",
            #     "jit": True,
            #     "jit_options": {"compiler": "gcc", "flags": ["-O3"]},
            # },
        )

    @property
    def order(self):
        return self._order

    # self.fun outputs STLOG as a cs function object, which can be called with self.fun()(x,u,t)
    # def fun(self):
    #     return cs.Function(
    #         "stlog_fun",
    #         [self._symbols["x"], self._symbols["u"], self._symbols["t"]],
    #         [self._stlog],
    #     )

    # self.objective outputs -(min singular value) of stlog.
    def objective(self, x=None, t=None):
        if x is None and t is None:

            def inner_objective(x1, u1, t1):
                return -1 * (np.linalg.norm(self._fun(x1, u1, t1), -2))

        elif x is None:

            def inner_objective(x1, u1):
                return -1 * (np.linalg.norm(self._fun(x1, u1, t), -2))

        elif t is None:

            def inner_objective(u1, t1):
                return -1 * (np.linalg.norm(self._fun(x, u1, t1), -2))

        else:

            def inner_objective(u1):
                return -1 * (np.linalg.norm(self._fun(x, u1, t), -2))

        return inner_objective

    def make_problem(self, x0, u0, t, u_lb, u_ub, log_scale=False, omit_leader=False):
        # log_scale still in testing
        # if max(abs(np.concatenate((u0, u_lb, u_ub)))) * t > 1.0:
        #     print(
        #         f"Warning: max(|u*t|) = {max(abs(np.concatenate((u0,u_lb,u_ub))))*t} > 1. STLOG convergence is not guaranteed."
        #     )
        obj_fun_primitive = self.objective(x=x0, t=t)
        if omit_leader:

            def obj_fun_omit_leader(u_follower):
                return obj_fun_primitive(np.concatenate((u0[0 : self._NU], u_follower)))

            obj_fun = obj_fun_omit_leader
            u0 = u0[self._NU :]
            u_lb = u_lb[self._NU :]
            u_ub = u_ub[self._NU :]
        else:
            obj_fun = obj_fun_primitive

        if log_scale:
            problem = MinimizeProblem(lambda arg: -np.log(1e-4 - obj_fun(arg)), u0)
        else:
            problem = MinimizeProblem(obj_fun, u0)
        problem.bounds = optimize.Bounds(u_lb, u_ub)
        problem.method = "trust-constr"
        problem.options = {
            "xtol": 1e-4,
            "gtol": 1e-8,
            "disp": False,
            "verbose": 0,
            "maxiter": 100,
        }
        return problem


def stlog(sys, order, is_psd=False):
    x = cs.MX.sym("x", sys.nx)
    u = cs.MX.sym("u", sys.nu)
    T = cs.MX.sym("T")
    stlog = cs.MX.zeros(sys.nx, sys.nx)
    lh_store = []
    lh_store.append(sys.observation(x))
    dlh_store = []
    dlh_store.append(cs.jacobian(lh_store[0], x))

    for l in range(0, order):
        lh_store.append(cs.jtimes(lh_store[l], x, sys.dynamics(x, u)))
        dlh_store.append(cs.jacobian(lh_store[l + 1], x))
    if is_psd:
        for a in range(0, order + 1):
            for b in range(0, order + 1):
                stlog += (
                    (T ** (a + b + 1))
                    / (math.factorial(a) * math.factorial(b) * (a + b + 1))
                ) * cs.mtimes(
                    dlh_store[a].T,
                    dlh_store[b],
                )
    else:
        for l in range(0, order):
            for k in range(0, l + 1):
                stlog += (
                    (T ** (l + 1))
                    / ((l + 1) * math.factorial(k) * math.factorial(l - k))
                ) * cs.mtimes(
                    dlh_store[k].T,
                    dlh_store[l - k],
                )

    stlog_fun = cs.Function("stlog_fun", [x, u, T], [stlog])

    return stlog_fun
