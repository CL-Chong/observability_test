import dataclasses
import functools
from typing import Any, Callable, Dict, Optional

import casadi as cs
import numpy as np
from scipy import optimize, special

from src.observability_aware_control.utils.minimize_problem import MinimizeProblem


def _default_stlog_metric(x):
    return -np.linalg.norm(x, -2)


@dataclasses.dataclass
class STLOGOptions:
    is_psd: bool = dataclasses.field(default=False)
    function_opts: Optional[Dict[str, Any]] = dataclasses.field(default=None)
    metric: Callable = dataclasses.field(default=_default_stlog_metric)


class STLOG:
    # Constructor: construct STLOG as cs MX given mdl and order (in t)
    def __init__(self, mdl, order, opts: STLOGOptions):
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
        self._stlog = create_stlog(self._symbols, self._order, opts.is_psd, mdl)

        self._metric = opts.metric

        function_opts = {} if opts.function_opts is None else opts.function_opts
        # self._fun computes STLOG as a cs function object, which can be called with self._fun()(x,u,t)
        self._fun = cs.Function(
            "stlog_fun",
            [self._symbols["x"], self._symbols["u"], self._symbols["t"]],
            [self._stlog],
            function_opts,
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
    def objective(self, x_fix=None, t_fix=None):
        def evaluate_metric(u, x, t):
            return self._metric(self._fun(x, u, t))

        bound_args = {}
        if x_fix is not None:
            bound_args["x"] = x_fix
        if t_fix is not None:
            bound_args["t"] = t_fix

        if bound_args:
            return functools.partial(evaluate_metric, **bound_args)
        return evaluate_metric

    def make_problem(self, x0, u0, t, u_lb, u_ub, log_scale=False, omit_leader=False):
        # log_scale still in testing
        # if max(abs(np.concatenate((u0, u_lb, u_ub)))) * t > 1.0:
        #     print(
        #         f"Warning: max(|u*t|) = {max(abs(np.concatenate((u0,u_lb,u_ub))))*t} > 1. STLOG convergence is not guaranteed."
        #     )
        obj_fun_primitive = self.objective(x_fix=x0, t_fix=t)
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


def create_stlog(symbols, order, is_psd=False, mdl=None):

    if order == 0:
        raise ValueError("Zeroth-order STLOG is meaningless")

    try:
        fcn = symbols["dx"]
        hfcn = symbols["y"]
    except KeyError as exc:
        if mdl is None:
            raise ValueError(
                "If no precomputed observation expression is given, the system model"
                " must be provided to define the observation"
            ) from exc
        fcn = mdl.dynamics(symbols["x"], symbols["u"])
        hfcn = mdl.observation(symbols["x"])

    # calculate L_f^k h and D L_f^k h for k = 0, ..., order
    lh = hfcn  # zeroth-order lie derivative is the function (observation model) itself
    dlh_store = [cs.jacobian(lh, symbols["x"])]  # zeroth-order lie gradient
    for _ in range(0, order):
        # Compute the lie derivative of the next order
        lh = cs.jtimes(lh, symbols["x"], fcn)
        # Compute lie gradient for the matching lie derivative
        dlh_store.append(cs.jacobian(lh, symbols["x"]))

    # up to the (order+1)-th factorial
    facts = special.factorial(np.arange(0, order + 1), exact=True)

    assert isinstance(facts, np.ndarray)

    # Define two ways to sum lie-gradients, one that ensures positive-definiteness of
    # the resulting Gramian, and one that does not
    def psd_stlog(a, b):
        coeff = symbols["t"] ** (a + b + 1) / (facts[a] * facts[b] * (a + b + 1))
        return coeff * (dlh_store[a].T @ dlh_store[b])

    def stlog(j, k):
        coeff = symbols["t"] ** (j + 1) / ((j + 1) * facts[k] * facts[j - k])
        return coeff * (dlh_store[k].T @ dlh_store[j - k])

    if is_psd:
        return sum(psd_stlog(a, b) for a in range(order + 1) for b in range(order + 1))
    else:
        return sum(stlog(l, k) for l in range(order) for k in range(l + 1))
