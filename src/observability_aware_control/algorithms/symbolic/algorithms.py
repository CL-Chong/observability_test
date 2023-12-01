import dataclasses
import functools
from typing import Any, Callable, Dict, Optional

import casadi as cs
import numpy as np
from scipy import optimize, special

from src.observability_aware_control.optimize import MinimizeProblem


def _default_stlog_metric(x):
    return -np.linalg.norm(x, -2)


@dataclasses.dataclass
class STLOGOptions:
    function_opts: Optional[Dict[str, Any]] = dataclasses.field(default=None)
    metric: Callable = dataclasses.field(default=_default_stlog_metric)
    window: int = dataclasses.field(default=1)


def _scan(f, init, xs):
    carry = init
    ys = []
    for idx in range(xs.shape[1]):
        carry, y = f(carry, xs[:, idx])
        ys.append(y)
    return cs.horzcat(*ys)


def _symsolve_sigma(symbols, mdl):
    x0 = symbols["x"]
    dt = symbols["t"]

    def _update(x_op, u):
        dx = mdl.dynamics(x_op, u)
        return x_op + dt * dx, x_op

    x = _scan(_update, init=x0, xs=symbols["us"])

    return x


class STLOG:
    # Constructor: construct STLOG as cs MX given mdl and order (in t)
    def __init__(self, mdl, order, opts: STLOGOptions):
        self._nx = mdl.nx
        self._nu = mdl.nu
        self._order = order
        self._window = opts.window
        self._NX = mdl.NX
        self._NU = mdl.NU

        self._symbols = {
            "x": cs.MX.sym("x", mdl.nx),
            "u": cs.MX.sym("u", mdl.nu),
            "t": cs.MX.sym("t"),
        }
        function_opts = {} if opts.function_opts is None else opts.function_opts

        # calculate self._stlog
        if opts.window > 1:
            self._symbols["us"] = cs.MX.sym("us", (mdl.nu, opts.window))
            self._mdl_predict = _symsolve_sigma(self._symbols, mdl)
            self._mdl_predict_fun = cs.Function(
                "predict_fun",
                [self._symbols["x"], self._symbols["us"], self._symbols["t"]],
                [self._mdl_predict],
                function_opts,
            )

        self._stlog = create_stlog(self._symbols, self._order, mdl)

        self._metric = opts.metric

        # self._fun computes STLOG as a cs function object, which can be called with self._fun()(x,u,t)
        self._stlog_fun = cs.Function(
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
            return self._metric(self._stlog_fun(x, u, t))

        def model_predictive_evaluate_metric(us, x, t):
            us = us.reshape(self._nu, -1, order="F")
            xs = self._mdl_predict_fun(x, us, t)
            return sum(
                evaluate_metric(us[:, idx], xs[:, idx], t)
                for idx in range(self._window)
            )

        bound_args = {}
        if x_fix is not None:
            bound_args["x"] = x_fix
        if t_fix is not None:
            bound_args["t"] = t_fix

        if bound_args:
            if self._window > 1:
                return functools.partial(model_predictive_evaluate_metric, **bound_args)
            return functools.partial(evaluate_metric, **bound_args)
        return evaluate_metric

    def make_problem(self, x0, u0, t, u_lb, u_ub, log_scale=False, omit_leader=False):
        # log_scale still in testing
        # if max(abs(np.concatenate((u0, u_lb, u_ub)))) * t > 1.0:
        #     print(
        #         f"Warning: max(|u*t|) = {max(abs(np.concatenate((u0,u_lb,u_ub))))*t} > 1. STLOG convergence is not guaranteed."
        #     )
        obj_fun_primitive = self.objective(x_fix=x0, t_fix=t)
        u_leader = u0[0 : self._NU]
        if omit_leader:

            def obj_fun_omit_leader(u_follower):
                if self._window > 1:
                    u_follower = u_follower.reshape(self._nu - self._NU, -1, order="F")
                return obj_fun_primitive(np.vstack((u_leader, u_follower)))

            obj_fun = obj_fun_omit_leader
            u0 = u0[self._NU :]
            u_lb = u_lb[self._NU :]
            u_ub = u_ub[self._NU :]
        else:
            obj_fun = obj_fun_primitive

        if self._window > 1:
            u0 = u0.ravel(order="F")
            u_lb = u_lb.ravel(order="F")
            u_ub = u_ub.ravel(order="F")

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


def _lie_derivative(fun, vector_field, order, sym_x):

    # Zeroth-order Lie Derivative
    lfh = fun

    # Implement the recurrence relationship for higher order lie derivatives
    for _ in range(order + 1):
        yield lfh
        lfh = cs.jtimes(lfh, sym_x, vector_field)


def create_stlog(symbols, order, mdl=None):

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
    lfh = _lie_derivative(hfcn, fcn, order, symbols["x"])
    dlh_store = [cs.jacobian(it, symbols["x"]) for it in lfh]

    # up to the (order+1)-th factorial
    facts = special.factorial(np.arange(0, order + 1), exact=True)

    assert isinstance(facts, np.ndarray)

    # Define two ways to sum lie-gradients, one that ensures positive-definiteness of
    # the resulting Gramian, and one that does not
    def psd_stlog(a, b):
        coeff = symbols["t"] ** (a + b + 1) / (facts[a] * facts[b] * (a + b + 1))
        return coeff * (dlh_store[a].T @ dlh_store[b])

    return sum(psd_stlog(a, b) for a in range(order + 1) for b in range(order + 1))
