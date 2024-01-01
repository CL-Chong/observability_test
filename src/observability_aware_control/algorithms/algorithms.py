import dataclasses
import functools
import inspect
import time
from typing import Callable, Optional, Tuple

import jax
import jax.numpy as jnp
from scipy import optimize, special

from observability_aware_control.models import model_base

from .. import utils
from ..optimize import minimize_problem

NX = 3
NU = 2


jitmember = functools.partial(jax.jit, static_argnames=("self",))


@functools.partial(jax.jit, static_argnames=("dynamics", "method"))
def forward_dynamics(dynamics, x0, u, dt, method="RK4"):
    """Run forward simulation of a dynamical system

    Details
    -------
    This function enables time-varying parameters (known beforehand) to be passed in,
    whereas constant parameters are intended to be put into the sys object

    Parameters
    ----------
    sys : ModelBase
        An object satisfying the ModelBase Interface
    x0 : ArrayLike
        Initial state
    u : ArrayLike
        A sys.nu-by-len(dt) array of control inputs
    dt : ArrayLike
        An array of time steps
    method: Literal["RK4"] | Literal["euler"]
        Specifies the integration method
    Returns
    -------
    Tuple[jnp.array, jnp.array] | jnp.array
        The state and optionally observation trajectory
    """

    def _update(x_op, tup):
        u, dt = tup
        if method == "RK4":
            k = jnp.empty((4, x_op.size))
            k = k.at[0, :].set(dynamics(x_op, u))
            k = k.at[1, :].set(dynamics(x_op + dt / 2 * k[0, :], u))
            k = k.at[2, :].set(dynamics(x_op + dt / 2 * k[1, :], u))
            k = k.at[3, :].set(dynamics(x_op + dt * k[2, :], u))
            increment = jnp.array([1, 2, 2, 1]) @ k / 6
        elif method == "euler":
            increment = dynamics(x_op, u)
        else:
            raise NotImplementedError(f"{method} is not a valid integration method")
        x_new = x_op + dt * increment
        return x_new, x_new

    u = jnp.atleast_2d(u)
    dt = jnp.broadcast_to(dt, u.shape[0])
    _, x = jax.lax.scan(_update, init=x0, xs=(u, dt))
    return x


def _default_stlog_metric(x):
    return -jnp.linalg.norm(x, -2)


@dataclasses.dataclass
class STLOGOptions:
    dt: float
    window: int = dataclasses.field(default=1)
    id_const: Tuple[int, ...] = dataclasses.field(default=())
    ub: jax.Array = dataclasses.field(default=-jnp.inf)
    lb: jax.Array = dataclasses.field(default=jnp.inf)
    max_num_iters: int = dataclasses.field(default=100)


def lfh_impl(fun, vector_field, x, u):
    _, f_jvp = jax.linearize(functools.partial(fun, u=u), x)
    return f_jvp(vector_field(x, u))


def _lie_derivative(fun, vector_field, order, proj=None):
    # Zeroth-order Lie Derivative
    funsig = inspect.signature(fun)
    if "u" not in funsig.parameters:
        lfh = lambda x, u: fun(x)
    else:
        lfh = fun

    # Prevent loop unrolling
    with jax.disable_jit():
        # Implement the recurrence relationship for higher order lie derivatives
        for _ in range(order + 1):
            yield proj(lfh) if proj is not None else lfh

            # Caveats:
            # - fh=lfh: Default-argument trick bypasses 'Cell variable ... defined in loop'
            #   bug and ensures the freshest lie derivative is used in the lambda
            # - The system input vector u does not take part in lie differentiation, hence
            #   it is bound into the target for lie differentiation fn

            lfh = jax.tree_util.Partial(lfh_impl, lfh, vector_field)


class STLOG:
    def __init__(
        self,
        mdl: model_base.MRSBase,
        order,
        dt,
        metric=_default_stlog_metric,
        components=(),
    ):
        self._order = order
        self._order_seq = jnp.arange(order + 1)
        self._model = mdl

        self._nx = mdl.nx
        self._nu = mdl.nu
        self._dt_stlog = dt
        if components:
            id_mut = jnp.asarray(components, dtype=jnp.int32)
            id_const = utils.complementary_indices(self._nx, id_mut)

            def jacobian_wrt_mutable_components(fun):
                jac = jax.jacfwd(utils.separate_array_argument(fun, self._nx, id_mut))
                return lambda x, *args: jac(x[id_mut], x[id_const], *args)

            proj = jacobian_wrt_mutable_components
        else:
            proj = jax.jacfwd

        self._dalfh_f = list(
            _lie_derivative(self.model.observation, self.model.dynamics, order, proj)
        )
        self._facts = jnp.asarray(special.factorial(self._order_seq, exact=True))

        self._metric = metric

    @property
    def model(self):
        return self._model

    @property
    def nx(self):
        return self.model.nx

    @property
    def nu(self):
        return self.model.nu

    def evaluate(self, x, u):
        dalfh = jnp.stack(jax.tree_util.tree_map(lambda it: it(x, u), self._dalfh_f))
        # dalfh = jnp.stack([it(x, u) for it in self._dalfh_f])

        a, b = jnp.ix_(self._order_seq, self._order_seq)
        k = a + b + 1
        coeff = self._dt_stlog**k / (self._facts[a] * self._facts[b] * k)
        coeff = jnp.expand_dims(coeff, range(2, 4))

        inner = dalfh[a].swapaxes(3, 2) @ dalfh[b]
        res = (coeff * inner).sum(axis=(0, 1))

        if self._metric is not None:
            return self._metric(res)
        return res


class OptimizeRecorder:
    def __init__(self, max_num_iters):
        self.fval = jnp.full(max_num_iters, jnp.nan)

    def update(self, _, info):
        self.fval = self.fval.at[info.nit].set(info.fun)


class STLOGMinimizeProblem:
    def __init__(self, stlog: STLOG, opts: STLOGOptions):
        self._stlog = stlog
        self._nx = self._stlog.nx
        self._nu = self._stlog.nu
        self._dt_stlog = opts.dt

        self.gradient = jax.jit(jax.grad(self.objective))

        self.cons_grad = jax.jit(jax.jacfwd(self.constraint))
        self._id_const = jnp.asarray(opts.id_const, dtype=jnp.int32)
        self._id_mut = utils.complementary_indices(self._nu, self._id_const)
        self._n_mut = len(self._id_mut)
        self._max_num_iters = opts.max_num_iters
        self._rec = OptimizeRecorder(100)

        def objective(us, x, dt):
            xs = forward_dynamics(self.stlog.model.dynamics, x, us, dt)
            return jnp.sum(jax.vmap(self.stlog.evaluate)(xs, us))

        self._u_shape = (opts.window, self._nu)
        self._fun = utils.separate_array_argument(
            objective, self._u_shape, self._id_mut
        )

        def constraint(us, x, dt):
            xs = forward_dynamics(self.stlog.model.dynamics, x, us, dt)
            pos = xs.reshape(xs.shape[0], -1, 10)[:, :, 0:3]
            dp = pos[:, 1:, :] - pos[:, [0], :]
            dp_nrm = (dp * dp).sum(axis=-1).ravel()
            return dp_nrm

        self._cons = utils.separate_array_argument(
            constraint, self._u_shape, self._id_mut
        )
        self.cons_grad = jax.jit(jax.jacfwd(self.constraint))
        self._u_lb = jnp.broadcast_to(opts.lb, self._u_shape)[..., self._id_mut].ravel()
        self._u_ub = jnp.broadcast_to(opts.ub, self._u_shape)[..., self._id_mut].ravel()

        self.problem = minimize_problem.MinimizeProblem(
            self.objective, jnp.zeros(self._u_shape)
        )
        self.problem.bounds = optimize.Bounds(self._u_lb, self._u_ub)  # type: ignore

        self.problem.jac = jax.jit(jax.grad(self.objective))
        self.problem.method = "trust-constr"
        self.problem.options = {
            "xtol": 1e-1,
            "gtol": 1e-4,
            "disp": False,
            "verbose": 0,
            "maxiter": 100,
        }
        self.problem.constraints = optimize.NonlinearConstraint(
            lambda u: self.constraint(u, *self.problem.args),
            lb=jnp.full(self._n_mut // self._stlog.model.robot_nu * opts.window, 0.01),
            ub=jnp.full(self._n_mut // self._stlog.model.robot_nu * opts.window, 10.0),
            jac=lambda u: self.cons_grad(u, *self.problem.args),
        )

    @property
    def stlog(self):
        return self._stlog

    @jitmember
    def objective(self, u_mut, u_const, x, dt):
        return self._fun(u_mut.reshape(-1, self._n_mut), u_const, x, dt)

    @jitmember
    def constraint(self, u_mut, u_const, x, dt):
        return self._cons(u_mut.reshape(-1, self._n_mut), u_const, x, dt)

    def minimize(self, x0, u0, t) -> optimize.OptimizeResult:
        if self.problem is None:
            raise ValueError("Unconfigured problem")
        self._u_const = u0[..., self._id_const]
        self.problem.x0 = u0[..., self._id_mut].ravel()
        self.problem.args = (self._u_const, x0, t)

        fun_hist = []

        def update(_, info):
            fun_hist.append(info.fun)

        self.problem.callback = update

        prob_dict = vars(self.problem)
        soln = optimize.minimize(**prob_dict)

        x = soln.x.reshape(-1, self._n_mut)  # type: ignore
        soln.x = utils.combine_array(
            self._u_shape, self._u_const, x, self._id_const, self._id_mut  # type: ignore
        )
        soln["fun_hist"] = fun_hist
        return soln


@functools.partial(jax.jit, static_argnames=("sys", "eps", "axis"))
def _numlog(sys, x0, u, dt, eps, axis, perturb_axis):
    if perturb_axis is None:
        perturb_axis = jnp.arange(0, sys.nx)
    perturb_axis = jnp.asarray(perturb_axis)

    observation = jax.vmap(sys.observation)

    @functools.partial(jax.vmap, out_axes=2)
    def _perturb(x0_plus, x0_minus):
        _, yi_plus = observation(forward_dynamics(sys.dynamics, x0_plus, u, dt))
        _, yi_minus = observation(forward_dynamics(sys.dynamics, x0_minus, u, dt))
        return yi_plus - yi_minus

    perturb_bases = jnp.eye(x0.size)[perturb_axis]
    x0_plus = x0 + eps * perturb_bases
    x0_minus = x0 - eps * perturb_bases
    y_all = _perturb(x0_plus, x0_minus) / (2.0 * eps)

    coord_vec = jnp.arange(0, perturb_axis.size)
    xm, ym = jnp.meshgrid(coord_vec, coord_vec)

    dt = dt[..., None, None, None]
    if axis is not None:
        dt = jnp.moveaxis(dt, 0, axis)

    return jnp.sum(dt * y_all[:, :, xm] * y_all[:, :, ym], axis=(0, 1))


def numlog(sys, x0, u, dt, eps, axis=None, perturb_axis=None, f_args=None, h_args=None):
    x0 = jnp.asarray(x0)
    u = jnp.asarray(u)
    dt = jnp.asarray(dt)
    n_steps = dt.size
    if x0.shape != (sys.nx,):
        raise ValueError(f"Expected x0 with shape ({sys.nx}), got {x0.shape}")

    if u.shape != (sys.nu, n_steps):
        raise ValueError(
            f"Expected matrix with shape ({sys.nu}, {n_steps}), got {u.shape}"
        )
    if jnp.any(dt <= 0):
        raise ValueError("Discrete time-step is not positive.")
    return _numlog(sys, x0, u, dt, eps, axis, perturb_axis, f_args, h_args)
