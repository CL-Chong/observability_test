import dataclasses
import functools
import inspect
import time
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
from scipy import optimize, special

from .. import utils
from ..optimize import minimize_problem

NX = 3
NU = 2


jitmember = functools.partial(jax.jit, static_argnames=("self",))


def _default_stlog_metric(x):
    return -jnp.linalg.norm(x, -2)


@dataclasses.dataclass
class STLOGOptions:
    dt: float
    window: int = dataclasses.field(default=1)
    id_const: Optional[Tuple[int, ...]] = dataclasses.field(default=None)


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
    def __init__(self, mdl, order, metric=_default_stlog_metric, components=()):
        self._order = order
        self._order_seq = jnp.arange(order + 1)

        self._observation = mdl.observation
        self._dynamics = mdl.dynamics
        self._nx = mdl.nx
        self._nu = mdl.nu
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
            _lie_derivative(self.observation, self.dynamics, order, proj)
        )
        self._facts = jnp.asarray(special.factorial(self._order_seq, exact=True))

        self._metric = metric

    @property
    def nx(self):
        return self._nx

    @property
    def nu(self):
        return self._nu

    @property
    def observation(self):
        return self._observation

    @property
    def dynamics(self):
        return self._dynamics

    def _stlog(self, x, u, stlog_t):
        dalfh = jnp.stack(jax.tree_util.tree_map(lambda it: it(x, u), self._dalfh_f))
        # dalfh = jnp.stack([it(x, u) for it in self._dalfh_f])

        a, b = jnp.ix_(self._order_seq, self._order_seq)
        k = a + b + 1
        coeff = stlog_t**k / (self._facts[a] * self._facts[b] * k)
        coeff = jnp.expand_dims(coeff, range(2, 4))

        inner = dalfh[a].swapaxes(3, 2) @ dalfh[b]
        return (coeff * inner).sum(axis=(0, 1))

    @jitmember
    def evaluate(self, x, u, t):
        if self._metric is not None:
            return self._metric(self._stlog(x, u, t))
        return self._stlog(x, u, t)


class STLOGMinimizeProblem:
    def __init__(self, stlog: STLOG, opts: STLOGOptions):
        self._stlog = stlog
        self._nx = self._stlog.nx
        self._nu = self._stlog.nu
        self._dt_stlog = opts.dt

        gradient = jax.jit(jax.grad(self.objective))
        if opts.window > 1:
            print("Precompiling gradient...", end="")
            tic = time.perf_counter()

            self.gradient = gradient.lower(
                jnp.zeros((opts.window, self._nu)),
                jnp.zeros(self._nx),
                0.2,
            ).compile()
            toc = time.perf_counter() - tic
            print(f"Done in {toc}s")
        else:
            self.gradient = gradient

    @jitmember
    def objective(self, us, x, dt):
        xs = self.forward_dynamics(x, us, dt)
        dt = jnp.broadcast_to(self._dt_stlog, us.shape[0])
        return jnp.sum(jax.vmap(self._stlog.evaluate)(xs, us, dt))

    @jitmember
    def forward_dynamics(self, x0, u, dt):
        def _update(x_op, tup):
            u, dt = tup
            x_new = x_op + dt * self._stlog.dynamics(x_op, u)
            return x_new, x_new

        u = jnp.atleast_2d(u)
        dt = jnp.broadcast_to(dt, u.shape[0])
        return jax.lax.scan(_update, init=x0, xs=(u, dt))[1]

    @jitmember
    def lf_constraint(self, us, x, dt):
        xs = self.forward_dynamics(x, us, dt)
        xs = jnp.moveaxis(xs.reshape(xs.shape[0], -1, 10)[:, :, 0:3], 1, 0)
        z = xs[:, :, -1].ravel()
        dp = jnp.linalg.norm(xs[1:, ...] - xs[0, ...], axis=-1).ravel()
        return jnp.concatenate([dp, z])

    def make_problem(self, x0, u0, t, u_lb, u_ub, id_const):
        # log_scale still in testing
        # if max(abs(np.concatenate((u0, u_lb, u_ub)))) * t > 1.0:
        #     print(
        #         f"Warning: max(|u*t|) = {max(abs(np.concatenate((u0,u_lb,u_ub))))*t} > 1. STLOG convergence is not guaranteed."
        #     )
        u0, u_lb, u_ub = jnp.broadcast_arrays(u0, u_lb, u_ub)
        args = (x0, t)

        problem = minimize_problem.MinimizeProblem(self.objective, u0)
        problem.args = args
        problem.id_const = id_const
        problem.constraints = optimize.NonlinearConstraint(
            lambda u: self.lf_constraint(u, x0, t),
            lb=jnp.r_[jnp.full(u0.shape[0] * 2, 0.2), jnp.full(u0.shape[0] * 3, 9.5)],
            ub=jnp.r_[jnp.full(u0.shape[0] * 2, 3.2), jnp.full(u0.shape[0] * 3, 10.5)],
        )
        problem.jac = self.gradient

        problem.bounds = optimize.Bounds(u_lb, u_ub)  # type: ignore
        problem.method = "trust-constr"
        problem.options = {
            "xtol": 1e-4,
            "gtol": 1e-8,
            "disp": False,
            "verbose": 0,
            "maxiter": 100,
        }

        return problem


@functools.partial(jax.jit, static_argnames=("sys", "without_observation", "axis"))
def numsolve_sigma(
    sys,
    x0,
    u,
    dt,
    axis=None,
    f_args=None,
    h_args=None,
    without_observation=False,
):
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
    axis: int
        Axis that defines each vector-valued input
    f_args : ArrayLike, optional
        Additional (time-dependent) vector-valued inputs for sys.dynamics specified by a
        *-by-len(dt) array, by default None
    h_args : ArrayLike, optional
        Additional (time-dependent) vector-valued inputs for sys.observation specified
        by a *-by-len(dt) array, by default None
    without_observation: bool, optional
        If true, the observation equation will not be run on the states, and only the
        states trajectory will be returned

    Returns
    -------
    Tuple[jnp.array, jnp.array] | jnp.array
        The state and optionally observation trajectory
    """

    def _process_in_axis(var):
        return jnp.moveaxis(var, axis, 0) if axis is not None else var

    def _process_out_axis(var):
        return jnp.moveaxis(var, 0, axis) if axis is not None else var

    def _update(x_op, tup):
        dt, *rem = tup
        dx = sys.dynamics(x_op, *rem)
        return x_op + dt * dx, x_op

    u = _process_in_axis(u)
    if f_args is None:
        xs = (dt, u)
    else:
        xs = (dt, u, _process_in_axis(f_args))

    _, x = jax.lax.scan(_update, init=x0, xs=xs)

    if without_observation:
        return _process_out_axis(x)

    if h_args is None:
        y = sys.observation(x)
    else:
        y = sys.observation(x, _process_in_axis(h_args))

    return _process_out_axis(x), _process_out_axis(y)


@functools.partial(jax.jit, static_argnames=("sys", "eps", "axis"))
def _numlog(sys, x0, u, dt, eps, axis, perturb_axis, f_args, h_args):
    if perturb_axis is None:
        perturb_axis = jnp.arange(0, sys.nx)
    perturb_axis = jnp.asarray(perturb_axis)

    @functools.partial(jax.vmap, out_axes=2)
    def _perturb(x0_plus, x0_minus):
        _, yi_plus = numsolve_sigma(sys, x0_plus, u, dt, axis, f_args, h_args)
        _, yi_minus = numsolve_sigma(sys, x0_minus, u, dt, axis, f_args, h_args)
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
