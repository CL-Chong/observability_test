import dataclasses
import functools
import inspect
from typing import Tuple

import jax
import jax.numpy as jnp
import jax.numpy.linalg as la
from scipy import optimize, special


from .. import utils
from ..optimize import minimize_problem

NX = 3
NU = 2


jitmember = functools.partial(jax.jit, static_argnames=("self",))


@functools.partial(jax.jit, static_argnames=("dynamics", "method"))
def forward_dynamics(dynamics, x0, u, dt, method="euler"):
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
    _, x = jax.lax.scan(_update, init=x0, xs=(u, dt))
    return x


def lfh_impl(fun, vector_field, x, u):
    _, f_jvp = jax.linearize(functools.partial(fun, u=u), x)
    return f_jvp(vector_field(x, u))


def _lie_derivative(fun, vector_field, order):
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
            yield lfh
            lfh = functools.partial(lfh_impl, lfh, vector_field)


class STLOG:
    """This class manages computation of the Short Time Local observability Gramian"""

    def __init__(self, mdl, order):
        self._model = mdl

        # Setup the lie derivatives
        self._dalfh_f = [
            jax.jacfwd(it)
            for it in _lie_derivative(
                self.model.observation,
                self.model.dynamics,
                order,
            )
        ]

        # Cache some order-dependent constant numeric data for stlog evaluation
        order_seq = jnp.arange(order + 1)
        self._a, self._b, *_ = jnp.ix_(order_seq, order_seq)
        self._k = self._a + self._b + 1
        facts = jnp.asarray(special.factorial(order_seq, exact=True))
        self._den = facts[self._a] * facts[self._b]

    @property
    def model(self):
        return self._model

    def __call__(self, x, u, dt):
        dalfh = jnp.stack(jax.tree_map(lambda it: it(x, u), self._dalfh_f))
        coeff = (dt**self._k / self._den)[..., None, None]
        inner = dalfh[self._a].swapaxes(3, 2) @ dalfh[self._b]
        return (coeff * inner).sum(axis=(0, 1))


@dataclasses.dataclass
class STLOGOptions:
    dt: float
    window: int = dataclasses.field(default=1)
    obs_comps: Tuple[int, ...] = dataclasses.field(default=())
    id_const: Tuple[int, ...] = dataclasses.field(default=())
    ub: jax.Array = dataclasses.field(default_factory=lambda: -jnp.array(jnp.inf))
    lb: jax.Array = dataclasses.field(default_factory=lambda: jnp.array(jnp.inf))
    max_num_iters: int = dataclasses.field(default=100)


class STLOGMinimizeProblem:
    def __init__(self, stlog: STLOG, opts: STLOGOptions):
        self._stlog = stlog
        self._nu = self.stlog.model.nu
        self._dt_stlog = opts.dt
        if opts.obs_comps:
            obs_comps = jnp.asarray(opts.obs_comps, dtype=jnp.int32)
            self._i_stlog = (...,) + jnp.ix_(obs_comps, obs_comps)
        else:
            self._i_stlog = ...

        self._id_const = jnp.asarray(opts.id_const, dtype=jnp.int32)
        self._id_mut = utils.complementary_indices(self._nu, self._id_const)
        self._n_mut = len(self._id_mut)
        self._max_num_iters = opts.max_num_iters

        self._u_shape = (opts.window, self._nu)

        self.cons_grad = jax.jit(jax.jacfwd(self.constraint))
        self._u_lb = jnp.broadcast_to(opts.lb, self._u_shape)[..., self._id_mut].ravel()
        self._u_ub = jnp.broadcast_to(opts.ub, self._u_shape)[..., self._id_mut].ravel()

        fcn = utils.separate_array_argument(self.objective, self._u_shape, self._id_mut)
        con = utils.separate_array_argument(
            self.constraint, self._u_shape, self._id_mut
        )
        self.problem = minimize_problem.MinimizeProblem(
            jax.jit(lambda u, *args: fcn(u.reshape(-1, self._n_mut), *args)),
            jnp.zeros(self._u_shape),
        )
        self.problem.bounds = optimize.Bounds(self._u_lb, self._u_ub)  # type: ignore

        self.problem.jac = jax.jit(jax.grad(self.problem.fun))
        self.problem.method = "trust-constr"
        self.problem.options = {
            "xtol": 1e-1,
            "gtol": 1e-4,
            "disp": False,
            "verbose": 0,
            "maxiter": 150,
        }
        self.problem.constraints = optimize.NonlinearConstraint(
            lambda u: con(u.reshape(-1, self._n_mut), *self.problem.args),
            lb=jnp.full(self._n_mut // self._stlog.model.robot_nu * opts.window, 0.2),
            ub=jnp.full(self._n_mut // self._stlog.model.robot_nu * opts.window, 10.0),
        )
        self.problem.constraints.jac = jax.jacfwd(self.problem.constraints.fun)

    @property
    def stlog(self):
        return self._stlog

    def objective(self, us, x, dt):
        dt = jnp.broadcast_to(dt, us.shape[0])
        xs = forward_dynamics(self.stlog.model.dynamics, x, us, dt)
        stlog = jax.vmap(self.stlog)(xs, us, dt)[self._i_stlog]
        return -la.norm(stlog, -2, axis=(1, 2)).sum()

    def constraint(self, us, x, dt):
        dt = jnp.broadcast_to(dt, us.shape[0])
        xs = forward_dynamics(self.stlog.model.dynamics, x, us, dt)
        pos = xs.reshape(xs.shape[0], -1, 10)[:, :, 0:3]
        dp = pos[:, 1:, :] - pos[:, [0], :]
        dp_nrm = (dp * dp).sum(axis=-1).ravel()
        return dp_nrm

    def minimize(self, x0, u0, t) -> optimize.OptimizeResult:
        if self.problem is None:
            raise ValueError("Unconfigured problem")
        self._u_const = u0[..., self._id_const]
        self.problem.x0 = u0[..., self._id_mut].ravel()
        self.problem.args = (self._u_const, x0, t)

        fun_hist = []

        def update(intermediate_result):
            fun_hist.append(intermediate_result.fun)

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
