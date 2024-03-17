import dataclasses
import functools
import itertools
import warnings
from typing import Any, Dict, Optional

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
from scipy import optimize

from .. import utils
from . import common, opc, stlog

NX = 3
NU = 2


jitmember = functools.partial(jax.jit, static_argnames=("self",))


@dataclasses.dataclass
class CooperativeLocalizationOptions:
    window: int = dataclasses.field(default=1)
    obs_comps: Optional[ArrayLike] = None
    id_leader: int = dataclasses.field(default=0)
    ub: ArrayLike = dataclasses.field(default_factory=lambda: -jnp.array(jnp.inf))
    lb: ArrayLike = dataclasses.field(default_factory=lambda: jnp.array(jnp.inf))
    min_v2v_dist: float = dataclasses.field(default=0.0)
    max_v2v_dist: float = dataclasses.field(default=jnp.inf)
    method: Optional[utils.optim_utils.Method] = dataclasses.field(default=None)
    optim_options: Optional[Dict[str, Any]] = dataclasses.field(default=None)


class CooperativeLocalizingOPC(opc.OPCCost):
    def __init__(self, model, opts: CooperativeLocalizationOptions):
        # Initialize the underlying cost function
        opc.OPCCost.__init__(self, model, opts.obs_comps)

        # Pick up some constants from the model inside the OPC
        self._nu = self.model.nu
        self._robot_nu = self.model.robot_nu
        self._n_robots = self.model.n_robots

        # The following members are used to split/reform the input trajectory to/from
        # constant/mutable parts
        self._u_shape = (opts.window, self._nu)
        self._id_const = jnp.arange(opts.id_leader, opts.id_leader + self._robot_nu)
        self._id_mut = utils.complementary_indices(self._nu, self._id_const)
        self._n_mut = len(self._id_mut)

        # Broadcast bounds over all time windows, takes mutable components, then flatten
        lb = jnp.broadcast_to(jnp.array(opts.lb), self._u_shape)[
            ..., self._id_mut
        ].ravel()

        ub = jnp.broadcast_to(jnp.array(opts.ub), self._u_shape)[
            ..., self._id_mut
        ].ravel()

        self.problem = utils.optim_utils.MinimizeProblem(
            jax.jit(self.objective),
            jnp.zeros(self._u_shape),
            jac=jax.jit(jax.grad(self.objective)),
            method=opts.method,
            options=opts.optim_options,
            bounds=optimize.Bounds(lb, ub),  # type: ignore
        )

        if 0 < opts.min_v2v_dist < opts.max_v2v_dist:
            # Compute pairs of quadrotors whose pair-wise distance must be constrained
            self._combs = jnp.array(
                list(itertools.combinations(range(self._n_robots), 2))
            )

            self.problem.constraints = optimize.NonlinearConstraint(
                lambda u: self.constraint(u, *self.problem.args),
                lb=jnp.full(self._combs.shape[0] * opts.window, opts.min_v2v_dist**2),
                ub=jnp.full(self._combs.shape[0] * opts.window, opts.max_v2v_dist**2),
            )
            cjac = jax.jacobian(self.constraint)
            self.problem.constraints.jac = lambda u: cjac(u, *self.problem.args)  # type: ignore
        else:
            warnings.warn(
                "Invalid bounds, vehicle-to-vehicle ranges will not be constrained"
            )

    def combine_input(self, u, u_const):
        u = u.reshape(-1, self._n_mut)
        return utils.combine_array(
            self._u_shape, u, u_const, self._id_mut, self._id_const
        )

    @functools.partial(jax.jit, static_argnames=("self", "return_stlog", "return_traj"))
    def opc(self, us, x, dt, return_stlog=False, return_traj=False):
        dt = jnp.broadcast_to(dt, us.shape[0])
        return super().opc(us, x, dt, return_stlog, return_traj)

    @functools.partial(jax.jit, static_argnames=("self",))
    def objective(self, u, u_const, x, dt):
        u = self.combine_input(u, u_const)
        return self.opc(u, x, dt, False, False)

    @jitmember
    def constraint(self, u, u_const, x, dt):
        us = self.combine_input(u, u_const)
        dt = jnp.broadcast_to(dt, us.shape[0])
        xs = common.forward_dynamics(self.model.dynamics, x, us, dt)
        pos = xs.reshape(xs.shape[0], -1, 10)[:, :, 0:3]
        dp = jnp.diff(pos[:, self._combs, :], axis=2)
        dp_nrm = (dp**2).sum(axis=-1).ravel()
        return dp_nrm

    def minimize(self, x0, u0, t) -> optimize.OptimizeResult:
        u_const = u0[..., self._id_const]
        self.problem.x0 = u0[..., self._id_mut].ravel()
        self.problem.args = (u_const, x0, t)

        rec = utils.optim_utils.OptimizationRecorder()
        self.problem.callback = rec.update

        prob_dict = vars(self.problem)
        soln = optimize.minimize(**prob_dict)

        soln["x"] = self.combine_input(soln.x, u_const)
        soln["fun_hist"] = rec.fun
        return soln
