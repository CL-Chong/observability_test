from typing import Optional

import jax
import jax.numpy as jnp
import jax.numpy.linalg as la
from jax.typing import ArrayLike

from . import common, stlog


class OPCCost:
    def __init__(self, stlog_: stlog.STLOG, obs_comps: Optional[ArrayLike] = None):
        self._stlog = stlog_
        if obs_comps is not None:
            obs_comps = jnp.asarray(obs_comps, dtype=jnp.int32)
            self._i_stlog = jnp.ix_(obs_comps, obs_comps)
        else:
            self._i_stlog = ...

    @property
    def model(self):
        return self._stlog.model

    def __call__(self, us, x, dt, return_stlog=False, return_traj=False):
        # Real-time integration: Predict the system trajectory over the following stages
        xs = common.forward_dynamics(self.model.dynamics, x, us, dt)

        # Wraps stlog to be batch-evaluated over the following stages, with the result
        # to be sliced
        @jax.vmap
        def eval_stlog(x, u, dt):
            return self._stlog(x, u, dt)[self._i_stlog]

        # STLOGs are cached to a variable that may be optionally returned
        stlog_ = eval_stlog(xs, us, dt)

        # Evaluate (minimum) singular values for each STLOG in the stack
        # Sum them up, then apply inverse-of-logarithm scaling
        objective = 1 / jnp.log(
            la.svd(stlog_, compute_uv=False, hermitian=True).min(axis=1).sum()
        )

        if not (return_stlog or return_traj):
            return objective

        # Return more than just the objective value
        # Wrap it in a tuple then selectively add elements to return
        res = (objective,)
        if return_stlog:
            res += (stlog_,)
        if return_traj:
            res += (xs,)
        return res
