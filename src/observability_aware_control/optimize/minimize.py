import functools

import jax
import jax.numpy as jnp
from scipy import optimize

from .. import utils
from . import minimize_problem


def minimize(prob: minimize_problem.MinimizeProblem):
    x_shape = prob.x0.shape
    id_const = prob.id_const
    has_const_components = bool(id_const)
    if has_const_components:
        id_const = jnp.asarray(id_const)
        id_mut = utils.complementary_indices(x_shape[-1], id_const)
        n_mut = len(id_mut)
        u_const = prob.x0[..., id_const]

        prob.x0 = prob.x0[..., id_mut].ravel()

        if prob.bounds is not None:
            prob.bounds.lb = prob.bounds.lb[..., id_mut].ravel()
            prob.bounds.ub = prob.bounds.ub[..., id_mut].ravel()
            prob.bounds.keep_feasible = prob.bounds.keep_feasible[..., id_mut].ravel()

        fun = utils.separate_array_argument(prob.fun, x_shape, id_const)
        prob.fun = lambda x, *args: fun(u_const, x.reshape(-1, n_mut), *args)

        if callable(prob.jac):
            jac = utils.separate_array_argument(prob.jac, x_shape, id_const)
            prob.jac = lambda x, *args: jac(u_const, x.reshape(-1, n_mut), *args)[
                ..., id_mut
            ].ravel()

    else:
        pass

    prob_dict = vars(prob)
    prob_dict.pop("id_const")
    soln = optimize.minimize(**prob_dict)

    if has_const_components:
        x = soln.x.reshape(-1, n_mut)  # type: ignore
        soln.x = utils.combine_array(
            x_shape, u_const, x, id_const, id_mut  # type: ignore
        )
    return soln
