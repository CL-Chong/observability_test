import functools
import inspect

import jax
import jax.numpy as jnp
from scipy import special


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
    # Implement the recurrence relationship for higher order lie derivatives
    for _ in range(order + 1):
        yield lfh
        lfh = functools.partial(lfh_impl, lfh, vector_field)


class STLOG:
    """This class manages computation of the Short Time Local observability Gramian"""

    def __init__(self, mdl, order, cov=None):
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
        if cov is not None:
            if cov.ndim == 2:
                self._cov = jnp.linalg.inv(cov)[None, None, ...]
        else:
            self._cov = jnp.eye(self.model.ny)[None, None, ...]

        # Cache some order-dependent constant numeric data for stlog evaluation
        order_seq = jnp.arange(order + 1)
        self._a, self._b, *_ = jnp.ix_(order_seq, order_seq)
        self._k = self._a + self._b + 1
        facts = jnp.asarray(special.factorial(order_seq, exact=True))
        self._den = facts[self._a] * facts[self._b] * self._k

    @property
    def model(self):
        return self._model

    def __call__(self, x, u, dt):
        dalfh = jnp.stack(jax.tree_map(lambda it: it(x, u), self._dalfh_f))
        coeff = (dt**self._k / self._den)[..., None, None]
        inner = dalfh[self._a].swapaxes(3, 2) @ self._cov @ dalfh[self._b]
        return (coeff * inner).sum(axis=(0, 1))
