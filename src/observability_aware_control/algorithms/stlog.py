import functools
import inspect
from typing import Callable

import jax
import jax.numpy as jnp
from jax.scipy import special


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

    # Implement the recurrence relationship for higher order lie derivatives
    for _ in range(order + 1):
        yield lfh
        lfh = functools.partial(lfh_impl, lfh, vector_field)


class STLOG(object):
    """This class manages computation of the Short Time Local observability Gramian"""

    observation: Callable
    dynamics: Callable
    nx: int

    def __init__(self, order, cov=None):

        # Setup the lie derivatives
        self._dalfh_f = [
            jax.jacobian(it)
            for it in _lie_derivative(
                self.observation,
                self.dynamics,
                order,
            )
        ]
        self._i_cov = jnp.linalg.inv(cov)[None, None, ...] if cov is not None else None

        self._order = order
        # Cache some order-dependent constant numeric data for stlog evaluation
        order_seq = jnp.arange(order + 1)
        self._a, self._b, *_ = jnp.ix_(order_seq, order_seq)
        self._k = self._a + self._b + 1
        facts = special.factorial(order_seq)
        self._den = facts[self._a] * facts[self._b] * self._k

    @property
    def order(self):
        return self._order

    @property
    def cov(self):
        return self._i_cov

    @cov.setter
    def cov(self, val):
        self._i_cov = val

    def stlog(self, x, u, dt):
        dalfh = jnp.stack(jax.tree_map(lambda it: it(x, u), self._dalfh_f))
        coeff = (dt**self._k / self._den)[..., None, None]
        if self._i_cov is None:
            return jnp.sum(coeff * dalfh[self._a].mT @ dalfh[self._b], axis=(0, 1))
        else:
            return jnp.sum(
                coeff * dalfh[self._a].mT @ self._i_cov @ dalfh[self._b], axis=(0, 1)
            )
