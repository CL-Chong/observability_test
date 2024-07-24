from typing import Callable

import jax
import jax.numpy as jnp
from jax.scipy import special

from observability_aware_control.algorithms.common import lie_derivative


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

        self._cov = cov if cov is not None else None
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
        return jnp.squeeze(self._cov) if self._cov is not None else None

    @cov.setter
    def cov(self, val):
        self._cov = val

    def stlog(self, x, u, dt):
        dalfh = jnp.stack(jax.tree_map(lambda it: it(x, u), self._dalfh_f))
        coeff = (dt**self._k / self._den)[..., None, None]
        if self._i_cov is None:
            return jnp.sum(coeff * dalfh[self._a].mT @ dalfh[self._b], axis=(0, 1))
        else:
            return jnp.sum(
                coeff * dalfh[self._a].mT @ self._i_cov @ dalfh[self._b], axis=(0, 1)
            )
