import jax
import jax.numpy as jnp
from jax.scipy import special

from observability_aware_control.algorithms.common import lie_derivative


class STLOG:
    """This class manages computation of the Short Time Local observability Gramian"""

    def __init__(self, dynamics, observation, order, cov=None):

        # Setup the lie derivatives
        self._dalfh_f = [
            jax.jacobian(it) for it in lie_derivative(observation, dynamics, order)
        ]

        self._cov = cov if cov is not None else None
        if cov is not None:
            self._inv_cov = jnp.linalg.inv(cov)[None, None, ...]
        else:
            self._inv_cov = None

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
        dalfh = jnp.stack([it(x, u) for it in self._dalfh_f])
        coeff = (dt**self._k / self._den)[..., None, None]
        if self._inv_cov is None:
            return jnp.sum(coeff * dalfh[self._a].mT @ dalfh[self._b], axis=(0, 1))
        else:
            return jnp.sum(
                coeff * dalfh[self._a].mT @ self._inv_cov @ dalfh[self._b], axis=(0, 1)
            )
