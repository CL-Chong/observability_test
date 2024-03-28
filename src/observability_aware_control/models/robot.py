import functools

import jax
import jax.numpy as jnp

from . import model_base

NX = 3
NU = 2


@jax.jit
def dynamics(x, u):
    psi = x[2]
    v = u[0]
    w = u[1]

    c = jnp.cos(psi)
    s = jnp.sin(psi)

    return jnp.array([c * v, s * v, w])


@property
def nx(self):
    return self.NX


@property
def nu(self):
    return self.NU


class PlanarRobot(model_base.ModelBase):
    @property
    def nx(self):
        return NX

    @property
    def nu(self):
        return NU

    @functools.partial(jax.jit, static_argnums=(0,))
    def dynamics(self, x, u):
        return dynamics(x, u)
