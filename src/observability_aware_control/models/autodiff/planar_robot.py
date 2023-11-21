import functools

import jax
import jax.numpy as jnp

from .. import model_base

NX = 3
NU = 2
NY = 1


@jax.jit
def dynamics(x, u):
    psi = x[2]
    v = u[0]
    w = u[1]

    c = jnp.cos(psi)
    s = jnp.sin(psi)

    return jnp.array([c * v, s * v, w])


@jax.jit
def observation(x, pos_ref):
    psi = x[2]
    c = jnp.cos(psi)
    s = jnp.sin(psi)

    p_diff = pos_ref - x[0:2]

    hx = c * p_diff[0] + s * p_diff[1]
    hy = -s * p_diff[0] + c * p_diff[1]
    return jnp.arctan2(hy, hx)


@property
def nx(self):
    return self.NX


@property
def nu(self):
    return self.NU


@property
def ny(self):
    return self.NY


class PlanarRobot(model_base.ModelBase):
    @property
    def nx(self):
        return NX

    @property
    def nu(self):
        return NU

    @property
    def ny(self):
        return NY

    @functools.partial(jax.jit, static_argnums=(0,))
    def dynamics(self, x, u):
        return dynamics(x, u)

    @functools.partial(jax.jit, static_argnums=(0,))
    def observation(self, x, pos_ref):
        return observation(x, pos_ref)
