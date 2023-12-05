import functools

import jax
import jax.numpy as jnp

from .. import model_base

NX = 4
NU = 3
NY = 2


@jax.jit
def dynamics(x, u):
    psi = x[3]
    v = u[0]
    dz = u[1]
    w = u[2]

    c = jnp.cos(psi)
    s = jnp.sin(psi)

    return jnp.array([c * v, s * v, dz, w])


@jax.jit
def observation(x, pos_ref):
    psi = x[3]
    c = jnp.cos(psi)
    s = jnp.sin(psi)

    p_diff = pos_ref - x[0:3]

    hx = c * p_diff[0] + s * p_diff[1]
    hy = -s * p_diff[0] + c * p_diff[1]
    return jnp.array(
        [
            jnp.arctan2(hy, hx),
            jnp.arctan2(p_diff[2], jnp.hypot(hx, hy)),
        ]
    )


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
