import functools

import jax
import jax.numpy as jnp

from .rotation import quaternion_product, quaternion_rotate_point

NX = 10
NU = 4


@functools.partial(jax.jit, donate_argnums=(0,))
def dynamics(x, u, mass):
    q = x[3:7]
    v = x[7:10]

    f = jnp.array([0.0, 0.0, u[0] / mass])
    w = jnp.array([u[1], u[2], u[3], 0.0]) / 2.0
    g = jnp.array([0.0, 0.0, -9.81])

    dx = jnp.empty(NX)
    dx = dx.at[0:3].set(v)
    dx = dx.at[3:7].set(quaternion_product(q, w))
    dx = dx.at[7:10].set(quaternion_rotate_point(q, f) + g)
    return dx


class Quadrotor:
    def __init__(self, mass):
        self._mass = mass

    def dynamics(self, x, u):
        return dynamics(x, u, self._mass)
