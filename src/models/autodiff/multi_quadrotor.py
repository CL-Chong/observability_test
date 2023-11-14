import functools

import jax
import jax.numpy as jnp

from .. import model_base
from . import quadrotor


class MultiQuadrotor(model_base.ModelBase):
    DIM_ATT_OBS = 4
    DIM_BRNG_OBS = 2

    def __init__(self, n_robots, mass):
        self._n_robots = n_robots
        self._mass = mass

    @property
    def nx(self):
        return self._n_robots * quadrotor.NX

    @property
    def nu(self):
        return self._n_robots * quadrotor.NU

    @property
    def n_robots(self):
        return self._n_robots

    @functools.partial(jax.jit, static_argnames=("self",))
    def dynamics(self, x, u):
        x = x.reshape(self._n_robots, quadrotor.NX)
        u = u.reshape(self._n_robots, quadrotor.NU)

        dynamics = jax.vmap(quadrotor.dynamics, out_axes=0)
        return dynamics(x, u, self._mass).ravel()

    @property
    def ny(self):
        return self._n_robots * (self.DIM_ATT_OBS + self.DIM_BRNG_OBS)

    @functools.partial(jax.jit, static_argnames=("self",))
    @functools.partial(jax.vmap, in_axes=(None, 0))
    def observation(self, x):
        x = x.reshape(self._n_robots, quadrotor.NX)
        h_att = x[:, 3:7].ravel()
        pos_ref = x[0, 0:2]

        obs = jax.vmap(quadrotor.observation, in_axes=(None, 0, None))

        h_bearings = obs(x, pos_ref)
        return jnp.concatenate([h_att, h_bearings])
