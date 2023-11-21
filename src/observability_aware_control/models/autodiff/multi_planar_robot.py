import functools

import jax
import jax.numpy as jnp

from .. import model_base
from . import planar_robot


class MultiRobot(model_base.ModelBase):
    def __init__(self, n_robots):
        self._n_robots = n_robots

    @functools.partial(jax.vmap, in_axes=(None, 0, None))
    def observation(self, x, pos_ref):
        return planar_robot.observation(x, pos_ref)

    @property
    def nx(self):
        return self._n_robots * planar_robot.NX

    @property
    def nu(self):
        return self._n_robots * planar_robot.NU

    @property
    def n_robots(self):
        return self._n_robots

    @functools.partial(jax.jit, static_argnames=("self",))
    def dynamics(self, x, u):
        x = x.reshape(self._n_robots, planar_robot.NX)
        u = u.reshape(self._n_robots, planar_robot.NU)

        dynamics = jax.vmap(planar_robot.dynamics, out_axes=0)
        return dynamics(x, u).ravel()


class ReferenceSensingRobots(MultiRobot):
    @property
    def ny(self):
        return self._n_robots + self._n_robots * 2

    @functools.partial(jax.jit, static_argnames=("self",))
    @functools.partial(jax.vmap, in_axes=(None, 0, 0))
    def observation(self, x, pos_ref):
        x = x.reshape(self._n_robots, planar_robot.NX)
        h_headings = x[:, 2].ravel()
        h_bearings = super().observation(x, pos_ref)

        return jnp.concatenate([h_headings, h_bearings])


class LeaderFollowerRobots(MultiRobot):
    @property
    def ny(self):
        return self._n_robots + (self._n_robots - 1) * 2

    def observation(self, x):
        if x.ndim > 1:
            return jax.vmap(self._observation, in_axes=(None, 0))(x)
        else:
            return self._observation(x)

    @functools.partial(jax.jit, static_argnames=("self",))
    def _observation(self, x):
        x = x.reshape((self._n_robots, planar_robot.NX))
        h_headings = x[:, 2].ravel()
        pos_ref = x[0, 0:2]

        h_bearings = super().observation(x[1:, :], pos_ref)

        return jnp.concatenate([pos_ref, h_headings, h_bearings])
