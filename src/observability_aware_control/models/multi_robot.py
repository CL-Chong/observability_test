import functools

import jax
import jax.numpy as jnp

from . import model_base, robot


class MultiRobot(model_base.ModelBase):
    """A system of multiple robots"""

    NX = robot.NX  # (innermost) state dimension
    NU = robot.NU  # (innermost) input dimension

    def __init__(self, n_robots):
        self._n_robots = n_robots

    @functools.partial(jax.vmap, in_axes=(None, 0, None))
    def observation(self, x, pos_ref):
        return robot.observation(x, pos_ref)

    @property
    def nx(self):
        return self._n_robots * robot.NX

    @property
    def nu(self):
        return self._n_robots * robot.NU

    @property
    def n_robots(self):
        return self._n_robots

    @functools.partial(jax.jit, static_argnames=("self",))
    def dynamics(self, x, u):
        x = x.reshape(self._n_robots, robot.NX)
        u = u.reshape(self._n_robots, robot.NU)

        dynamics = jax.vmap(robot.dynamics, out_axes=0)
        return dynamics(x, u).ravel()


class ReferenceSensingRobots(MultiRobot):
    @property
    def ny(self):
        return self._n_robots + self._n_robots * 2

    @functools.partial(jax.jit, static_argnames=("self",))
    @functools.partial(jax.vmap, in_axes=(None, 0, 0))
    def observation(self, x, pos_ref):
        x = x.reshape(self._n_robots, robot.NX)
        h_headings = x[:, 3].ravel()
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
        x = x.reshape((self._n_robots, robot.NX))
        h_headings = x[:, 3].ravel()
        pos_ref = x[0, 0:3]

        h_bearings = super().observation(x[1:, :], pos_ref).ravel()

        return jnp.concatenate([pos_ref, h_headings, h_bearings])
