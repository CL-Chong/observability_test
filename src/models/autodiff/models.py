import functools

import jax
import jax.numpy as jnp

from .. import model_base


class Robot:
    NX = 3
    NU = 2
    NY = 1

    @staticmethod
    @jax.jit
    def dynamics(x, u):
        psi = x[2]
        v = u[0]
        w = u[1]

        c = jnp.cos(psi)
        s = jnp.sin(psi)

        return jnp.array([c * v, s * v, w])

    @staticmethod
    @jax.jit
    def observation(x, pos_ref):
        psi = x[2]
        c = jnp.cos(psi)
        s = jnp.sin(psi)

        p_diff = x[0:2] - pos_ref

        hx = c * p_diff[0] - s * p_diff[1]
        hy = s * p_diff[0] + c * p_diff[1]
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


class MultiRobot(model_base.ModelBase):
    def __init__(self, n_robots):
        self._n_robots = n_robots

    @functools.partial(jax.vmap, in_axes=(None, 0, None))
    def observation(self, x, pos_ref):
        return Robot.observation(x, pos_ref)

    @property
    def nx(self):
        return self._n_robots * Robot.NX

    @property
    def nu(self):
        return self._n_robots * Robot.NU

    @property
    def n_robots(self):
        return self._n_robots

    @functools.partial(jax.jit, static_argnames=("self",))
    def dynamics(self, x, u):
        x = x.reshape(self._n_robots, Robot.NX)
        u = u.reshape(self._n_robots, Robot.NU)

        dynamics = jax.vmap(Robot.dynamics, out_axes=0)
        return dynamics(x, u).ravel()


class ReferenceSensingRobots(MultiRobot):
    @property
    def ny(self):
        return self._n_robots + (self._n_robots - 1) * 2

    @functools.partial(jax.jit, static_argnames=("self",))
    @functools.partial(jax.vmap, in_axes=(None, 0, 0))
    def observation(self, x, pos_ref):
        x = x.reshape(self._n_robots, Robot.NX)
        h_headings = x[:, 2].ravel()
        h_bearings = super().observation(x, pos_ref)

        return jnp.concatenate([h_headings, h_bearings])


class LeaderFollowerRobots(MultiRobot):
    @property
    def ny(self):
        return self._n_robots + (self._n_robots - 1) * 2

    @functools.partial(jax.jit, static_argnames=("self",))
    @functools.partial(jax.vmap, in_axes=(None, 0))
    def observation(self, x):
        x = x.reshape(self._n_robots, Robot.NX)
        h_headings = x[:, 2].ravel()
        pos_ref = x[0, 0:2]

        h_bearings = super().observation(x[1:, :], pos_ref)

        return jnp.concatenate([h_headings, h_bearings])
