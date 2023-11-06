import functools

import jax
import jax.numpy as jnp

from .. import model_base


class Robot(model_base.ModelBase):
    NX = 3
    NU = 2

    @staticmethod
    @jax.jit
    @functools.partial(jax.vmap, in_axes=(1, 1), out_axes=1)
    def dynamics(x, u):
        psi = x[2]
        v = u[0]
        w = u[1]

        c = jnp.cos(psi)
        s = jnp.sin(psi)

        return jnp.array([c * v, s * v, w])

    @staticmethod
    @jax.jit
    @functools.partial(jax.vmap, in_axes=(1, None), out_axes=0)
    def observation(x, pos_ref):
        psi = x[2]
        c = jnp.cos(psi)
        s = jnp.sin(psi)

        p_diff = x[0:2] - pos_ref

        hx = c * p_diff[0] - s * p_diff[1]
        hy = s * p_diff[0] + c * p_diff[1]
        return jnp.arctan2(hy, hx)


class MultiRobot(model_base.ModelBase):
    def __init__(self, n_robots):
        self._n_robots = n_robots

    @property
    def nx(self):
        return self._n_robots * Robot.NX

    @property
    def nu(self):
        return self._n_robots * Robot.NU

    @property
    def ny(self):
        return self._n_robots + (self._n_robots - 1) * 2

    @property
    def n_robots(self):
        return self._n_robots

    @functools.partial(jax.jit, static_argnames=("self",))
    def dynamics(self, x, u):
        x = jnp.reshape(x, (Robot.NX, -1), order="F")
        u = jnp.reshape(u, (Robot.NU, -1), order="F")

        return Robot.dynamics(x, u).ravel(order="F")

    @functools.partial(jax.jit, static_argnames=("self",))
    @functools.partial(jax.vmap, in_axes=(None, 1), out_axes=1)
    def observation(self, x):
        x = jnp.reshape(x, (Robot.NX, -1), order="F")
        h_headings = x[2, :].T
        pos_ref = x[0:2, 0]
        h_bearings = Robot.observation(x[:, 1:], pos_ref)

        return jnp.concatenate([h_headings, h_bearings])
