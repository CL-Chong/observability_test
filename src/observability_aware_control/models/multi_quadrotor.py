import functools

import jax
import jax.numpy as jnp

from . import model_base, quadrotor


class MultiQuadrotor(model_base.MRSBase):
    DIM_LEADER_POS_OBS = 3
    DIM_ATT_OBS = 4
    DIM_BRNG_OBS = 2
    DIM_ALT_OBS = 1

    def __init__(self, n_robots, mass):
        self._n_robots = n_robots
        self._mass = jnp.broadcast_to(mass, n_robots)

    @property
    def robot_nx(self):
        return quadrotor.NX

    @property
    def robot_nu(self):
        return quadrotor.NU

    @property
    def n_robots(self):
        return self._n_robots

    def dynamics(self, x, u):
        x = self.reshape_x_vec(x)
        u = self.reshape_u_vec(u)

        dynamics = jax.vmap(quadrotor.dynamics, out_axes=0)
        return dynamics(x, u, self._mass).ravel()

    @property
    def ny(self):
        return (
            self.DIM_LEADER_POS_OBS
            + (self._n_robots - 1) * self.DIM_ALT_OBS
            + self._n_robots * self.DIM_ATT_OBS
            + (self._n_robots - 1) * self.DIM_BRNG_OBS
        )

    def observation(self, x):
        x = self.reshape_x_vec(x)
        h_att = x[:, 3:7].ravel()
        pos_ref = x[0, 0:3]
        alt = x[1:, 2]

        obs = jax.vmap(quadrotor.observation, in_axes=(0, None))

        h_bearings = obs(x[1:, :], pos_ref).ravel()
        return jnp.concatenate([pos_ref, alt, h_att, h_bearings])
