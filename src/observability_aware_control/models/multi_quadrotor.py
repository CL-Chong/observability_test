import functools

import jax
import jax.numpy as jnp

from . import model_base, quadrotor


class MultiQuadrotor(model_base.MRSBase):
    DIM_LEADER_POS_OBS = 3
    DIM_ATT_OBS = 4
    DIM_BRNG_OBS = 2
    DIM_ALT_OBS = 1
    DIM_VEL_OBS = 3

    def __init__(self, n_robots, mass, has_baro=False, has_odom=False):
        self._n_robots = n_robots
        self._mass = jnp.broadcast_to(mass, n_robots)
        self._has_baro = has_baro
        self._has_odom = has_odom
        self._ny = (
            self.DIM_LEADER_POS_OBS
            + self._n_robots * self.DIM_ATT_OBS
            + (self._n_robots - 1) * self.DIM_BRNG_OBS
        )
        if has_baro:
            self._ny += (self._n_robots - 1) * self.DIM_ALT_OBS
        if has_odom:
            self._ny += self._n_robots * self.DIM_VEL_OBS

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
        return self._ny

    def observation(self, x):
        x = self.reshape_x_vec(x)
        pos_ref = x[0, 0:3]
        att = x[:, 3:7].ravel()
        obs = jax.vmap(quadrotor.observation, in_axes=(0, None))

        h_bearings = obs(x[1:, :], pos_ref).ravel()

        res = (pos_ref, att, h_bearings)
        if self._has_baro:
            alt = x[1:, 2]
            res += (alt,)
        if self._has_odom:
            vel = x[:, 7:10].ravel()
            res += (vel,)
        return jnp.concatenate(res)

        # return jnp.concatenate([pos_ref, alt, h_att, h_bearings])
