import jax
import jax.numpy as jnp

from ..algorithms import stlog
from . import model_base, quadrotor

DIM_LEADER_POS_OBS = 3
DIM_ATT_OBS = 4
DIM_ALT_OBS = 1
DIM_VEL_OBS = 3


class MultiQuadrotor(model_base.MRSBase, stlog.STLOG):

    def __init__(
        self,
        n_robots,
        mass,
        stlog_order,
        has_baro=False,
        has_odom=False,
        stlog_cov=None,
        interrobot_observation_kind="bearings",
    ):
        model_base.MRSBase.__init__(self, interrobot_observation_kind)
        stlog.STLOG.__init__(self, stlog_order, stlog_cov)

        self._n_robots = n_robots
        self._mass = jnp.broadcast_to(mass, n_robots)
        self._has_baro = has_baro
        self._has_odom = has_odom

        self._state_dims = {"position": 3, "attitude": 4, "velocity": 3}

    @property
    def state_dims(self):
        return self._state_dims

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

        dynamics = jax.vmap(quadrotor.dynamics)
        return dynamics(x, u, self._mass).ravel()

    @property
    def nx(self):
        return self._n_robots * self.robot_nx

    @property
    def nu(self):
        return self._n_robots * self.robot_nu

    def observation(self, x):
        x = self.reshape_x_vec(x)
        pos_ref = x[0, 0:3]
        att = x[:, 3:7].ravel()
        obs = jax.vmap(self.interrobot_observation, in_axes=(0, None))

        h_bearings = obs(x[1:, :], pos_ref).ravel()

        res = (pos_ref, att, h_bearings)
        if self._has_baro:
            alt = x[1:, 2]
            res += (alt,)
        if self._has_odom:
            vel = x[:, 7:10].ravel()
            res += (vel,)
        return jnp.concatenate(res)
