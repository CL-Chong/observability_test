import functools
import math

import jax
import jax.numpy as jnp

from ..algorithms import stlog
from . import model_base, robot

DIM_LEADER_POS_OBS = 2
DIM_HDG_OBS = 1
DIM_RNG_OBS = 1
DIM_BRNG_OBS = 1


class MultiRobot(model_base.MRSBase, stlog.STLOG):
    """A system of multiple robots"""

    NX = robot.NX  # (innermost) state dimension
    NU = robot.NU  # (innermost) input dimension

    def __init__(self, n_robots, stlog_order, *, stlog_cov=None):
        stlog.STLOG.__init__(self, stlog_order, stlog_cov)
        self._n_robots = n_robots

        self._state_dims = {"position": 2, "heading": 1}
        self._observation_dims = {
            "leader_position": 2,
            "heading": self._n_robots,
            "bearing": math.comb(self._n_robots, 2),
        }
        self._ny = sum(self._observation_dims.values())

    @property
    def state_dims(self):
        return self._state_dims

    @property
    def robot_nx(self):
        return robot.NX

    @property
    def robot_nu(self):
        return robot.NU

    @property
    def n_robots(self):
        return self._n_robots

    @property
    def nx(self):
        return self._n_robots * robot.NX

    @property
    def nu(self):
        return self._n_robots * robot.NU

    @property
    def ny(self):
        return self._ny

    @functools.partial(jax.jit, static_argnames=("self",))
    def dynamics(self, x, u):
        x = self.reshape_x_vec(x)
        u = self.reshape_u_vec(u)

        dynamics = jax.vmap(robot.dynamics)
        return dynamics(x, u).ravel()

    def observation(self, x):
        x = self.reshape_x_vec(x)
        pos_ref = x[0, 0:2]
        h_headings = x[:, 2]

        obs = jax.vmap(robot.observation, in_axes=(0, None))

        h_bearings = obs(x[1:, :], pos_ref).ravel()

        return jnp.concatenate([pos_ref, h_headings, h_bearings])
