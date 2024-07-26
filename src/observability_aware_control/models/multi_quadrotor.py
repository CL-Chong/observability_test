import functools

import jax
import jax.numpy as jnp

from . import model_base, quadrotor, rotation

DIM_LEADER_POS_OBS = 3
DIM_ATT_OBS = 4
DIM_ALT_OBS = 1
DIM_VEL_OBS = 3


class MultiQuadrotor(model_base.ModelBase):

    def __init__(
        self,
        n_robots,
        mass,
        input_kind="thrust",
    ):
        model_base.ModelBase.__init__(self)

        self._n_robots = n_robots
        self._mass = jnp.broadcast_to(mass, n_robots)

        self._state_dims = {"position": 3, "attitude": 4, "velocity": 3}

        self._input_kind = input_kind

    @property
    def state_dims(self):
        return self._state_dims

    @property
    def robot_nx(self):
        return quadrotor.NX

    @property
    def robot_nu(self):
        return (
            quadrotor.NU_THRUST_RATES
            if self._input_kind == "thrust"
            else quadrotor.NU_ACCEL_RATES
        )

    @property
    def n_robots(self):
        return self._n_robots

    def dynamics(self, x, u):
        x = jnp.reshape(x, (self._n_robots, -1))
        u = jnp.reshape(u, (self._n_robots, -1))

        dynamics = jax.vmap(
            functools.partial(quadrotor.dynamics, input_kind=self._input_kind)
        )
        return dynamics(x, u, self._mass).ravel()

    @property
    def nx(self):
        return self._n_robots * self.robot_nx

    @property
    def nu(self):
        return self._n_robots * self.robot_nu

    def observation(self, x, u, p=None):
        x = jnp.reshape(x, (self._n_robots, -1))
        leader_pos = x[0, 0:3]
        att = x[:, 3:7].ravel()
        vel = x[:, 7:10].ravel()
        return jnp.concatenate([leader_pos, att, vel])


class RangeBasedCooperativeQuadrotor(MultiQuadrotor):
    def observation(self, x, u, p=None):
        x = jnp.reshape(x, (self._n_robots, -1))
        leader_pos = x[0, 0:3]

        # Range measurement is NOT sensitive to rotation of the tracker platform
        relative_positions = leader_pos - x[1:, 0:3]
        relative_ranges = jnp.linalg.norm(relative_positions, axis=1)

        return jnp.concatenate([super().observation(x, u, p), relative_ranges])


class BearingsBasedCooperativeQuadrotor(MultiQuadrotor):
    def observation(self, x, u, p=None):
        x = jnp.reshape(x, (self._n_robots, -1))
        leader_pos = x[0, 0:3]

        # Bearings measurement IS sensitive to rotation of the tracker platform
        rotate = jax.vmap(
            functools.partial(rotation.quaternion_rotate_point, invert_rotation=True)
        )
        relative_positions = rotate(x[1:, 3:7], leader_pos - x[1:, 0:3])
        azimuth = jnp.arctan2(relative_positions[:, 1], relative_positions[:, 0])
        elevation = jnp.arctan2(
            relative_positions[:, 2],
            jnp.hypot(relative_positions[:, 1], relative_positions[:, 0]),
        )
        relative_bearings = jnp.column_stack([azimuth, elevation]).ravel()

        return jnp.concatenate([super().observation(x, u, p), relative_bearings])
