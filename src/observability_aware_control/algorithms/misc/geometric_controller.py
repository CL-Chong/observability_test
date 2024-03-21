import abc
from typing import NamedTuple

import jax
import jax.numpy as jnp
import jax.numpy.linalg as la
from jax.scipy.spatial.transform import Rotation


class AcclerationSetpointShaping:
    @property
    @abc.abstractmethod
    def min_z_accel(self) -> float:
        pass

    @property
    @abc.abstractmethod
    def max_z_accel(self) -> float:
        pass

    @property
    @abc.abstractmethod
    def max_tilt_ratio(self) -> float:
        pass

    def reshape_acceleration_setpoint(self, acc_sp):
        def saturated():
            # Lift alone saturates actuators, deliver as much lift as possible and
            # no lateral acc
            return jnp.array([0.0, 0.0, self.max_z_accel])

        def unsaturated():
            z_sp = jnp.maximum(acc_sp[2], self.min_z_accel)
            # Lift does not saturate actuators, aim to deliver requested lift
            # exactly while scaling back lateral acc
            max_lateral_acc = jnp.sqrt(self.max_z_accel**2 - z_sp**2)

            lateral_acc_at_max_tilt = abs(z_sp) * self.max_tilt_ratio
            max_lateral_acc = jnp.minimum(max_lateral_acc, lateral_acc_at_max_tilt)

            xy_sp = acc_sp[0:2]
            lateral_acc_sqnorm = xy_sp @ xy_sp

            xy_sp = jax.lax.select(
                lateral_acc_sqnorm > max_lateral_acc**2,
                xy_sp * max_lateral_acc / jnp.sqrt(lateral_acc_sqnorm),
                xy_sp,
            )
            return jnp.array([xy_sp[0], xy_sp[1], z_sp])

        return jax.lax.cond(acc_sp[2] < self.max_z_accel, unsaturated, saturated)


def acceleration_vector_to_rotation(acc_sp, yaw):
    proj_xb_des = jnp.array([jnp.cos(yaw), jnp.sin(yaw), 0.0])
    zb_des = acc_sp / la.norm(acc_sp)
    yb_des = jnp.cross(zb_des, proj_xb_des)
    yb_des /= la.norm(yb_des)
    xb_des = jnp.cross(yb_des, zb_des)
    xb_des /= la.norm(xb_des)

    return jnp.column_stack([xb_des, yb_des, zb_des])


class AttitudeControllerState(NamedTuple):
    attitude: jnp.ndarray


class AttitudeControllerReference(NamedTuple):
    attitude: jnp.ndarray


class AttitudeControllerOutput(NamedTuple):
    body_rate: jnp.ndarray


class AttitudeControllerError(NamedTuple):
    attitude_error: jnp.ndarray


class AttitudeController:
    State = AttitudeControllerState
    Reference = AttitudeControllerReference

    def __init__(self, attctrl_tau=0.1):
        self._attctrl_tau = attctrl_tau

    def run(self, state: AttitudeControllerState, refs: AttitudeControllerReference):
        rotmat = Rotation.from_quat(state.attitude).as_matrix()
        rotmat_sp = Rotation.from_quat(refs.attitude).as_matrix()

        def vee(mat):
            return jnp.array([mat[2, 1], mat[0, 2], mat[1, 0]])

        attitude_error = 0.5 * vee(rotmat_sp.T @ rotmat - rotmat.T @ rotmat_sp)
        body_rate_sp = -(2.0 / self._attctrl_tau) * attitude_error

        return (
            AttitudeControllerOutput(body_rate_sp),
            AttitudeControllerError(attitude_error),
        )


class TrackingControllerParams(NamedTuple):
    k_pos: jnp.ndarray = jnp.array([1.0, 1.0, 10.0])
    k_vel: jnp.ndarray = jnp.array([1.5, 1.5, 3.3])
    drag_d: jnp.ndarray = jnp.zeros((3, 3))
    min_z_accel: float = 0.0
    max_z_accel: float = 20.0
    max_tilt_ratio: float = float(jnp.deg2rad(45.0))


class TrackingControllerState(NamedTuple):
    position: jnp.ndarray
    attitude: jnp.ndarray
    velocity: jnp.ndarray


class TrackingControllerReference(NamedTuple):
    position: jnp.ndarray
    velocity: jnp.ndarray
    acceleration: jnp.ndarray = jnp.zeros(3)
    yaw: float = 0.0


class TrackingControllerOutput(NamedTuple):
    thrust: float
    orientation: jnp.ndarray


class TrackingControllerError(NamedTuple):
    position_error: jnp.ndarray
    velocity_error: jnp.ndarray


class TrackingController(AcclerationSetpointShaping):
    GRAVITY = jnp.array([0.0, 0.0, 9.81])

    State = TrackingControllerState
    Reference = TrackingControllerReference
    Params = TrackingControllerParams

    def __init__(self, params=None) -> None:
        self._params = params if params is not None else self.Params()

    @property
    def params(self):
        return self._params

    @property
    def drag_d(self):
        return self.params.drag_d

    @property
    def min_z_accel(self) -> float:
        return self.params.min_z_accel

    @property
    def max_z_accel(self) -> float:
        return self.params.max_z_accel

    @property
    @abc.abstractmethod
    def max_tilt_ratio(self) -> float:
        return self.params.max_tilt_ratio

    def run(self, state: TrackingControllerState, refs: TrackingControllerReference):
        # Position Controller

        position_error = state.position - refs.position
        velocity_error = state.velocity - refs.velocity

        feedback = (
            self.params.k_pos * position_error + self.params.k_vel * velocity_error
        )
        # Reference acceleration
        accel_sp = self.reshape_acceleration_setpoint(
            -feedback + self.GRAVITY + refs.acceleration
        )
        attitude_sp = acceleration_vector_to_rotation(accel_sp, refs.yaw)
        thrust_sp = accel_sp @ attitude_sp @ jnp.array([0.0, 0.0, 1.0])

        return (
            TrackingControllerOutput(
                thrust_sp, Rotation.from_matrix(attitude_sp).as_quat()
            ),
            TrackingControllerError(position_error, velocity_error),
        )
