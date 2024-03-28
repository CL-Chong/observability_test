import jax
import jax.numpy as jnp

from .rotation import angle_rotate_point, quaternion_rotate_point

DIM_INTERROBOT_OBSERVATION = {"distance": 1, "bearings": 2}


@jax.jit
def extrinsics(tracker_state, target_position):
    if len(tracker_state) == 3:
        tracker_position = tracker_state[0:2]
        tracker_heading = tracker_state[2]
        return angle_rotate_point(
            tracker_heading, target_position - tracker_position, invert_rotation=True
        )

    elif len(tracker_state) in (10, 13):
        tracker_position = tracker_state[0:3]
        tracker_attitude = tracker_state[3:7]
        return quaternion_rotate_point(
            tracker_attitude, target_position - tracker_position, invert_rotation=True
        )

    else:
        raise ValueError(
            "Dimension of state does not match either a 2D kinematics model or a 3D"
            "rigid-body dynamics model"
        )


@jax.jit
def bearings(point):
    azimuth = jnp.arctan2(point[1], point[0])
    if len(point) == 2:
        return azimuth
    elif len(point) == 3:
        elevation = jnp.arctan2(point[2], jnp.hypot(point[0], point[1]))
        return jnp.array([azimuth, elevation])
    else:
        raise ValueError(
            "Dimension of observation does not match either 2D or 3D interrobot"
            "observation"
        )


@jax.jit
def distance(point):
    return jnp.linalg.norm(point)
