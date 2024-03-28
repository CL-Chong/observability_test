import functools

import jax
import jax.numpy as jnp


@functools.partial(jax.jit, static_argnums=(2,))
def angle_rotate_point(angle, point, invert_rotation=False):
    if invert_rotation:
        angle = -angle
    c = jnp.cos(angle)
    s = jnp.sin(angle)

    return jnp.array(
        [
            c * point[0] + s * point[1],
            -s * point[0] + c * point[1],
        ]
    )


@jax.jit
def quaternion_product(lhs, rhs):
    return jnp.array(
        [
            lhs[3] * rhs[0] + lhs[0] * rhs[3] + lhs[1] * rhs[2] - lhs[2] * rhs[1],
            lhs[3] * rhs[1] + lhs[1] * rhs[3] + lhs[2] * rhs[0] - lhs[0] * rhs[2],
            lhs[3] * rhs[2] + lhs[2] * rhs[3] + lhs[0] * rhs[1] - lhs[1] * rhs[0],
            lhs[3] * rhs[3] - lhs[0] * rhs[0] - lhs[1] * rhs[1] - lhs[2] * rhs[2],
        ]
    )


@functools.partial(jax.jit, static_argnums=(2,))
def quaternion_rotate_point(quaternion, point, invert_rotation=False):
    vec = -quaternion[0:3] if invert_rotation else quaternion[0:3]
    uv = jnp.cross(vec, point)
    uv += uv
    return point + quaternion[3] * uv + jnp.cross(vec, uv)
