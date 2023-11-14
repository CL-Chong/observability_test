import functools

import jax
import jax.numpy as jnp

NX = 10
NU = 4


def _fast_cross(a, b):
    return jnp.array(
        [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
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
    uv = _fast_cross(vec, point)
    uv += uv
    return point + quaternion[3] * uv + _fast_cross(vec, uv)


@jax.jit
def dynamics(x, u, mass):
    q = x[3:7]
    v = x[7:10]

    f = jnp.array([0.0, 0.0, u[0] / mass])
    w = jnp.array([u[1], u[2], u[3], 0.0]) / 2.0
    g = jnp.array([0.0, 0.0, -9.81])

    dx = jnp.empty(NX)
    dx = dx.at[0:3].set(v)
    dx = dx.at[3:7].set(quaternion_product(q, w))
    dx = dx.at[7:10].set(quaternion_rotate_point(q, f) + g)
    return dx


@jax.jit
def observation(x, pos_ref):
    p = x[0:3]
    q = x[3:7]
    p_diff = quaternion_rotate_point(q, pos_ref - p, True)
    azimuth = jnp.arctan2(p_diff[1], p_diff[0])
    elevation = jnp.arctan2(p_diff[2], jnp.hypot(p_diff[0], p_diff[1]))
    return jnp.array([azimuth, elevation])
