import casadi as cs


def quaternion_product(lhs, rhs):
    return cs.vertcat(
        lhs[3] * rhs[0] + lhs[0] * rhs[3] + lhs[1] * rhs[2] - lhs[2] * rhs[1],
        lhs[3] * rhs[1] + lhs[1] * rhs[3] + lhs[2] * rhs[0] - lhs[0] * rhs[2],
        lhs[3] * rhs[2] + lhs[2] * rhs[3] + lhs[0] * rhs[1] - lhs[1] * rhs[0],
        lhs[3] * rhs[3] - lhs[0] * rhs[0] - lhs[1] * rhs[1] - lhs[2] * rhs[2],
    )


def quaternion_rotate_point(quaternion, point, invert_rotation=False):
    vec = -quaternion[0:3] if invert_rotation else quaternion[0:3]
    uv = cs.cross(vec, point)
    uv += uv
    return point + quaternion[3] * uv + cs.cross(vec, uv)


NX = 10
NU = 4


def dynamics(x, u, mass):
    q = x[3:7]
    v = x[7:10]

    f = cs.vertcat(0.0, 0.0, u[0] / mass)
    w = cs.vertcat(u[1], u[2], u[3], 0.0) / 2.0
    g = cs.vertcat(0.0, 0.0, -9.81)

    dx = cs.vertcat(
        v,
        quaternion_product(q, w),
        quaternion_rotate_point(q, f) + g,
    )
    return dx


def observation(x, pos_ref):
    p = x[0:3]
    q = x[3:7]
    p_diff = quaternion_rotate_point(q, pos_ref - p, True)
    azimuth = cs.atan2(p_diff[1], p_diff[0])
    elevation = cs.atan2(p_diff[2], cs.hypot(p_diff[0], p_diff[1]))
    return cs.vertcat(azimuth, elevation)