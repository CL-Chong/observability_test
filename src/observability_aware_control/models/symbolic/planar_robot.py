import casadi as cs

from .. import model_base

NX = 3  # (innermost) state dimension
NU = 2  # (innermost) input dimension
NY = 1  # (innermost) observation dimension


def dynamics(x, u):
    psi = x[2]
    v = u[0]
    w = u[1]

    c = cs.cos(psi)
    s = cs.sin(psi)

    return cs.vertcat(c * v, s * v, w)


def observation(x, pos_ref):
    psi = x[2]
    c = cs.cos(psi)
    s = cs.sin(psi)

    p_diff = pos_ref - x[0:2]

    hx = c * p_diff[0] + s * p_diff[1]
    hy = -s * p_diff[0] + c * p_diff[1]
    return cs.atan2(hy, hx)


class PlanarRobot(model_base.ModelBase):
    """
    Base class for a single planar robot, defining the state dynamics and
    dimensions
    """

    def dynamics(self, x, u):
        return dynamics(x, u)

    def observation(self, x, pos_ref):
        return observation(x, pos_ref)

    @property
    def nx(self):
        return NX

    @property
    def nu(self):
        return NU

    @property
    def ny(self):
        return NY
