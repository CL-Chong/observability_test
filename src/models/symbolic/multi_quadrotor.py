import casadi as cs

from .. import model_base
from . import quadrotor


class MultiQuadrotor(model_base.ModelBase):
    DIM_ATT_OBS = 4
    DIM_BRNG_OBS = 2

    def __init__(self, n_robots, mass):
        self._n_robots = n_robots
        self._mass = mass

    @property
    def nx(self):
        return self._n_robots * quadrotor.NX

    @property
    def nu(self):
        return self._n_robots * quadrotor.NU

    @property
    def n_robots(self):
        return self._n_robots

    def dynamics(self, x, u):
        x = x.reshape(self._n_robots, quadrotor.NX)
        u = u.reshape(self._n_robots, quadrotor.NU)

        dynamics = cs.MX.zeros(quadrotor.NX, self._n_robots)
        for i in self._n_robots:
            dynamics[:, i] = quadrotor.dynamics(x[:, i], u[:, i], self._mass[i])
        return dynamics.reshape()

    @property
    def ny(self):
        return self._n_robots * (self.DIM_ATT_OBS + self.DIM_BRNG_OBS)

    def observation(self, x):
        x = x.reshape(self._n_robots, quadrotor.NX)
        h_att = x[:, 3:7].reshape()
        pos_ref = x[0, 0:2]

        h_bearings = cs.MX.zeros(self.DIM_BRNG_OBS, self._n_robots)
        for i in self._n_robots:
            h_bearings[:, i] = quadrotor.observation(x[:, i], pos_ref)
        return cs.vertcat(h_att, h_bearings.reshape())
