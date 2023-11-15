import casadi as cs

from .. import model_base
from . import planar_robot


class MultiRobot(model_base.ModelBase):
    """A system of multiple robots"""

    NX = 3  # (innermost) state dimension
    NU = 2  # (innermost) input dimension

    def __init__(self, n_robots: int):
        self._n_robots = n_robots

    @property
    def n_robots(self):
        return self._n_robots

    def dynamics(self, x, u):
        x = x.reshape((planar_robot.NX, self._n_robots))
        u = u.reshape((planar_robot.NU, self._n_robots))

        dx = cs.MX.zeros((planar_robot.NX, self._n_robots))
        for idx in range(self._n_robots):
            dx[:, idx] = planar_robot.dynamics(x[:, idx], u[:, idx])

        return dx.reshape((-1, 1))

    @property
    def nx(self):
        return self._n_robots * planar_robot.NX

    @property
    def nu(self):
        return self._n_robots * planar_robot.NU


class ReferenceSensingRobots(MultiRobot):
    @property
    def ny(self):
        return self._n_robots + self._n_robots * planar_robot.NY

    def observation(self, x, pos_ref):
        x = x.reshape((planar_robot.NX, self._n_robots))
        h_headings = x[2, :].T
        h_bearings = cs.MX.zeros((planar_robot.NY, self._n_robots))
        for idx, robot in enumerate(self._robots):
            h_bearings[:, idx] = planar_robot.observation(x[:, idx], pos_ref)

        h_bearings = cs.reshape(h_bearings, (-1, 1))

        return cs.vertcat(h_headings, h_bearings)


class LeaderFollowerRobots(MultiRobot):
    @property
    def ny(self):
        return self._n_robots + (self._n_robots - 1) * planar_robot.NY + 2

    def observation(self, x):
        x = x.reshape((planar_robot.NX, self._n_robots))
        h_headings = x[2, :].T
        h_bearings = cs.MX.zeros((planar_robot.NY, self._n_robots - 1))
        pos_ref = x[0:2, 0]
        for idx in range(1, self._n_robots):
            h_bearings[:, idx - 1] = planar_robot.observation(x[:, idx], pos_ref)

        h_bearings = cs.reshape(h_bearings, (-1, 1))

        return cs.vertcat(pos_ref, h_headings, h_bearings)
