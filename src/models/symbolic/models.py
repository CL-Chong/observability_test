import casadi as cs

from .. import model_base


class Robot(model_base.ModelBase):
    """
    Base class for a single planar robot, defining the state dynamics and
    dimensions
    """

    NX = 3  # (innermost) state dimension
    NU = 2  # (innermost) input dimension
    NY = 1  # (innermost) observation dimension

    def dynamics(self, x, u):
        psi = x[2]
        v = u[0]
        w = u[1]

        c = cs.cos(psi)
        s = cs.sin(psi)

        return cs.vertcat(c * v, s * v, w)

    def observation(self, x, pos_ref):
        psi = x[2]
        c = cs.cos(psi)
        s = cs.sin(psi)

        p_diff = x[0:2] - pos_ref

        hx = c * p_diff[0] - s * p_diff[1]
        hy = s * p_diff[0] + c * p_diff[1]
        return cs.atan2(hy, hx)

    @property
    def nx(self):
        return self.NX

    @property
    def nu(self):
        return self.NU

    @property
    def ny(self):
        return self.NY


class MultiRobot(model_base.ModelBase):
    """A system of multiple robots"""

    NX = 3  # (innermost) state dimension
    NU = 2  # (innermost) input dimension

    def __init__(self, n_robots: int):
        self._n_robots = n_robots
        self._robots = [Robot() for _ in range(self._n_robots)]

    @property
    def n_robots(self):
        return self._n_robots

    def dynamics(self, x, u):
        x = self._state_as2d(x)
        u = self._input_as2d(u)

        return cs.vertcat(
            *(
                robot.dynamics(x[:, idx], u[:, idx])
                for idx, robot in enumerate(self._robots)
            )
        )

    def _state_as2d(self, x):
        return cs.reshape(x, (Robot.NX, self._n_robots))

    def _input_as2d(self, u):
        return cs.reshape(u, (Robot.NU, self._n_robots))

    @property
    def nx(self):
        return self._n_robots * Robot.NX

    @property
    def nu(self):
        return self._n_robots * Robot.NU


class ReferenceSensingRobots(MultiRobot):
    @property
    def ny(self):
        return self._n_robots + self._n_robots * Robot.NY

    def observation(self, x, pos_ref):
        x = self._state_as2d(x)
        h_headings = x[2, :].T
        h_bearings = cs.MX.zeros((Robot.NY, self._n_robots))
        for idx, robot in enumerate(self._robots):
            h_bearings[:, idx] = robot.observation(x[:, idx], pos_ref)

        h_bearings = cs.reshape(h_bearings, (-1, 1))

        return cs.vertcat(h_headings, h_bearings)


class LeaderFollowerRobots(MultiRobot):
    @property
    def ny(self):
        return self._n_robots + (self._n_robots - 1) * Robot.NY

    def observation(self, x):
        x = self._state_as2d(x)
        h_headings = x[2, :].T
        h_bearings = cs.MX.zeros((Robot.NY, self._n_robots - 1))
        for idx, robot in enumerate(self._robots[1:], 1):
            h_bearings[:, idx - 1] = robot.observation(x[:, idx], x[0:2, 0])

        h_bearings = cs.reshape(h_bearings, (-1, 1))

        return cs.vertcat(h_headings, h_bearings)
