import abc
import functools
import math

import casadi as cs
import numpy as np


class ModelBase(abc.ABC):
    """Base class (Interface) for all nonlinear dynamical system modesl"""

    @abc.abstractmethod
    def dynamics(self, x, u, *args):
        ...

    @abc.abstractmethod
    def observation(self, x, *args):
        ...

    @property
    @abc.abstractmethod
    def nx(self):
        return -1

    @property
    @abc.abstractmethod
    def nu(self):
        return -1

    @property
    @abc.abstractmethod
    def ny(self):
        return -1


class MathFunctions:
    """A base class that binds common mathematical functions to either symbolic
    or numerical implementations to an instance ahead of time
    """

    def __init__(self, is_symbolic=False):
        """
        Creates an instance with either symbolic or numeric function bindings

        Parameters
        ----------
        is_symbolic : bool, optional
            Toggles symbolic or numerical implementation, by default False
        """

        self._is_symbolic = is_symbolic
        if is_symbolic:
            self._cos = cs.cos
            self._sin = cs.sin
            self._atan2 = cs.atan2
            self._zeros = cs.MX.zeros
            self._reshape = cs.MX.reshape
            self._cat = lambda args: cs.vertcat(*args)
            self._squeeze = lambda _: _
        else:
            self._cos = math.cos
            self._sin = math.sin
            self._atan2 = math.atan2
            self._zeros = np.zeros
            self._reshape = functools.partial(np.reshape, order="F")
            self._cat = np.concatenate
            self._squeeze = np.squeeze


class Robot(MathFunctions, ModelBase):
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

        c = self._cos(psi)
        s = self._sin(psi)

        dx = self._zeros(self.NX)
        dx[0] = c * v
        dx[1] = s * v
        dx[2] = w
        return dx

    def observation(self, x, pos_ref):
        psi = x[2]
        c = self._cos(psi)
        s = self._sin(psi)

        p_diff = x[0:2] - pos_ref

        hx = c * p_diff[0] - s * p_diff[1]
        hy = s * p_diff[0] + c * p_diff[1]
        return self._atan2(hy, hx)

    @property
    def nx(self):
        return self.NX

    @property
    def nu(self):
        return self.NU

    @property
    def ny(self):
        return self.NY


class MultiRobot(MathFunctions, ModelBase):
    """A system of multiple robots"""

    NX = 3  # (innermost) state dimension
    NU = 2  # (innermost) input dimension

    def __init__(self, n_robots, is_symbolic=False):
        self._n_robots = n_robots
        self._robots = [Robot(is_symbolic) for _ in range(self._n_robots)]
        super().__init__(is_symbolic)

    @property
    def n_robots(self):
        return self._n_robots

    def dynamics(self, x, u):
        x = self._state_as2d(x)
        u = self._input_as2d(u)

        f = self._zeros((Robot.NX, self._n_robots))
        for idx, robot in enumerate(self._robots):
            f[:, idx] = robot.dynamics(x[:, idx], u[:, idx])
        return self._squeeze(self._reshape(f, (-1, 1)))

    def _state_as2d(self, x):
        return self._reshape(x, (Robot.NX, self._n_robots))

    def _input_as2d(self, u):
        return self._reshape(u, (Robot.NU, self._n_robots))

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
        h_bearings = self._zeros((Robot.NY, self._n_robots))
        for idx, robot in enumerate(self._robots):
            h_bearings[:, idx] = robot.observation(x[:, idx], pos_ref)

        h_bearings = self._squeeze(self._reshape(h_bearings, (-1, 1)))

        return self._cat((h_headings, h_bearings))


class LeaderFollowerRobots(MultiRobot):
    @property
    def ny(self):
        return self._n_robots + (self._n_robots - 1) * Robot.NY

    def observation(self, x):
        x = self._state_as2d(x)
        h_headings = x[2, :].T
        h_bearings = self._zeros((Robot.NY, self._n_robots - 1))
        for idx, robot in enumerate(self._robots[1:], 1):
            h_bearings[:, idx - 1] = robot.observation(x[:, idx], x[0:2, 0])

        h_bearings = self._squeeze(self._reshape(h_bearings, (-1, 1)))

        return self._cat((h_headings, h_bearings))
