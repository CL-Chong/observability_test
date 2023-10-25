import functools
import math

import casadi as cs
import numpy as np


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
        else:
            self._cos = math.cos
            self._sin = math.sin
            self._atan2 = math.atan2
            self._zeros = np.zeros
            self._reshape = functools.partial(np.reshape, order="F")
            self._cat = np.concatenate


class Robot(MathFunctions):
    """
    Base class for a single planar robot, defining the state dynamics and
    dimensions
    """

    NX = 3  # (innermost) state dimension
    NU = 2  # (innermost) input dimension

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


class MultiRobot(Robot):
    """A system of multiple robots"""

    def __init__(self, n_robots, is_symbolic=False):
        self._n_robots = n_robots
        super().__init__(is_symbolic)

    @property
    def n_robots(self):
        return self._n_robots

    def dynamics(self, x, u_control_t, u_drone_t):
        x = self._state_as2d(x)
        u_drone_t = self._input_as2d(u_drone_t)

        f = self._zeros((self.NX, self._n_robots))
        for idx in range(self._n_robots):
            u = u_control_t if idx == 0 else u_drone_t[:, idx - 1]
            f[:, idx] = super(MultiRobot, self).dynamics(x[:, idx], u)
        return self._reshape(f, (-1, 1))

    def observation(self, x):
        x = self._state_as2d(x)
        h_abs_pos = x[0:2, 0]
        h_headings = x[2, :].T
        h_bearings = []
        for idx in range(1, self._n_robots):
            h_bearings.append(super(MultiRobot, self).observation(x[:, idx], x[0:2, 0]))

        return self._cat((h_abs_pos, h_headings, *h_bearings))

    def _state_as2d(self, x):
        return self._reshape(x, (self.NX, self._n_robots))

    def _input_as2d(self, u):
        return self._reshape(u, (self.NU, self._n_robots - 1))
