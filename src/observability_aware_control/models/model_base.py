import abc

from observability_aware_control.models import sensors


class ModelBase(abc.ABC):
    """Base class (Interface) for all nonlinear dynamical system modesl"""

    @abc.abstractmethod
    def dynamics(self, x, u, *args): ...

    @abc.abstractmethod
    def observation(self, x, *args): ...

    @property
    @abc.abstractmethod
    def nx(self):
        return -1

    @property
    @abc.abstractmethod
    def nu(self):
        return -1


class MRSBase(ModelBase):

    def __init__(self, interrobot_observation_kind):
        self._interrobot_observation_dim = sensors.DIM_INTERROBOT_OBSERVATION[
            interrobot_observation_kind
        ]
        self._intrinsics = getattr(sensors, interrobot_observation_kind)

    def interrobot_observation(self, tracker_state, target_position):
        return self._intrinsics(sensors.extrinsics(tracker_state, target_position))

    @property
    def interrobot_observation_dim(self):
        return self._interrobot_observation_dim

    @property
    @abc.abstractmethod
    def robot_nx(self):
        return -1

    @property
    def nx(self):
        return self.n_robots * self.robot_nx

    @property
    @abc.abstractmethod
    def robot_nu(self):
        return -1

    @property
    def nu(self):
        return self.n_robots * self.robot_nu

    @property
    @abc.abstractmethod
    def n_robots(self):
        return -1

    def reshape_x_vec(self, x_vec):
        return x_vec.reshape(self.n_robots, self.robot_nx)

    def reshape_u_vec(self, u_vec):
        return u_vec.reshape(self.n_robots, self.robot_nu)
