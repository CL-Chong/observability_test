import abc


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


class MRSBase(ModelBase):
    @property
    @abc.abstractmethod
    def robot_nx(self):
        raise NotImplementedError("This is a base class")

    @property
    def nx(self):
        return self.n_robots * self.robot_nx

    @property
    @abc.abstractmethod
    def robot_nu(self):
        raise NotImplementedError("This is a base class")

    @property
    def nu(self):
        return self.n_robots * self.robot_nu

    @property
    @abc.abstractmethod
    def n_robots(self):
        raise NotImplementedError("This is a base class")

    def reshape_x_vec(self, x_vec):
        return x_vec.reshape(self.n_robots, self.robot_nx)

    def reshape_u_vec(self, u_vec):
        return u_vec.reshape(self.n_robots, self.robot_nu)
