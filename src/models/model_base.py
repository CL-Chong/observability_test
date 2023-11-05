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
