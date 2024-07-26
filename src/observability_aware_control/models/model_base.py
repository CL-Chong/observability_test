import abc

from observability_aware_control.models import sensors


class ModelBase(abc.ABC):
    """Base class (Interface) for all nonlinear dynamical system modesl"""

    @abc.abstractmethod
    def dynamics(self, x, u, p):
        """System dynamics equation describing the evolution of the state by an
        ODE

        Parameters
        ----------
        x : ArrayLike
            State of the system
        u : ArrayLike
            Input to the system
        p : ArrayLike
            Online data/parameters for the system

        Returns
        -------

        """
        pass

    @abc.abstractmethod
    def observation(self, x, u, p):
        """Observation/output equation describing observations into the
        system/output of sensors

        Parameters
        ----------
        x : ArrayLike
            State of the system
        u : ArrayLike
            Input to the system
        p : ArrayLike
            Online data/parameters for the system

        Returns
        -------

        """

        return

    @property
    @abc.abstractmethod
    def nx(self):
        return -1

    @property
    @abc.abstractmethod
    def nu(self):
        return -1
