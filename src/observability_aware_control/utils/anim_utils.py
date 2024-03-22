import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation


class Animated2DTrajectory:
    def __init__(self, n_robots) -> None:
        # Disable pesky focus stealing
        # https://stackoverflow.com/questions/45729092/make-interactive-matplotlib-window-not-pop-to-front-on-each-update-windows-7
        self._enable_plot = True
        try:
            mpl.use("Qt5agg")
        except ImportError:
            self._enable_plot = False
        except ValueError:
            mpl.use("Qt4agg")
        self.t = np.array([])
        self.annotation = ""
        self.x = [np.array([])] * n_robots
        self.y = [np.array([])] * n_robots
        if self._enable_plot:
            self._fig, self._ax = plt.subplots()
            self._fig.tight_layout()
            self._anim = animation.FuncAnimation(
                self._fig, self.animate, save_count=100
            )

    @property
    def fig(self):
        return self._fig

    @property
    def anim(self):
        return self._anim

    def set_labels(self):
        self._ax.set_xlabel("X Position (m)")
        self._ax.set_ylabel("Y Position (m)")

    def animate(self, _):
        if not self._enable_plot:
            return ()
        self._ax.clear()

        for x, y in zip(self.x, self.y):
            self._ax.plot(x, y)
        self._ax.relim()
        self._ax.autoscale_view(True, True)

        an_x = np.mean(self._ax.get_xlim())
        ylim_v = self._ax.get_ylim()
        an_y = ylim_v[0] + 0.8 * np.diff(ylim_v)
        self._ax.annotate(self.annotation, (an_x, an_y), horizontalalignment="center")

        self.set_labels()
        return ()


class Animated3DTrajectory:
    def __init__(self, n_robots) -> None:
        # Disable pesky focus stealing
        # https://stackoverflow.com/questions/45729092/make-interactive-matplotlib-window-not-pop-to-front-on-each-update-windows-7
        self._enable_plot = True
        try:
            mpl.use("Qt5agg")
        except ImportError:
            self._enable_plot = False
        except ValueError:
            mpl.use("Qt4agg")
        self.t = np.array([])
        self.annotation = ""
        self.x = [np.array([])] * n_robots
        self.y = [np.array([])] * n_robots
        self.z = [np.array([])] * n_robots
        if self._enable_plot:
            self._fig, self._ax = plt.subplots(1, 2)
            self._fig.tight_layout()
            self._anim = animation.FuncAnimation(
                self._fig, self.animate, save_count=100
            )

    @property
    def fig(self):
        return self._fig

    @property
    def anim(self):
        return self._anim

    def set_labels(self):
        self._ax[0].set_xlabel("X Position (m)")
        self._ax[0].set_ylabel("Y Position (m)")
        self._ax[1].set_xlabel("Time (s)")
        self._ax[1].set_ylabel("Altitude (m)")

    def animate(self, _):
        if not self._enable_plot:
            return ()
        for it in self._ax:
            it.clear()

        for x, y, z in zip(self.x, self.y, self.z):
            self._ax[0].plot(x, y)
            self._ax[1].plot(self.t, z)
        for it in self._ax:
            it.relim()
            it.autoscale_view(True, True)

        an_x = np.mean(self._ax[0].get_xlim())
        ylim_v = self._ax[0].get_ylim()
        an_y = ylim_v[0] + 0.8 * np.diff(ylim_v)
        self._ax[0].annotate(
            self.annotation, (an_x, an_y), horizontalalignment="center"
        )

        self.set_labels()
        return ()
