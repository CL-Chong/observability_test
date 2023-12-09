import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation


class Animated3DTrajectory:
    def __init__(self, n_robots) -> None:
        # Disable pesky focus stealing
        # https://stackoverflow.com/questions/45729092/make-interactive-matplotlib-window-not-pop-to-front-on-each-update-windows-7
        try:
            mpl.use("Qt5agg")
        except ValueError:
            mpl.use("Qt4agg")
        self.t = np.array([])
        self.annotation = ""
        self.x = [np.array([])] * n_robots
        self.y = [np.array([])] * n_robots
        self.z = [np.array([])] * n_robots
        self._fig, self._ax = plt.subplots(1, 2)
        self._fig.tight_layout()
        self._anim = animation.FuncAnimation(self._fig, self.animate, save_count=100)

    def set_labels(self):
        self._ax[0].set_xlabel("X Position (m)")
        self._ax[0].set_ylabel("Y Position (m)")
        self._ax[1].set_xlabel("Time (s)")
        self._ax[1].set_ylabel("Altitude (m)")

    def animate(self, _):
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
