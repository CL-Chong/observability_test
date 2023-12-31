import dataclasses
import pathlib
import time
from typing import Any, Callable, Dict, Literal, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from jax.typing import ArrayLike
from scipy import optimize

Method = Union[
    Literal["Nelder-Mead"],
    Literal["Powell"],
    Literal["CG"],
    Literal["BFGS"],
    Literal["Newton-CG"],
    Literal["L-BFGS-B"],
    Literal["TNC"],
    Literal["COBYLA"],
    Literal["SLSQP"],
    Literal["trust-constr"],
    Literal["dogleg"],
    Literal["trust-ncg"],
    Literal["trust-exact"],
    Literal["trust-krylov"],
]

FiniteDifferenceMethods = Union[
    Literal["2-point"],
    Literal["3-point"],
    Literal["cs"],
]


@dataclasses.dataclass
class MinimizeProblem:
    """Dataclass container for parameters to scipy.optimize.minimize"""

    fun: Callable

    x0: ArrayLike

    args: Tuple[Any, ...] = dataclasses.field(default=())
    bounds: Optional[optimize.Bounds] = dataclasses.field(default=None)

    jac: Union[Callable, FiniteDifferenceMethods, bool, None] = dataclasses.field(
        default=None
    )
    hess: Union[
        Callable, FiniteDifferenceMethods, optimize.HessianUpdateStrategy, None
    ] = dataclasses.field(default=None)

    hessp: Optional[Callable] = dataclasses.field(default=None)
    callback: Optional[Callable] = dataclasses.field(default=None)
    constraints: Optional[optimize.NonlinearConstraint] = dataclasses.field(
        default=None
    )
    method: Optional[Method] = dataclasses.field(default=None)
    tol: Optional[float] = dataclasses.field(default=None)
    options: Optional[Dict[str, Any]] = dataclasses.field(default=None)


class OptimizationRecorder:
    def __init__(self):
        self._fun = []
        self._nit = []

    def update(self, intermediate_result):
        self._fun.append(intermediate_result.fun)
        self._nit.append(intermediate_result.nit)

    @property
    def fun(self):
        return np.asarray(self._fun)

    @property
    def nit(self):
        return np.asarray(self._nit)


class OptimPlotter:
    def __init__(self, keys, profile_plot=False, specs=None):
        if specs is not None:
            self._specs = specs
            spec_keys = {it for it in specs.keys()}
            if spec_keys.difference(keys):
                raise ValueError("")
        else:
            self._specs = {}
        self.setup_plots(keys)

        self._retval = 0

        self._fig.canvas.mpl_connect("close_event", lambda _: self.set_retval(1))

        self._profile_plot = profile_plot
        if profile_plot:
            self.setup_profile_plot()

        self._fig.tight_layout()

    def setup_plots(self, keys):
        n_plots = len(keys)
        self._fig, self._ax = plt.subplots(n_plots, figsize=(8, 3 * n_plots))
        self._data = {}
        self._keys = keys
        try:
            iter(self._ax)
        except TypeError:
            self._ax = [self._ax]
        for ax, k in zip(self._ax, self._keys):
            try:
                ax.set_ylabel(self._specs[k]["ylabel"])
            except KeyError:
                ax.set_ylabel(k)
            ax.set_xlabel("Iterations")
            (ln,) = ax.plot([], [])
            self._data[k] = {"x": [], "y": [], "line": ln}

    def setup_profile_plot(self):
        self._p_fig, self._p_ax = plt.subplots()
        self._p_ax.set_xlabel("Iterations")
        self._p_ax.set_ylabel("Time (s) per iteration")
        self._p_fig.canvas.mpl_connect("close_event", lambda _: self.set_retval(1))
        self._p_data = {"x": [], "y": []}
        (self._p_line,) = self._p_ax.plot([], [])
        self._tic = -1.0
        self._p_fig.tight_layout()

    def update(self, _, info):
        if self._profile_plot:
            self.update_tictoc(info)
            self._p_ax.relim()
            self._p_ax.autoscale_view(True, True)
            self._p_fig.canvas.draw_idle()

        for ax, k in zip(self._ax, self._keys):
            self.set_data(k, info["nit"], info[k])
            ax.relim()
            ax.autoscale_view(True, True)
        self._fig.canvas.draw_idle()
        plt.pause(0.01)
        return self._retval

    def set_data(self, k, x, y):
        self._data[k]["x"].append(x)
        self._data[k]["y"].append(y)
        self._data[k]["line"].set_data(self._data[k]["x"], self._data[k]["y"])

    def set_retval(self, val):
        self._retval = val

    def update_tictoc(self, info):
        toc = time.time()
        if self._tic < 0:
            self._tic = toc
            return
        else:
            dt = toc - self._tic
        self._tic = toc

        self._p_data["x"].append(info["nit"])
        self._p_data["y"].append(dt)
        self._p_line.set_data(self._p_data["x"], self._p_data["y"])

    def save_plots(self, prefix):
        prefix = pathlib.Path(prefix)
        self._fig.savefig(prefix / "optimization_results.png")
        self._p_fig.savefig(prefix / "optimization_performance.png")
