import dataclasses
from typing import Any, Callable, Dict, Literal, Optional, Union

from numpy.typing import ArrayLike
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
    Literal["trust-const"],
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
