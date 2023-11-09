import numpy as np
import enum
import joblib


class NumericalDiff:
    def __init__(self, fcn, eps, n_jobs=6):
        self._fcn = fcn
        self._eps = np.sqrt(np.maximum(eps, np.finfo(np.double).eps))
        self._par = joblib.Parallel(n_jobs)

    def df(self, x):
        x = np.asarray(x)

        h = self._eps * np.abs(x)
        h[h < 1e-8] = self._eps

        jac = self._par(_df_ith(self._fcn, x, idx, it) for idx, it in enumerate(h))
        if jac is None:
            raise RuntimeError("Jacobian computation failed!")
        if np.isscalar(jac[0]):
            return np.asarray(jac)
        return np.column_stack(jac)


@joblib.delayed
def _df_ith(fcn, x, i, h):
    x_op = np.array(x)
    x_op[i] += h
    val2 = fcn(x_op)
    x_op[i] -= h
    val1 = fcn(x_op)
    return val2 - val1 / (2.0 * h)
