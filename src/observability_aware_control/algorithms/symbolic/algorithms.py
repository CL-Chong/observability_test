import math
import casadi


def stlog(sys, order, is_psd=False):
    x = casadi.MX.sym("x", sys.nx)
    u = casadi.MX.sym("u", sys.nu)
    T = casadi.MX.sym("T")
    stlog = casadi.MX.zeros(sys.nx, sys.nx)
    lh_store = []
    lh_store.append(sys.observation(x))
    dlh_store = []
    dlh_store.append(casadi.jacobian(lh_store[0], x))

    for l in range(0, order):
        lh_store.append(casadi.jtimes(lh_store[l], x, sys.dynamics(x, u)))
        dlh_store.append(casadi.jacobian(lh_store[l + 1], x))
    if is_psd:
        for a in range(0, order + 1):
            for b in range(0, order + 1):
                stlog += (
                    (T ** (a + b + 1))
                    / (math.factorial(a) * math.factorial(b) * (a + b + 1))
                ) * casadi.mtimes(
                    dlh_store[a].T,
                    dlh_store[b],
                )
    else:
        for l in range(0, order):
            for k in range(0, l + 1):
                stlog += (
                    (T ** (l + 1))
                    / ((l + 1) * math.factorial(k) * math.factorial(l - k))
                ) * casadi.mtimes(
                    dlh_store[k].T,
                    dlh_store[l - k],
                )

    stlog_fun = casadi.Function("stlog_fun", [x, u, T], [stlog])

    return stlog_fun
