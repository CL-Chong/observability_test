import casadi as cs
import numpy as np


class SimpleEKF:
    def __init__(self, mdl, x0, kf_cov_0, t0=0, in_cov=None, obs_cov=None):
        self._nx = mdl.nx
        self._nu = mdl.nu
        self._ny = mdl.ny
        self.build_implementation(mdl)

        x0 = np.asarray(x0)
        if x0.shape != (self._nx,):
            raise ValueError("Invalid shape for initial state")
        self._x = x0

        self._t = t0

        kf_cov_0 = np.asarray(kf_cov_0)
        if kf_cov_0.shape != (self._nx, self._nx):
            raise ValueError("Invalid shape for initial error covariance")
        self._kf_cov = kf_cov_0

        if in_cov is None:
            self._in_cov = np.eye(self._nx)
        else:
            in_cov = np.asarray(in_cov)
            if in_cov.shape != (self._nu, self._nu):
                raise ValueError("Invalid shape for input covariance")
            self._in_cov = in_cov

        if obs_cov is None:
            self._obs_cov = np.eye(self._ny)
        else:
            obs_cov = np.asarray(obs_cov)
            if obs_cov.shape != (self._ny, self._ny):
                raise ValueError("Invalid shape for observation covariance")
            self._obs_cov = obs_cov

    def build_implementation(self, mdl):
        self._sym = {}

        def sym(*syms):
            return [self._sym[it] for it in syms]

        self._sym["x"] = cs.MX.sym("x", self._nx)  # type: ignore
        self._sym["u"] = cs.MX.sym("u", self._nu)  # type: ignore
        self._sym["dt"] = cs.MX.sym("dt")  # type: ignore
        self._sym["fcn"] = self._sym["x"] + self._sym["dt"] * mdl.dynamics(
            self._sym["x"], self._sym["u"]
        )
        self._sym["fjac"] = cs.jacobian(self._sym["fcn"], self._sym["x"])
        self._sym["gjac"] = cs.jacobian(self._sym["fcn"], self._sym["u"])

        self._fcn = cs.Function("fcn", sym("x", "u", "dt"), sym("fcn"))
        self._fjac = cs.Function("fjac", sym("x", "u", "dt"), sym("fjac"))
        self._gjac = cs.Function("gjac", sym("x", "u", "dt"), sym("gjac"))

        self._sym["p"] = cs.MX.sym("p", 2)  # type: ignore
        self._sym["hfcn"] = mdl.observation(self._sym["x"], self._sym["p"])
        self._sym["hjac"] = cs.jacobian(self._sym["hfcn"], self._sym["x"])

        self._hfcn = cs.Function("hfcn", sym("x", "p"), sym("hfcn"))
        self._hjac = cs.Function("hjac", sym("x", "p"), sym("hjac"))

    @property
    def x_op(self):
        return self._x

    @x_op.setter
    def x_op(self, val):
        val = np.asarray(val, dtype=np.double)
        if val.shape != (self._nx,):
            raise ValueError("Invalid shape for state")
        self._x[:] = val

    @property
    def t(self):
        return self._t

    @t.setter
    def t(self, val):
        self._t = float(val)

    @property
    def in_cov(self):
        return self._in_cov

    @property
    def obs_cov(self):
        return self._obs_cov

    @property
    def kf_cov(self):
        return self._kf_cov

    @kf_cov.setter
    def kf_cov(self, val):
        val = np.asarray(val, dtype=np.double)
        if val.shape != (self._nx, self.nx):
            raise ValueError("Invalid shape for error covariance")
        self._kf_cov[:] = val

    @property
    def nx(self):
        return self._nx

    @property
    def nu(self):
        return self._nu

    @property
    def ny(self):
        return self._ny

    def fcn(self, x, u, dt):
        x, u = np.asarray(x, dtype=np.double), np.asarray(u, dtype=np.double)
        return np.asarray(self._fcn(x, u, dt)).squeeze()

    def fjac(self, x, u, dt):
        x, u = np.asarray(x, dtype=np.double), np.asarray(u, dtype=np.double)
        return np.asarray(self._fjac(x, u, dt)).squeeze()

    def gjac(self, x, u, dt):
        x, u = np.asarray(x, dtype=np.double), np.asarray(u, dtype=np.double)
        return np.asarray(self._gjac(x, u, dt)).squeeze()

    def hfcn(self, x, p):
        x, p = np.asarray(x, dtype=np.double), np.asarray(p, dtype=np.double)
        return np.asarray(self._hfcn(x, p)).squeeze()

    def hjac(self, x, p):
        x, p = np.asarray(x, dtype=np.double), np.asarray(p, dtype=np.double)
        return np.asarray(self._hjac(x, p)).squeeze()

    def predict(self, u, dt):
        fjac = self.fjac(self.x_op, u, dt)
        gjac = self.gjac(self.x_op, u, dt)
        self._kf_cov = np.linalg.multi_dot(
            (fjac, self.kf_cov, fjac.T)
        ) + np.linalg.multi_dot((gjac, self.in_cov, gjac.T))
        self.x_op = self.fcn(self.x_op, u, dt)
        self._t += dt
        return self.x_op, self._kf_cov

    def update(self, y, p):
        hjac = self.hjac(self.x_op, p)
        resid_cov = np.linalg.multi_dot((hjac, self.kf_cov, hjac.T)) + self.obs_cov
        kf_gain = np.linalg.solve(resid_cov.T, hjac @ self.kf_cov).T
        self._kf_cov -= (
            np.linalg.multi_dot((kf_gain, hjac, self.kf_cov))
            + np.linalg.multi_dot((self.kf_cov, hjac.T, kf_gain.T))
            - np.linalg.multi_dot((kf_gain, resid_cov, kf_gain.T))
        )

        z = y - self.hfcn(self.x_op, p)
        self.x_op += kf_gain @ z
        return self.x_op, self._kf_cov, z

    def estimate(self, inputs, dt, params, meas):
        x_op_save = np.array(self.x_op)
        kf_cov_save = np.array(self.kf_cov)
        t_save = self.t
        x_hist = []
        cov_hist = []
        t_hist = []
        for idx, dt_k in enumerate(dt):
            u, p = inputs[:, idx], params[:, idx]
            x, _ = self.predict(u, dt_k)
            if callable(meas):
                state = {"idx": idx, "x": self.x_op, "u": u, "p": p}
                y = meas(state)
            else:
                y = meas[:, idx]
            x, kf_cov, _ = self.update(y, p)
            x_hist.append(np.array(x))
            cov_hist.append(np.array(kf_cov))
            t_hist.append(self.t)
        x_hist = np.asarray(x_hist).T
        cov_hist = np.asarray(cov_hist)

        self.x_op = x_op_save
        self.kf_cov = kf_cov_save
        self.t = t_save
        return x_hist, cov_hist, t_hist
