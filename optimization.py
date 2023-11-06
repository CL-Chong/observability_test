import importlib
import pathlib

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize

from models.autodiff import models
from observability_aware_control.algorithms.autodiff import numlog, numsolve_sigma

eps = 1e-3
n_steps = 100
dt = jnp.ones(n_steps) * 0.1
rng = np.random.default_rng(seed=114514)
mdl = models.MultiRobot(n_robots=3)


pattern_angle = jnp.linspace(0, 2 * jnp.pi, mdl.n_robots + 1)[:-1][None]
x0 = jnp.vstack(
    [
        jnp.cos(pattern_angle) / jnp.sqrt(3),
        jnp.sin(pattern_angle) / jnp.sqrt(3),
        jnp.zeros((1, 3)),
    ]
)
x0 = jnp.ravel(x0, order="F")
u_full = jnp.vstack(
    [
        jnp.ones((1, n_steps)),
        jnp.zeros((1, n_steps)),
        1.0 + rng.normal(0.0, 0.1, (1, n_steps)),
        jnp.zeros((1, n_steps)),
        1.0 + rng.normal(0.0, 0.1, (1, n_steps)),
        jnp.zeros((1, n_steps)),
    ]
)
u_control = u_full[:2, :]
u_active = u_full[2:, :]


def numlog_objective(u):
    u = jnp.vstack([u_control, u.reshape(-1, n_steps, order="F")])
    W_o = numlog(mdl, x0, u, dt, eps, [3, 4, 6, 7])
    return jnp.linalg.cond(W_o)


x, _ = numsolve_sigma(mdl, x0, u_full, dt)

x_dst = x[:, -1].reshape(mdl.n_robots, -1)[:, 0:2]


def leader_tracking_constr(u):
    u = jnp.vstack([u_control, u.reshape(-1, n_steps, order="F")])
    x_actual, _ = numsolve_sigma(mdl, x0, u, dt)
    x_actual = jnp.reshape(x_actual, (-1, mdl.n_robots, x_actual.shape[-1]), order="F")[
        0:2, :, :
    ]

    return jnp.linalg.norm(
        x_actual[:, 1:, :] - x_actual[:, 0, :][:, None, :], axis=0
    ).ravel()


def destination_constraint(u):
    u = jnp.vstack([u_control, u.reshape(-1, n_steps, order="F")])
    x_actual, _ = numsolve_sigma(mdl, x0, u, dt)
    x_actual = x_actual[:, -1].reshape(mdl.n_robots, -1)[:, 0:2]

    dx = x_actual[1:, :] - x_dst[1:, :]

    return dx.ravel()


print("JIT compiled constraint, continuing")

cfg = importlib.import_module("config.optimcfg")

problem = {
    "fun": numlog_objective,
    "x0": u_active.ravel(order="F"),
    "jac": jax.jacobian(numlog_objective),
    "hess": jax.hessian(numlog_objective),
}


problem.update(
    {
        "bounds": optimize.Bounds(
            np.tile(np.asarray(cfg.LB), len(problem["x0"]) // len(cfg.LB)),
            np.tile(np.asarray(cfg.UB), len(problem["x0"]) // len(cfg.UB)),
        ),
        "method": cfg.METHOD,
        "options": cfg.OPTIONS,
        "constraints": optimize.NonlinearConstraint(
            leader_tracking_constr,
            lb=jnp.ones(2 * n_steps) * 0.5,
            ub=jnp.ones(2 * n_steps) * 1.5,
            jac=jax.jacobian(leader_tracking_constr),
        ),
    }
)

print("Running optimization problem")

soln = optimize.minimize(**problem)

soln_save = {k: v for k, v in soln.items() if isinstance(v, (jnp.ndarray, int, float))}
(pathlib.Path.cwd() / "data").mkdir(exist_ok=True)
jnp.savez("data/optimization_results.npz", **soln_save)


u_opt = jnp.reshape(soln.x, (-1, n_steps), order="F")

_, ax = plt.subplots()
for u_it, style in zip([u_opt, u_active], [":", "--"]):
    u = jnp.vstack([u_control, u_it])
    x, y = numsolve_sigma(mdl, x0, u, dt)

    x = jnp.reshape(x, (models.Robot.NX, mdl.n_robots, -1), order="F")
    for i_robot in range(mdl.n_robots):
        ax.plot(x[0, i_robot, :].ravel(), x[1, i_robot, :].ravel(), style)

plt.show()
