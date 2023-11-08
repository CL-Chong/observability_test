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

N_ROBOTS = 3


pattern_angle = jnp.linspace(0, 2 * jnp.pi, N_ROBOTS + 1)[:-1][None]
x0_basic = jnp.vstack(
    [
        jnp.cos(pattern_angle) / jnp.sqrt(3),
        jnp.sin(pattern_angle) / jnp.sqrt(3),
        jnp.zeros((1, 3)),
    ]
)

x0_leader = x0_basic[:, 0]
x0_follower = x0_basic[:, 1:].ravel(order="F")

u_leader = jnp.tile(jnp.array([1, 0])[..., None], (1, n_steps))
u_follower = jnp.vstack(
    [
        1.0 + rng.normal(0.0, 0.01, (1, n_steps)),
        jnp.zeros((1, n_steps)),
        1.0 + rng.normal(0.0, 0.01, (1, n_steps)),
        jnp.zeros((1, n_steps)),
    ]
)

mdl = models.ReferenceSensingRobots(n_robots=2)
leader_x = numsolve_sigma(
    models.Robot(), x0_leader, u_leader, dt, without_observation=True
)[0:2, :]


def numlog_objective(u):
    u = u.reshape(mdl.nu, n_steps, order="F")
    W_o = numlog(
        mdl, x0_follower, u, dt, eps, perturb_axis=[0, 1, 3, 4], h_args=leader_x
    )
    return jnp.linalg.cond(W_o)


def leader_tracking_constr(u):
    u = u.reshape(mdl.nu, n_steps, order="F")
    x_actual = numsolve_sigma(mdl, x0_follower, u, dt, without_observation=True)
    x_actual = x_actual.reshape(-1, mdl.n_robots, x_actual.shape[-1], order="F")[
        0:2, :, :
    ]

    dx = x_actual - leader_x[:, None, :]
    return jnp.linalg.norm(dx, axis=0).ravel()


x = numsolve_sigma(mdl, x0_follower, u_follower, dt, without_observation=True)

x_dst = x[:, -1].reshape(mdl.n_robots, -1)[:, 0:2]


def destination_constraint(u):
    u = u.reshape(mdl.nu, n_steps, order="F")
    x_actual = numsolve_sigma(mdl, x0_follower, u, dt, without_observation=True)
    x_actual = x_actual[:, -1].reshape(mdl.n_robots, -1)[:, 0:2]

    return (x_actual - x_dst).ravel()


print("JIT compiled constraint, continuing")

cfg = importlib.import_module("config.optimcfg")

problem = {
    "fun": numlog_objective,
    "x0": u_follower.ravel(order="F"),
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
            destination_constraint,
            lb=0,
            ub=0,
            jac=jax.jacobian(destination_constraint),
        ),
    }
)

soln = optimize.minimize(**problem)

soln_save = {k: v for k, v in soln.items() if isinstance(v, (jnp.ndarray, int, float))}
(pathlib.Path.cwd() / "data").mkdir(exist_ok=True)
jnp.savez("data/optimization_results.npz", **soln_save)


u_opt = jnp.reshape(soln.x, (-1, n_steps), order="F")

_, ax = plt.subplots()
for u, style in zip([u_opt, u_follower], [":", "--"]):
    x, y = numsolve_sigma(mdl, x0_follower, u, dt, h_args=leader_x)

    x = jnp.reshape(x, (models.Robot.NX, mdl.n_robots, -1), order="F")
    for i_robot in range(mdl.n_robots):
        ax.plot(x[0, i_robot, :].ravel(), x[1, i_robot, :].ravel(), style)

plt.show()
