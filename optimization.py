import argparse
import importlib
import pathlib

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize

from models.autodiff import models
from observability_aware_control.algorithms.autodiff import numlog, numsolve_sigma
from utils import utils
from utils.optim_plotter import OptimPlotter


def parse_cli():
    parser = argparse.ArgumentParser("optimization")
    parser.add_argument(
        "--eps", type=float, default=1e-3, help="Numlog finite-difference stepsize"
    )
    parser.add_argument("--n_steps", type=int, default=100, help="Number of timesteps")
    parser.add_argument("--dt", type=float, default=1e-1, help="Size of a timestep")
    parser.add_argument(
        "--maxiter",
        type=int,
        default=-1,
        help="Number of optimization iterations (Will override config file)",
    )
    parser.add_argument(
        "--use_global_optimizer",
        action="store_true",
        help="Toggles using global optimizer",
    )
    parser.add_argument(
        "--constraint_type",
        type=str,
        default="destination",
        help="Type of constraint (destination / leader_tracking)",
    )

    args = parser.parse_args()
    return args


def main():
    args = parse_cli()
    params = {}
    params["eps"] = args.eps
    params["n_steps"] = args.n_steps
    params["dt"] = jnp.ones(params["n_steps"]) * args.dt

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

    params["x0_leader"] = x0_basic[:, 0]
    params["x0_follower"] = x0_basic[:, 1:].ravel(order="F")

    params["u_leader"] = jnp.tile(jnp.array([1, 0])[..., None], (1, params["n_steps"]))
    params["u_follower"] = jnp.vstack(
        [
            1.0 + rng.normal(0.0, 0.01, (1, params["n_steps"])),
            jnp.zeros((1, params["n_steps"])),
            1.0 + rng.normal(0.0, 0.01, (1, params["n_steps"])),
            jnp.zeros((1, params["n_steps"])),
        ]
    )

    mdl = models.ReferenceSensingRobots(n_robots=2)
    leader_x = numsolve_sigma(
        models.Robot(),
        params["x0_leader"],
        params["u_leader"],
        params["dt"],
        without_observation=True,
    )[0:2, :]

    def numlog_objective(u):
        u = u.reshape(mdl.nu, params["n_steps"], order="F")
        W_o = numlog(
            mdl,
            params["x0_follower"],
            u,
            params["dt"],
            params["eps"],
            perturb_axis=[0, 1, 3, 4],
            h_args=leader_x,
        )
        return jnp.linalg.cond(W_o)

    def leader_tracking_constr(u):
        u = u.reshape(mdl.nu, params["n_steps"], order="F")
        x_actual = numsolve_sigma(
            mdl, params["x0_follower"], u, params["dt"], without_observation=True
        )
        x_actual = x_actual.reshape(-1, mdl.n_robots, x_actual.shape[-1], order="F")[
            0:2, :, :
        ]

        dx = x_actual - leader_x[:, None, :]
        return jnp.linalg.norm(dx, axis=0).ravel()

    traj_follower = numsolve_sigma(
        mdl,
        params["x0_follower"],
        params["u_follower"],
        params["dt"],
        without_observation=True,
    )

    x_dst = traj_follower[:, -1].reshape(mdl.n_robots, -1)[:, 0:2]

    def destination_constr(u):
        u = u.reshape(mdl.nu, params["n_steps"], order="F")
        x_actual = numsolve_sigma(
            mdl, params["x0_follower"], u, params["dt"], without_observation=True
        )
        x_actual = x_actual[:, -1].reshape(mdl.n_robots, -1)[:, 0:2]

        dx = x_actual - x_dst
        return jnp.sum(dx * dx, axis=0)

    cfg = importlib.import_module("config.optimcfg")

    if args.constraint_type == "destination":
        constr = destination_constr
    elif args.constraint_type == "leader_tracking":
        constr = leader_tracking_constr
    else:
        raise RuntimeError("invalid valid constraint type")

    p = OptimPlotter(["fun"], True, specs={"fun": {"ylabel": "Objective Value"}})
    problem = {
        "fun": numlog_objective,
        "x0": params["u_follower"].ravel(order="F"),
        "jac": jax.jacobian(numlog_objective),
        "hess": jax.hessian(numlog_objective),
        "callback": p.update,
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
                constr,
                lb=np.zeros(2),
                ub=np.ones(2) * 0.1,
                jac=jax.jacobian(constr),
            ),
        }
    )
    if args.maxiter > 0:
        problem["options"]["maxiter"] = args.maxiter

    if args.use_global_optimizer:
        soln = optimize.basinhopping(
            problem.pop("fun"), problem.pop("x0"), minimizer_kwargs=problem
        )
    else:
        soln = optimize.minimize(**problem)

    save = save_data(params, problem, soln)
    p.save_plots("data")
    jnp.savez("data/optimization_results.npz", **save)

    u_opt = jnp.reshape(soln.x, (-1, params["n_steps"]), order="F")
    traj_follower_opt = numsolve_sigma(
        mdl,
        params["x0_follower"],
        u_opt,
        params["dt"],
        h_args=leader_x,
        without_observation=True,
    )
    _, ax = plt.subplots()
    for traj, style in zip([traj_follower, traj_follower_opt], [":", "--"]):
        traj = jnp.reshape(traj, (models.Robot.NX, mdl.n_robots, -1), order="F")
        for i_robot in range(mdl.n_robots):
            ax.plot(traj[0, i_robot, :].ravel(), traj[1, i_robot, :].ravel(), style)

    plt.show()


def save_data(params, problem, soln):
    save = utils.flatten_dict(
        {
            "soln": utils.take_arrays(soln),
            "problem": utils.take_arrays(problem),
            "params": utils.take_arrays(params),
        }
    )
    (pathlib.Path.cwd() / "data").mkdir(exist_ok=True)
    return save


if __name__ == "__main__":
    main()
