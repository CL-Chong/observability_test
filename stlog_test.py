import math

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import tqdm
import joblib
from scipy import optimize
from scipy.optimize import NonlinearConstraint

import src.observability_aware_control.models.autodiff.multi_planar_robot as nummodels
from src.observability_aware_control.algorithms.autodiff.algorithms import (
    STLOG,
    STLOGMinimizeProblem,
    STLOGOptions,
)
from src.observability_aware_control.optimize import minimize
from src.observability_aware_control.utils import utils

# testing list: (X) = bad, (1/2) = not sure, (O) = good
# nonlinear constraints (X)
# log-scaling (1/2)
# det and tr replacements for min eig (X)
# waypoints (1/2 - doesn't break, but doesn't achieve much)
# adaptive dt_stlog (O - continue to refine)
# psd mod (O - less transients than base, but less turning too)

jax.config.update("jax_enable_x64", True)


def test(anim=False):
    order_psd = 3

    num_mdl = nummodels.LeaderFollowerRobots(3)
    win_sz_arr = np.arange(5, 1000, dtype=int)
    win_fmin = 1e-5
    stlog = STLOG(num_mdl, order_psd)
    dt = 0.05
    dt_stlog = 0.2
    n_steps = 4000
 
    x0 = np.array(
        [0.5, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, -5.0, 0.0],
    )
    rot_magnitude = 2
    min_thurst = 0.0
    max_thrust = 4.0
    u_leader = np.array([1.0, 0.0])
    u0 = np.concatenate((u_leader, [4.0, -1.0, 4.0, 1.0]))
    u_lb = np.concatenate(
        (u_leader, [min_thurst, -rot_magnitude, min_thurst, -rot_magnitude])
    )
    u_ub = np.concatenate(
        (u_leader, [max_thrust, rot_magnitude, max_thrust, rot_magnitude])
    )
    # some failed results
    failed_dict = np.load("failed_optimization_results.npz")
    pre_mortem_index = 2850
    # fatal_index = 2630
    x0 = failed_dict["states"][:, pre_mortem_index]
    u0 = failed_dict["inputs"][:, pre_mortem_index]

    x = np.zeros((num_mdl.nx, n_steps))
    x[:, 0] = x0

    u = np.zeros((num_mdl.nu, n_steps))
    u[0 : num_mdl.NU, :] = u_leader[:, None]
    u[:, 0] = u0

    if anim:
        anim, anim_ax = plt.subplots()
        plt.ioff()

        anim_data = {
            idx: {"line": anim_ax.plot([], [])[0]} for idx in range(num_mdl.n_robots)
        }

    # optim_hist = {}
    # def con(a):
    #     return np.array([a[1] / a[0], a[3] / a[2]])

    # nlc = NonlinearConstraint(con, [-1e4, -1e4], [1e4, 1e4])

    # u_leader = np.tile(u_leader[..., None], [1, win_sz])
    @joblib.delayed
    def _make_problem(win_sz):
        return STLOGMinimizeProblem(stlog, STLOGOptions(dt=dt_stlog, window=win_sz))
    par_evaluator = joblib.Parallel(12)
    min_problem_arr = par_evaluator(_make_problem(win_sz) for win_sz in win_sz_arr)
    skip_until = 0
    print("Compilation of STLOGMinimizeProblem complete.")
    for i in tqdm.tqdm(range(1, n_steps)):
        if i < skip_until:
            continue
        j_tmp = 0
        while j_tmp < len(win_sz_arr):
            problem = min_problem_arr[j_tmp].make_problem(
                x[:, i - 1],
                jnp.broadcast_to(u[:, i - 1], (win_sz_arr[j_tmp], len(u[:, i - 1]))),
                dt,
                u_lb,
                u_ub,
                id_const=(0, 1),
            )

            soln = minimize(problem)
            if abs(soln.fun) > win_fmin or j_tmp == len(win_sz_arr) - 1: # accept solution
                soln_u = soln.x.T
                sentinel_idx = min(i + win_sz_arr[j_tmp], n_steps)
                win_idx = slice(i, sentinel_idx)
                u[:, win_idx] = soln_u[:, : win_sz_arr[j_tmp] + min(n_steps - i - win_sz_arr[j_tmp], 0)]
                x[:, win_idx] = min_problem_arr[j_tmp].forward_dynamics(x[:, i - 1], soln.x, dt).T[:, : win_sz_arr[j_tmp] + min(n_steps - i - win_sz_arr[j_tmp], 0)]

                x[::-3, win_idx] = utils.wrap_to_pi(x[::-3, win_idx])

                if anim:
                    x_drawable = np.reshape(
                        x[:, 0:i], (num_mdl.NX, num_mdl.n_robots, i), order="F"
                    )
                    for j in range(num_mdl.n_robots):
                        anim_data[j]["x"] = x_drawable[0, j, :]
                        anim_data[j]["y"] = x_drawable[1, j, :]
                        anim_data[j]["line"].set_data(anim_data[j]["x"], anim_data[j]["y"])
                    anim_ax.relim()
                    anim_ax.autoscale_view(True, True)
                    anim.canvas.draw_idle()
                    plt.pause(0.01)
                
                skip_until = i + win_sz_arr[j_tmp]
                if j_tmp == len(win_sz_arr) - 1:
                    print(f"Maximum window size {win_sz_arr[j_tmp]} reached with objective value {soln.fun}. Continuing.")

                break
            else:
                print(f"Reject solution with window size {win_sz_arr[j_tmp]} and objective value {soln.fun}.")
                j_tmp += 1 # reject solution. try next window size
                


                

    # optim_hist = {k: np.asarray(v) for k, v in optim_hist.items()}
    # np.savez("data/optimization_results.npz", states=x, inputs=u, **optim_hist)

    def plotting_simple(model, x):
        figs = {}
        figs[0], ax = plt.subplots()
        ax.plot(x[0, :], x[1, :], "C9")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        for idx in range(1, model.n_robots):  # Vary vehicles
            ax.plot(x[0 + idx * model.NX, :], x[1 + idx * model.NX, :], f"C{idx}")

        figs[0].savefig("data/stlog_planning.png")
        # figs[0].savefig("stlog_planning_for_go.png")

        # plt.show()

    plotting_simple(num_mdl, x)

    return


print(test(anim=True))
