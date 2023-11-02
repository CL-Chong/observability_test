import numpy as np
import casadi as cs
import math
import pathlib
import pandas as pd
import joblib
from tqdm import tqdm

# from observability_test_small import compute_gramian
from three_drones import stlog_symbolic, numlog
from models import models


def main():
    dt = 0.001
    eps = 1e-4
    n_steps_arr = [250, 500, 750, 1000]
    num_sys = models.MultiRobot(n_robots=3, is_symbolic=False)
    sym_sys = models.MultiRobot(n_robots=3, is_symbolic=True)
    n_samples = 1000
    max_order = 6

    @joblib.delayed
    def _case_check(i, n_steps, order_stlog):
        x0, u_control_const, u_drone_const = get_ic(i, sym_sys)

        u_control = u_control_const.reshape((sym_sys.NU, 1)) * np.ones(
            (sym_sys.NU, n_steps)
        )  # broadcasting column vector
        u_drone = u_drone_const.reshape((sym_sys.nu - sym_sys.NU, 1)) * np.ones(
            (sym_sys.nu - sym_sys.NU, n_steps)
        )  # broadcasting column vector

        diagnostic_log = [i, dt * n_steps] + log_comparison(
            dt,
            eps,
            n_steps,
            x0,
            u_control_const,
            u_drone_const,
            u_control,
            u_drone,
            order_stlog,
            num_sys,
            sym_sys,
        )
        return diagnostic_log

    def get_ic(i, sym_sys):
        rng = np.random.default_rng(i**2)
        x0 = rng.uniform(-1.0, 1.0, (sym_sys.nx,))
        u_control_const = rng.uniform(-1.0, 1.0, (sym_sys.NU,))
        u_drone_const = rng.uniform(-1.0, 1.0, (sym_sys.nu - sym_sys.NU,))
        return x0, u_control_const, u_drone_const

    par_evaluator = joblib.Parallel(4)
    data_arr = par_evaluator(
        _case_check(i, n_steps, order_stlog)
        for i in tqdm(range(0, n_samples))
        for n_steps in n_steps_arr
        for order_stlog in range(1, max_order + 1)
    )

    df = pd.DataFrame(
        data_arr,
        columns=[
            "case_id",
            "t_span",
            "order(stlog)",
            "tr(numlog)",
            "sing_max(numlog)",
            "sing_min(numlog)",
            "tr(stlog)",
            "sing_max(stlog)",
            "sing_min(stlog)",
            "mae(diff)",
            "rms(diff)",
        ],
    )
    df.to_csv(f"stlog_vs_numlog_data{n_samples}.csv", index=False)


def log_comparison(
    dt,
    eps,
    n_steps,
    x0,
    u_control_const,
    u_drone_const,
    u_control,
    u_drone,
    order_stlog,
    num_sys,
    sym_sys,
):
    numlog_x = numlog(num_sys, x0, u_control, u_drone, dt, n_steps, eps)
    tr_num = np.trace(numlog_x)
    sing_max_num = np.linalg.norm(numlog_x, 2)
    sing_min_num = np.linalg.norm(numlog_x, -2)

    stlog_fun = stlog_symbolic(sym_sys, order_stlog)
    stlog_x = stlog_fun(x0, u_control_const, u_drone_const, dt * n_steps)
    tr_st = np.trace(stlog_x)
    sing_max_st = np.linalg.norm(stlog_x, 2)
    sing_min_st = np.linalg.norm(stlog_x, -2)

    err_mat = numlog_x - stlog_x
    err_mae = np.linalg.norm(err_mat, 1) / (sym_sys.nx**2)
    err_rms = np.linalg.norm(err_mat, 2) / (sym_sys.nx)

    return [
        order_stlog,
        tr_num,
        sing_max_num,
        sing_min_num,
        tr_st,
        sing_max_st,
        sing_min_st,
        err_mae,
        err_rms,
    ]


# old ICs
# x0 = np.array([0.0, 0.0, 0.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0])
# [v0, v1, u0, u1, u2, u3] = [1.0, -0.5, 0.3, -0.4, 0.5, 0.6]
# u_control = np.array([v0 * np.ones(n_steps), v1 * np.ones(n_steps)])
# u_drone = np.array(
#    [
#        u0 * np.ones(n_steps),
#        u1 * np.ones(n_steps),
#        u2 * np.ones(n_steps),
#        u3 * np.ones(n_steps),
#    ]
# )
# Old I/O check:
# savefile = pathlib.Path("save_data.npz")
#  if savefile.exists() and savefile.is_file():
#      save_data = np.load(savefile)
#      ok = np.allclose(save_data["numlog_x"], numlog_x)
#      if not ok:
#          print("FAILURE")
#          print(save_data["numlog_x"], numlog_x)
#      ok = np.allclose(save_data["stlog_x"], stlog_x)
#      if not ok:
#          print("FAILURE")
#          print(save_data["stlog_x"], stlog_x)

#  else:
#      np.savez(savefile, stlog_x=stlog_x, numlog_x=numlog_x)

# def main_old():
#    dt = 0.01
#    eps = 1e-3
#    n_timesteps = 1000
#    x0 = np.array([1.0, 0.0, 0.0])
#    u_all = 1.0 * np.ones((2, n_timesteps))
#
#    gramian = compute_gramian(x0, u_all, dt, n_timesteps, eps)
#    condition_number = np.linalg.cond(gramian)
#    print(gramian)
#    print(condition_number)


if __name__ == "__main__":
    main()
