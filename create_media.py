import argparse
import pathlib

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation

N_POS_COMPS = 3


def minmax(v, *args):
    return np.array([np.min(v, *args), np.max(v, *args)])


def parse_cli():
    parser = argparse.ArgumentParser("create_media")
    parser.add_argument("file", type=pathlib.Path, help="Path to datafile")
    parser.add_argument("-s", nargs="+", default=None, help="Slice of the datafile")
    args = parser.parse_args()
    data = np.load(args.file)
    if args.s is not None:
        data = {k: v[..., slice(*map(int, args.s))] for k, v in data.items()}

    return data


def plot_cost(t, fun_hist):
    fig, ax = plt.subplots()
    # print(fun_hist)

    fun_mean = fun_hist.mean(axis=0)
    fun_stddev = fun_hist.std(axis=0)
    # ax.plot(fun_mean, label="Mean of objective value\n(over all simulations)")
    # ax.plot(fun_mean + fun_stddev)
    ax.fill_between(
        np.r_[0 : len(fun_mean)],
        fun_mean + fun_stddev,
        fun_mean - fun_stddev,
        alpha=0.1,
    )
    ax.boxplot(
        fun_hist[:, 0::10],
        positions=np.r_[0 : len(fun_mean) : 10],
        patch_artist=True,
        medianprops={"color": "white", "linewidth": 0.5},
        boxprops={"linewidth": 0, "facecolor": "C1", "alpha": 0.5},
        flierprops={
            "markersize": 3,
            "markerfacecolor": "C2",
            "markeredgecolor": "None",
            "alpha": 0.5,
        },
        widths=5,
    )
    # ax.set_xticks(np.arange(0, 151, 10))
    ax.set_xlabel("Solver Iterations")
    ax.set_ylabel("Objective Value")
    ax.set_title(
        "Statistical trends of change in objective value over solver iterations"
    )
    ax.legend()
    # for it in fun_hist:
    #     ax.plot(it, alpha=0.2)


def plot_3d_trajectory(t, states):
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    n_robots = states.shape[1]
    for j in range(n_robots):
        ax.plot(
            states[:, j, 0],
            states[:, j, 1],
            states[:, j, 2],
            label=f"Traj. of {f'Follower {j}' if j > 0 else 'Leader'}",
        )
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax = plt.gca()
    ax.set_zlabel("Z (m)")
    ax.set_zlim([5, 12])
    ax.legend()
    fig.savefig(
        "/home/hs293go/Documents/tex/hsgo-papers/papers/observability_aware_control/graphics/3d_lemniscate_o1_trajectories.png"
    )

    fig, ax = plt.subplots()
    for j in range(n_robots):
        ax.plot(
            states[:, j, 0],
            states[:, j, 1],
            label=f"Traj. of {f'Follower {j}' if j > 0 else 'Leader'}",
        )
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.legend()
    fig.savefig(
        "/home/hs293go/Documents/tex/hsgo-papers/papers/observability_aware_control/graphics/2d_lemniscate_o1_trajectories.png"
    )

    fig, ax = plt.subplots()
    for j in range(n_robots):
        ax.plot(
            t,
            states[:, j, 2],
            label=f"Traj. of {f'Follower {j}' if j > 0 else 'Leader'}",
        )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Z (m)")
    ax.legend()
    fig.savefig(
        "/home/hs293go/Documents/tex/hsgo-papers/papers/observability_aware_control/graphics/z_lemniscate_o1_trajectories.png"
    )


def animate_3d_trajectory(states, xlim, ylim):
    fig, ax = plt.subplots(
        # subplot_kw={"projection": "3d"},
    )
    n_robots = states.shape[1]

    ln = []
    for _ in range(n_robots):
        ln.append(ax.plot([], [])[0])
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    # zlims = minmax(states[:, :, 2])
    # zm = zlims.mean()
    # zd = zlims[-1] - zm
    # ax.set_zlim([zm - 3 * zd, zm + 3 * zd])
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    # ax.set_zlabel("Z (m)")

    def update(frame):
        for j in range(n_robots):
            comps, z = map(
                np.squeeze, np.split(states[:frame, j, :], N_POS_COMPS, axis=-1)
            )
            ln[j].set_data(*comps)
            ln[j].set_3d_properties(z)

        return ln

    anim = animation.FuncAnimation(fig, update, frames=states.shape[0], interval=50)
    anim.save("observability_predictive_control_ground_circ.mp4")
    return anim


def stats(arr, key):
    mean = np.mean(arr)
    std = np.std(arr)
    min_ = np.min(arr)
    if isinstance(min_, float):
        min_ = f"{min_:.3g}"

    max_ = np.max(arr)
    if isinstance(max_, float):
        max_ = f"{max_:.3g}"
    print(rf"{key} & {min_} & {max_} & {mean:.3g} \(\pm\) {std:.3g} \\ \midrule")


def main():
    data = parse_cli()

    states = data["states"]
    states = states.reshape(-1, 3, 10)

    t = data["time"]
    fun_hist = data["fun_hist"]
    fun_min = np.min(fun_hist, axis=0)
    fun_max = np.max(fun_hist, axis=0)
    fun_delta = np.abs(fun_hist[:, -1] - fun_hist[:, 0])

    stats(fun_min, r"Min. Objective Value")
    stats(fun_max, r"Max. Objective Value")
    stats(fun_delta, r"$\Delta$ Objective Value")
    stats(data["nit"], r"\# Solver Iterations")
    stats(data["execution_time"], "Execution Time (s)")
    stats(data["optimality"], "Optimality")
    # stats(data["constr_violation"], "Constraint Violation")
    plot_cost(t, fun_hist)
    plot_3d_trajectory(t, states)
    # _ = animate_3d_trajectory(states, xlim, ylim)
    plt.show()


if __name__ == "__main__":
    main()
