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
    fun_mean = fun_hist.mean(axis=0)
    fun_stddev = fun_hist.std(axis=0)
    ax.plot(fun_mean)
    ax.plot(fun_mean + fun_stddev)
    # ax.fill_between(
    # np.r_[0:20], fun_mean + fun_stddev, fun_mean - fun_stddev, alpha=0.2
    # )
    for it in fun_hist:
        ax.plot(it, alpha=0.2)


def plot_3d_trajectory(states):
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    n_robots = states.shape[1]
    for j in range(n_robots):
        comps = np.split(states[:, j, :], N_POS_COMPS, axis=0)
        ax.plot(*comps)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    # ax.set_zlabel("Z (m)")
    ax.set_zlim([8, 12])
    return xlim, ylim


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


def main():
    data = parse_cli()

    states = data["states"]
    states = states.reshape(-1, 3, 10)[..., 0:N_POS_COMPS]

    t = data["time"]
    fun_hist = data["fun_hist"]
    plot_cost(t, fun_hist)
    xlim, ylim = plot_3d_trajectory(states)
    # _ = animate_3d_trajectory(states, xlim, ylim)
    plt.show()


if __name__ == "__main__":
    main()
