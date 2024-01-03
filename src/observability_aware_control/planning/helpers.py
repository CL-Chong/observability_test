"""
Copyright © 2023 FSC Lab

Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE
OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import numpy as np

from .trajectories import MultirotorTrajectory, forward


def loop_trajectory(
    discretization_dt,
    radius,
    z,
    v_max,
    lin_acc,
    clockwise=False,
    yawing=False,
    vehicle_mass=1.0,
):
    """
    Creates a circular trajectory on the x-y plane that increases speed by 1m/s at every revolution.

    :param params: vehicle model
    :param discretization_dt: Sampling period of the trajectory.
    :param radius: radius of loop trajectory in meters
    :param z: z position of loop plane in meters
    :param lin_acc: linear acceleration of trajectory (and successive deceleration) in m/s^2
    :param clockwise: True if the rotation will be done clockwise.
    :param yawing: True if the vehicle yaws along the trajectory. False for 0 yaw trajectory.
    :param v_max: Maximum speed at peak velocity. Revolutions needed will be calculated automatically.
    :param plot: Whether to plot an analysis of the planned trajectory or not.
    :return: The full 13-DoF trajectory with time and input vectors
    """

    ramp_up_t = 2  # s

    # Calculate simulation time to achieve desired maximum velocity with specified
    # acceleration
    t_total = 2 * v_max / lin_acc + 2 * ramp_up_t

    # Transform to angular acceleration
    alpha_acc = lin_acc / radius  # rad/s^2

    # Generate time and angular acceleration sequences
    # Ramp up sequence
    refs = {}
    alphas = {}
    refs["ramp"] = np.arange(0, ramp_up_t, discretization_dt)
    alphas["ramp_up"] = alpha_acc * np.sin(np.pi / (2 * ramp_up_t) * refs["ramp"]) ** 2
    # Acceleration phase
    coasting_duration = (t_total - 4 * ramp_up_t) / 2
    refs["coasting"] = ramp_up_t + np.arange(0, coasting_duration, discretization_dt)
    alphas["coasting"] = np.ones_like(refs["coasting"]) * alpha_acc
    # Transition phase: decelerate
    refs["transition"] = np.arange(0, 2 * ramp_up_t, discretization_dt)
    alphas["transition"] = alpha_acc * np.cos(
        np.pi / (2 * ramp_up_t) * refs["transition"]
    )
    refs["transition"] += refs["coasting"][-1] + discretization_dt
    # Deceleration phase
    refs["downcoasting"] = (
        refs["transition"][-1]
        + np.arange(0, coasting_duration, discretization_dt)
        + discretization_dt
    )
    alphas["down_coasting"] = -np.ones_like(refs["downcoasting"]) * alpha_acc
    # Bring to rest phase
    refs["ramp_up"] = (
        refs["downcoasting"][-1]
        + np.arange(0, ramp_up_t, discretization_dt)
        + discretization_dt
    )
    alphas["ramp_up_end"] = alphas["ramp_up"] - alpha_acc

    # Concatenate all sequences
    t_ref = np.concatenate(
        (
            refs["ramp"],
            refs["coasting"],
            refs["transition"],
            refs["downcoasting"],
            refs["ramp_up"],
        )
    )
    alpha_vec = np.concatenate(
        (
            alphas["ramp_up"],
            alphas["coasting"],
            alphas["transition"],
            alphas["down_coasting"],
            alphas["ramp_up_end"],
        )
    )

    # Calculate derivative of angular acceleration (alpha_vec)
    ramp_up_alpha_dt = (
        alpha_acc * np.pi / (2 * ramp_up_t) * np.sin(np.pi / ramp_up_t * refs["ramp"])
    )
    coasting_alpha_dt = np.zeros_like(alphas["coasting"])
    transition_alpha_dt = (
        -alpha_acc
        * np.pi
        / (2 * ramp_up_t)
        * np.sin(np.pi / (2 * ramp_up_t) * refs["transition"])
    )
    alpha_dt = np.concatenate(
        (
            ramp_up_alpha_dt,
            coasting_alpha_dt,
            transition_alpha_dt,
            coasting_alpha_dt,
            ramp_up_alpha_dt,
        )
    )

    if not clockwise:
        alpha_vec *= -1
        alpha_dt *= -1

    # Compute angular integrals
    w_vec = np.cumsum(alpha_vec) * discretization_dt
    angle_vec = np.cumsum(w_vec) * discretization_dt

    # Compute position, velocity, acceleration, jerk
    pos_traj_x = radius * np.sin(angle_vec)
    pos_traj_y = radius * np.cos(angle_vec)
    pos_traj_z = np.ones_like(pos_traj_x) * z

    vel_traj_x = radius * w_vec * np.cos(angle_vec)
    vel_traj_y = -(radius * w_vec * np.sin(angle_vec))

    acc_traj_x = radius * (
        alpha_vec * np.cos(angle_vec) - w_vec**2 * np.sin(angle_vec)
    )
    acc_traj_y = -radius * (
        alpha_vec * np.sin(angle_vec) + w_vec**2 * np.cos(angle_vec)
    )

    jerk_traj_x = radius * (
        alpha_dt * np.cos(angle_vec)
        - alpha_vec * np.sin(angle_vec) * w_vec
        - np.cos(angle_vec) * w_vec**3
        - 2 * np.sin(angle_vec) * w_vec * alpha_vec
    )
    jerk_traj_y = -radius * (
        np.cos(angle_vec) * w_vec * alpha_vec
        + np.sin(angle_vec) * alpha_dt
        - np.sin(angle_vec) * w_vec**3
        + 2 * np.cos(angle_vec) * w_vec * alpha_vec
    )

    if yawing:
        yaw_traj = -angle_vec
    else:
        yaw_traj = np.zeros_like(angle_vec)

    traj = np.concatenate(
        (
            np.row_stack((pos_traj_x, pos_traj_y, pos_traj_z))[None, ...],
            np.row_stack((vel_traj_x, vel_traj_y, np.zeros_like(vel_traj_x)))[
                None, ...
            ],
            np.row_stack((acc_traj_x, acc_traj_y, np.zeros_like(acc_traj_x)))[
                None, ...
            ],
            np.row_stack((jerk_traj_x, jerk_traj_y, np.zeros_like(jerk_traj_x)))[
                None, ...
            ],
        ),
        0,
    )

    yaw = np.row_stack((yaw_traj, w_vec))
    len_traj = traj.shape[-1]
    traj_ref = np.zeros((10, len_traj))
    traj_ref[0:3, :] = np.squeeze(traj[0, :, :])
    traj_ref[3:7, :], u_ref = forward(traj[1:, :, :], yaw, vehicle_mass)
    traj_ref[7:10, :] = np.squeeze(traj[1, :, :])

    return MultirotorTrajectory(traj_ref, u_ref, t_ref)


def lemniscate_trajectory(
    discretization_dt,
    radius,
    z,
    lin_acc,
    v_max,
    vehicle_mass=1.0,
):
    """

    :param params:
    :param discretization_dt:
    :param radius:
    :param z:
    :param lin_acc:
    :param clockwise:
    :param yawing:
    :param v_max:
    :param map_name:
    :param plot:
    :return:
    """

    assert z > 0

    ramp_up_t = 2  # s

    # Calculate simulation time to achieve desired maximum velocity with specified
    # acceleration
    t_total = 2 * v_max / lin_acc + 2 * ramp_up_t

    # Transform to angular acceleration
    alpha_acc = lin_acc / radius  # rad/s^2

    # Generate time and angular acceleration sequences
    # Ramp up sequence
    refs = {}
    alphas = {}
    refs["ramp"] = np.arange(0, ramp_up_t, discretization_dt)
    alphas["ramp_up"] = alpha_acc * np.sin(np.pi / (2 * ramp_up_t) * refs["ramp"]) ** 2
    # Acceleration phase
    coasting_duration = (t_total - 4 * ramp_up_t) / 2
    refs["coasting"] = ramp_up_t + np.arange(0, coasting_duration, discretization_dt)
    alphas["coasting"] = np.ones_like(refs["coasting"]) * alpha_acc
    # Transition phase: decelerate
    refs["transition"] = np.arange(0, 2 * ramp_up_t, discretization_dt)
    alphas["transition"] = alpha_acc * np.cos(
        np.pi / (2 * ramp_up_t) * refs["transition"]
    )
    refs["transition"] += refs["coasting"][-1] + discretization_dt
    # Deceleration phase
    refs["downcoasting"] = (
        refs["transition"][-1]
        + np.arange(0, coasting_duration, discretization_dt)
        + discretization_dt
    )
    alphas["down_coasting"] = -np.ones_like(refs["downcoasting"]) * alpha_acc
    # Bring to rest phase
    refs["ramp_up"] = (
        refs["downcoasting"][-1]
        + np.arange(0, ramp_up_t, discretization_dt)
        + discretization_dt
    )
    alphas["ramp_up_end"] = alphas["ramp_up"] - alpha_acc

    # Concatenate all sequences
    t_ref = np.concatenate(tuple(refs.values()))
    alpha_vec = np.concatenate(tuple(alphas.values()))

    # Compute angular integrals
    w_vec = np.cumsum(alpha_vec) * discretization_dt
    angle_vec = np.cumsum(w_vec) * discretization_dt

    # Adaption: we achieve the highest spikes in the bodyrates when passing through the 'center' part of the figure-8
    # This leads to negative reference thrusts.
    # Let's see if we can alleviate this by adapting the z-reference in these parts to add some acceleration in the
    # z-component
    z_dim = 0.0

    # Compute position, velocity, acceleration, jerk
    traj = np.zeros((4, 3, np.size(angle_vec)))
    traj[0, 0, :] = radius * np.cos(angle_vec)
    traj[0, 1, :] = radius * (np.sin(angle_vec) * np.cos(angle_vec))
    traj[0, 2, :] = -z_dim * np.cos(4.0 * angle_vec) + z

    traj[1, 0, :] = -radius * (w_vec * np.sin(angle_vec))
    traj[1, 1, :] = radius * (
        w_vec * np.cos(angle_vec) ** 2 - w_vec * np.sin(angle_vec) ** 2
    )
    traj[1, 2, :] = 4.0 * z_dim * w_vec * np.sin(4.0 * angle_vec)

    traj[2, 0, :] = -radius * (
        alpha_vec * np.sin(angle_vec) + w_vec**2 * np.cos(angle_vec)
    )
    traj[2, 1, :] = radius * (
        alpha_vec * np.cos(angle_vec) ** 2
        - 2.0 * w_vec**2 * np.cos(angle_vec) * np.sin(angle_vec)
        - alpha_vec * np.sin(angle_vec) ** 2
        - 2.0 * w_vec**2 * np.sin(angle_vec) * np.cos(angle_vec)
    )
    traj[2, 2, :] = (
        16.0
        * z_dim
        * (w_vec**2 * np.cos(4.0 * angle_vec) + alpha_vec * np.sin(4.0 * angle_vec))
    )

    len_traj = traj.shape[-1]
    yaw_derivatives = np.zeros((2, len_traj), dtype=np.float64)
    traj_ref = np.zeros((10, len_traj))
    traj_ref[0:3, :] = np.squeeze(traj[0, :, :])
    traj_ref[3:7, :], u_ref = forward(traj[1:, :, :], yaw_derivatives, vehicle_mass)
    traj_ref[7:10, :] = np.squeeze(traj[1, :, :])

    return MultirotorTrajectory(traj_ref, u_ref, t_ref)
