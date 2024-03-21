import functools
from collections.abc import Callable, MutableMapping, Sequence
from typing import Any, Dict, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import minsnap_trajectories as ms
import numpy as np
from jax.typing import ArrayLike

from observability_aware_control.algorithms import common
from observability_aware_control.algorithms.misc import geometric_controller
from observability_aware_control.models import quadrotor

Archive = Dict[str, ArrayLike]


def complementary_indices(size: int, ids: ArrayLike) -> jax.Array:
    """Finds the complement of given indices in a index sequence

    Parameters
    ----------
    size : int
        Size of the index sequence
    ids : ArrayLike
        Index whose complement is to be taken

    Returns
    -------
    jax.Array
        Array containing the complement of ids
    """
    id_complement = jnp.setdiff1d(jnp.arange(0, size), ids)
    return id_complement


@jax.jit
def separate_array(
    mat: jax.Array, idx: ArrayLike, idy: Optional[ArrayLike] = None
) -> Tuple[ArrayLike, ArrayLike]:
    """Extracts the idx and idy-th components (along the last axis) of the input matrix

    Parameters
    ----------
    mat : jax.Array
        Input array to be separated
    idx : ArrayLike
        First components of the array to be separated out
    idy : Optional[ArrayLike], optional
        Second components of the array to be separated out, by default None, in which
        case said components will be the complement of idx

    Returns
    -------
    Tuple[ArrayLike, ArrayLike]
        The idx and idy-th components of the input matrix respectively
    """

    if idy is None:
        idy = complementary_indices(mat.shape[-1], idx)
    return mat[..., idx], mat[..., idy]


@functools.partial(jax.jit, static_argnums=[0])
def combine_array(
    shape: Union[int, Sequence[int]],
    m_1: ArrayLike,
    m_2: ArrayLike,
    idx: ArrayLike,
    idy: ArrayLike,
) -> jax.Array:
    """Builds an array by combining idx and idy-th components (along the last axis) with
    values m_1 and m_2. Inverse of separate_array

    Parameters
    ----------
    shape : Union[int, Sequence[int]]
        The shape of the array to be built
    m_1 : ArrayLike
        Values of the idx-th components
    m_2 : ArrayLike
        Values of the idy-th components
    idx : ArrayLike
        First components
    idy : ArrayLike
        Second components

    Returns
    -------
    jax.Array
        The built array
    """
    m = jnp.empty(shape)

    m = m.at[..., idx].set(m_1)
    m = m.at[..., idy].set(m_2)
    return m


def separate_array_argument(
    fun: Callable,
    shape: Union[int, Sequence[int]],
    idx: ArrayLike,
    idy: Optional[ArrayLike] = None,
) -> Callable:
    """Wraps a function taking an array as the first argument to give a function taking
    the 'idx' and 'idy' components of said array along the last axis as the first and
    second arguments respectively

    Parameters
    ----------
    fun : Callable
        A function taking an array as its first argument to be wrapped
    shape : Union[int, Sequence[int]]
        Size of the array to be used as the first argument
    idx : ArrayLike
        Components of the array to be passed in the first argument of the resulting
        function
    idy : Optional[ArrayLike], optional
        Components of the array to be passed in the second argument of the resulting
        function , by default None, in which case said components will be the
        complement of idx

    Returns
    -------
    _type_
        _description_
    """
    if idy is None:
        idy = complementary_indices(shape if isinstance(shape, int) else shape[-1], idx)

    def wrapped(m_1, m_2, *args, **kwargs):
        m = combine_array(shape, m_1, m_2, idx, idy)
        return fun(m, *args, **kwargs)

    return wrapped


def wrap_to_2pi(angle):
    angle = np.atleast_1d(angle)
    positive_input = angle > 0
    angle = np.mod(angle, 2 * np.pi)
    angle[(angle == 0) & positive_input] = 2 * np.pi
    return angle


def wrap_to_pi(angle):
    angle = np.atleast_1d(angle)
    q = (angle < -np.pi) | (np.pi < angle)
    angle[q] = wrap_to_2pi(angle[q] + np.pi) - np.pi
    return angle


def cart_to_sph_3d(vector_3d: ArrayLike) -> jax.Array:
    """returns an array of 3d vectors in spherical coordinates
    Parameters
    ----------
    vector_3d: ArrayLike
        array of input vectors of shape (3,n). x, y, z components taken along 0-axis.
        ignores indices > 2 in the 0-axis if vector_3d.shape[0] > 3.
    Returns
    -------
    jax.Array
        array of shape (3,n). r, theta, phi components along 0-axis.
    """
    r = jnp.sqrt(vector_3d[0, :] ** 2 + vector_3d[1, :] ** 2 + vector_3d[2, :] ** 2)
    theta = jnp.arctan2(
        jnp.sqrt(vector_3d[0, :] ** 2 + vector_3d[1, :] ** 2), vector_3d[2, :]
    )
    phi = jnp.arctan2(vector_3d[1, :], vector_3d[0, :])
    return jnp.stack((r, theta, phi), axis=0)


def sph_to_cart_3d(vector_rtp: ArrayLike) -> jax.Array:
    """returns an array of 3d vectors in spherical coordinates
    Parameters
    ----------
    vector_rtp: ArrayLike
        array of input vectors of size (3,n). r, theta, phi components taken along 0-axis.
        ignores indices > 2 in the 0-axis if vector_3d.shape[0] > 3.
    Returns
    -------
    jax.Array
        array of size (3,n). x, y, z components along 0-axis.
    """
    x = vector_rtp[0, :] * jnp.sin(vector_rtp[1, :]) * jnp.cos(vector_rtp[2, :])
    y = vector_rtp[0, :] * jnp.sin(vector_rtp[1, :]) * jnp.sin(vector_rtp[2, :])
    z = vector_rtp[0, :] * jnp.cos(vector_rtp[1, :])
    return jnp.stack((x, y, z), axis=0)


def quat_to_eul(qs: ArrayLike) -> jax.Array:
    """returns an array of 3d vectors in Euler angles
    Parameters
    ----------
    qs: ArrayLike
        array of input vectors of size (4,n). quarternion components taken along 0-axis.
        ignores indices > 3 in the 0-axis if qs.shape[0] > 4.
        convention: qs[3,:] denotes the real part
        automatically normalises quarternion if qs is unnormalised
    Returns
    -------
    jax.Array
        array of size (3,n). phi, theta, psi components along 0-axis.
    """
    phi = jnp.arctan2(
        2
        * (qs[3, :] * qs[0, :] + qs[1, :] * qs[2, :])
        / (qs[0, :] ** 2 + qs[1, :] ** 2 + qs[2, :] ** 2 + qs[3, :] ** 2),
        1
        - 2
        * (qs[0, :] ** 2 + qs[1, :] ** 2)
        / (qs[0, :] ** 2 + qs[1, :] ** 2 + qs[2, :] ** 2 + qs[3, :] ** 2),
    )
    theta = -np.pi / 2 + 2 * jnp.arctan2(
        jnp.sqrt(
            1
            + 2
            * (qs[3, :] * qs[1, :] - qs[0, :] * qs[2, :])
            / (qs[0, :] ** 2 + qs[1, :] ** 2 + qs[2, :] ** 2 + qs[3, :] ** 2)
        ),
        jnp.sqrt(
            1
            - 2
            * (qs[3, :] * qs[1, :] - qs[0, :] * qs[2, :])
            / (qs[0, :] ** 2 + qs[1, :] ** 2 + qs[2, :] ** 2 + qs[3, :] ** 2)
        ),
    )
    psi = jnp.arctan2(
        2
        * (qs[3, :] * qs[2, :] + qs[0, :] * qs[1, :])
        / (qs[0, :] ** 2 + qs[1, :] ** 2 + qs[2, :] ** 2 + qs[3, :] ** 2),
        1
        - 2
        * (qs[1, :] ** 2 + qs[2, :] ** 2)
        / (qs[0, :] ** 2 + qs[1, :] ** 2 + qs[2, :] ** 2 + qs[3, :] ** 2),
    )
    return jnp.stack((phi, theta, psi), axis=0)


def eul_to_quat(euls: ArrayLike) -> jax.Array:
    """returns an array of quarternion in Euler angles
    Parameters
    ----------
    qs: ArrayLike
        array of input vectors of size (3,n). Euler angle components taken along 0-axis.
        ignores indices > 3 in the 0-axis if qs.shape[0] > 4.
        convention: phi = euls[0,:], theta = eu1s[1, :], psi = euls[2,:]
    Returns
    -------
    jax.Array
        array of size (4,n). quaternion components along 0-axis.
        convention: qs[3,:] is the real part.
    """
    q3 = jnp.cos(euls[0, :] / 2) * jnp.cos(euls[1, :] / 2) * jnp.cos(
        euls[2, :] / 2
    ) + jnp.sin(euls[0, :] / 2) * jnp.sin(euls[1, :] / 2) * jnp.sin(euls[2, :] / 2)
    q0 = jnp.sin(euls[0, :] / 2) * jnp.cos(euls[1, :] / 2) * jnp.cos(
        euls[2, :] / 2
    ) - jnp.cos(euls[0, :] / 2) * jnp.sin(euls[1, :] / 2) * jnp.sin(euls[2, :] / 2)
    q1 = jnp.cos(euls[0, :] / 2) * jnp.sin(euls[1, :] / 2) * jnp.cos(
        euls[2, :] / 2
    ) + jnp.sin(euls[0, :] / 2) * jnp.cos(euls[1, :] / 2) * jnp.sin(euls[2, :] / 2)
    q2 = jnp.cos(euls[0, :] / 2) * jnp.cos(euls[1, :] / 2) * jnp.sin(
        euls[2, :] / 2
    ) - jnp.sin(euls[0, :] / 2) * jnp.sin(euls[1, :] / 2) * jnp.cos(euls[2, :] / 2)
    return jnp.stack((q0, q1, q2, q3), axis=0)


def prefix_key(d: Dict[str, Any], prefix: str):
    return {f"{prefix}.{k}": v for k, v in d.items()}


def take_arrays(d: Dict[str, Any]) -> Archive:
    res = {}
    for k, v in d.items():
        try:
            v = np.asarray(v)
            if not np.issubdtype(v.dtype, np.number) and not np.issubdtype(
                v.dtype, np.string_
            ):
                continue
            res[k] = v
        except ValueError:
            pass
    return res


def flatten_dict(
    nested: Dict[str, Dict[str, Any]], delim: str = ".", parent_key: str = ""
) -> Dict[str, Any]:
    items = []
    for key, value in nested.items():
        new_key = parent_key + delim + key if parent_key else key
        if isinstance(value, MutableMapping):
            items.extend(flatten_dict(value, delim, parent_key=new_key).items())
        else:
            items.append((new_key, value))
    return dict(items)


def expand_dict(flat, delim: str = "."):
    result = {}

    def _expand_dict_impl(k: str, v: Any, result: Dict[str, Any]):
        k, *rest = k.split(delim, 1)
        if rest:
            _expand_dict_impl(rest[0], v, result.setdefault(k, {}))
        else:
            result[k] = v

    for k, v in flat.items():
        _expand_dict_impl(k, v, result)
    return result


def generate_leader_trajectory(timestamps, waypoints, t_sample, quadrotor_mass, ctl_dt):

    polys = ms.generate_trajectory(
        [ms.Waypoint(float(t), p) for t, p in zip(timestamps, waypoints)],
        5,
        idx_minimized_orders=[2, 3],
    )

    traj = ms.compute_quadrotor_trajectory(
        polys, t_sample, quadrotor_mass, yaw="velocity"
    )

    pc = geometric_controller.TrackingController(
        geometric_controller.TrackingControllerParams(
            k_pos=jnp.array([0.8, 0.8, 0.9]),
            k_vel=jnp.array([0.4, 0.4, 0.6]),
            max_z_accel=jnp.inf,
        )
    )
    ac = geometric_controller.AttitudeController(0.25)

    quad = quadrotor.Quadrotor(quadrotor_mass)

    @jax.jit
    def loop(state, tup):
        position, attitude, velocity, *_ = jnp.split(state, (3, 7))
        pc_out, _ = pc.run(pc.State(position, attitude, velocity), pc.Reference(*tup))
        ac_out, _ = ac.run(ac.State(attitude), ac.Reference(pc_out.orientation))
        u = jnp.concatenate([jnp.array([pc_out.thrust]), ac_out.body_rate])

        state = common.forward_dynamics(quad.dynamics, state, u, ctl_dt, "euler")
        return state, (state, u)

    state_in = traj.state[0, :]
    _, (x_leader, u_leader) = jax.lax.scan(
        loop, state_in, (traj.position, traj.velocity)
    )

    return x_leader, u_leader
