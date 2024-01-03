import functools
from collections.abc import Callable, MutableMapping, Sequence
from typing import Any, Dict, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike

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
