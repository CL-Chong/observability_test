from collections.abc import MutableMapping
from typing import Any, Dict

import numpy as np
from numpy.typing import ArrayLike

Archive = Dict[str, ArrayLike]


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
