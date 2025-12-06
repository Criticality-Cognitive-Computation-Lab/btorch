from typing import Any, Callable, Sequence


def reverse_map(map: dict[Any, Any | Sequence[Any]]) -> dict[Any, Any]:
    ret = {}
    for key, items in map.items():
        if isinstance(items, Sequence) and not isinstance(items, str):
            for item in items:
                ret[item] = key
        else:
            ret[items] = key
    return ret


def recurse_dict(d: dict, mapper: Callable, include_sequence: bool = False) -> dict:
    def _f(d, k):
        if isinstance(d, dict):
            return {k: _f(v, k) for k, v in d.items()}
        if include_sequence:
            if isinstance(d, tuple):
                return tuple(_f(ve, None) for ve in d)
            elif isinstance(d, list):
                return list(_f(ve, None) for ve in d)
        return mapper(k, d)

    return _f(d, None)


def flatten_dict(d, dot=False):
    """Transform nested dict like {"a": {"b": arr1}, "c": arr2} into {("a",
    "b"): arr1, ("c",): arr2}"""

    def _flatten_dict(d, parent_key):
        items = []
        for k, v in d.items():
            new_key = parent_key + "." + k if dot else parent_key + (k,)
            if isinstance(v, dict):
                items.extend(_flatten_dict(v, new_key))
            else:
                items.append((new_key, v))
        return items

    items = _flatten_dict(d, "" if dot else ())
    if dot:
        # remove the leading '.'
        items = [(k.lstrip("."), v) for k, v in items]
    return dict(items)


def unflatten_dict(flattened_dict, dot=False):
    """Transform a flattened dict with tuple keys back into a nested dict.

    Example:
        Input: {('a',): 1, ('b', 'c'): 2, ('b', 'd'): 3}
        Output: {'a': 1, 'b': {'c': 2, 'd': 3}}
    """
    result = {}
    for key_tuple, value in flattened_dict.items():
        if dot:
            key_tuple = key_tuple.split(".")
        current_level = result
        for i, key_part in enumerate(key_tuple):
            if i == len(key_tuple) - 1:
                # Assign the value at the last key part
                current_level[key_part] = value
            else:
                # Ensure the key part exists and is a dict, then move down
                if key_part not in current_level:
                    current_level[key_part] = {}
                current_level = current_level[key_part]
    return result
