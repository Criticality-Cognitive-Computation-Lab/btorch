"""Dictionary manipulation utilities.

Helpers for transforming, flattening, and mapping nested dictionaries
commonly used in configuration and data preprocessing pipelines.
"""

from typing import Any, Callable, Sequence


def reverse_map(map: dict[Any, Any | Sequence[Any]]) -> dict[Any, Any]:
    """Reverse a mapping, handling sequence values.

    Flattens sequence values so each item maps to the original key.
    Non-sequence values map directly.

    Args:
        map: Dictionary with scalar or sequence values.

    Returns:
        Reversed mapping where each original value (or sequence item)
        maps to its original key.

    Example:
        >>> reverse_map({"a": [1, 2], "b": 3})
        {1: "a", 2: "a", 3: "b"}
    """
    ret = {}
    for key, items in map.items():
        if isinstance(items, Sequence) and not isinstance(items, str):
            for item in items:
                ret[item] = key
        else:
            ret[items] = key
    return ret


def recurse_dict(d: dict, mapper: Callable, include_sequence: bool = False) -> dict:
    """Recursively apply function to dictionary leaf values.

    Args:
        d: Input dictionary (potentially nested).
        mapper: Function called with (key, value) for each leaf.
        include_sequence: If True, also recurse into tuples and lists.

    Returns:
        New dictionary with transformed leaf values.
    """

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
    """Flatten nested dictionary into single-level dictionary.

    Args:
        d: Nested dictionary to flatten.
        dot: If True, use dot-notation keys ("a.b"). If False,
            use tuple keys (("a", "b")).

    Returns:
        Flattened dictionary.

    Example:
        >>> flatten_dict({"a": {"b": 1}, "c": 2})
        {("a", "b"): 1, ("c",): 2}
        >>> flatten_dict({"a": {"b": 1}}, dot=True)
        {"a.b": 1}
    """

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
    """Unflatten dictionary with compound keys into nested structure.

    Args:
        flattened_dict: Dictionary with tuple or dot-notation keys.
        dot: If True, split keys on dots. If False, keys are tuples.

    Returns:
        Nested dictionary.

    Example:
        >>> unflatten_dict({("a",): 1, ("b", "c"): 2})
        {"a": 1, "b": {"c": 2}}
        >>> unflatten_dict({"a.b": 1}, dot=True)
        {"a": {"b": 1}}
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
