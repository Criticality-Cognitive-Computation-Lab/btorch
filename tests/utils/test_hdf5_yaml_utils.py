"""Round-trip tests for the HDF5 and YAML serialization helpers."""

import numpy as np

from btorch.utils.dict_utils import (
    flatten_dict,
    recurse_dict,
    reverse_map,
    unflatten_dict,
)
from btorch.utils.hdf5_utils import load_dict_from_hdf5, save_dict_to_hdf5
from btorch.utils.pandas_utils import groupby_to_dict
from btorch.utils.yaml_utils import load_yaml, save_yaml


def test_hdf5_roundtrip_nested(tmp_path):
    """Nested dicts, arrays and scalars survive a save/load cycle."""
    data = {
        "a": np.arange(6).reshape(2, 3),
        "nested": {"b": np.ones(4), "scalar": 3.5},
        "skipped": None,  # None values are silently dropped on save
    }
    # Use the (folder, filename) calling convention.
    save_dict_to_hdf5(data, tmp_path, filename="x.h5")
    out = load_dict_from_hdf5(tmp_path / "x.h5")

    assert "skipped" not in out
    np.testing.assert_array_equal(out["a"], data["a"])
    np.testing.assert_array_equal(out["nested"]["b"], data["nested"]["b"])
    assert out["nested"]["scalar"] == 3.5


def test_hdf5_compression_threshold(tmp_path):
    """Arrays above the threshold are compressed; content is unchanged."""
    arr = np.zeros(1000, dtype=np.float32)
    # threshold of 0 forces the compression branch; gzip avoids plugin needs.
    save_dict_to_hdf5(
        {"arr": arr}, tmp_path / "c.h5", compression="gzip", compression_threshold=0
    )
    out = load_dict_from_hdf5(tmp_path, "c.h5")
    np.testing.assert_array_equal(out["arr"], arr)


def test_yaml_roundtrip(tmp_path):
    """save_yaml/load_yaml work with both path conventions."""
    obj = {"lr": 0.1, "layers": [1, 2, 3]}
    save_yaml(obj, str(tmp_path / "sub"), "cfg.yaml")  # folder + filename
    assert load_yaml(str(tmp_path / "sub"), "cfg.yaml") == obj

    save_yaml(obj, str(tmp_path / "deep" / "cfg2.yaml"))  # full path
    assert load_yaml(str(tmp_path / "deep" / "cfg2.yaml")) == obj


def test_yaml_falls_back_to_object_dict(tmp_path):
    """Objects that safe_dump rejects are saved via their ``__dict__``."""

    class Args:
        def __init__(self):
            self.x = 1
            self.y = "a"

    save_yaml(Args(), str(tmp_path), "args.yaml")
    assert load_yaml(str(tmp_path / "args.yaml")) == {"x": 1, "y": "a"}


def test_dict_utils_basics():
    """reverse_map / flatten / unflatten / recurse_dict behave as
    documented."""
    assert reverse_map({"a": [1, 2], "b": 3, "c": "s"}) == {
        1: "a",
        2: "a",
        3: "b",
        "s": "c",
    }

    nested = {"a": {"b": 1}, "c": 2}
    assert flatten_dict(nested) == {("a", "b"): 1, ("c",): 2}
    assert flatten_dict(nested, dot=True) == {"a.b": 1, "c": 2}
    # Flatten then unflatten is the identity for both key styles.
    assert unflatten_dict(flatten_dict(nested)) == nested
    assert unflatten_dict(flatten_dict(nested, dot=True), dot=True) == nested

    # mapper receives (key, value); sequences are only entered on request.
    doubled = recurse_dict({"a": 1, "b": {"c": 2}}, lambda k, v: v * 2)
    assert doubled == {"a": 2, "b": {"c": 4}}
    seq = recurse_dict({"a": [1, 2]}, lambda k, v: v + 1, include_sequence=True)
    assert seq == {"a": [2, 3]}


def test_groupby_to_dict():
    """groupby_to_dict returns one sub-frame per key, optionally column-
    limited."""
    import pandas as pd

    df = pd.DataFrame({"a": [1, 1, 2], "b": [3, 4, 5]})
    full = groupby_to_dict(df, by="a")
    assert set(full) == {1, 2}
    assert list(full[1]["b"]) == [3, 4]

    sel = groupby_to_dict(df, column_select=["b"], by="a")
    assert list(sel[2].columns) == ["b"]
