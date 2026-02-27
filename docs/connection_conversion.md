# Connection Layer Conversion

This page documents conversion utilities for connection layers:

- `DenseConn`
- `SparseConn`
- `SparseConstrainedConn`

The implementation lives in `btorch.models.connection_conversion`.

## Overview

Two layouts are supported:

1. `base`: connection shape `(N_pre, N_post)`
2. `heter`: connection shape `(N_pre, N_post * n_receptor)`

Supported receptor modes:

1. `neuron`
2. `connection`

By default, `base -> heter` uses no-split semantics: each nonzero base edge maps
into exactly one receptor channel.

Dense conversion is implemented directly on dense tensors.

## API

### `convert_connection_layer`

```python
from btorch.models.connection_conversion import convert_connection_layer
```

Converts a live connection layer instance between `base` and `heter` layouts.

Common inputs:

- `target_layout`: `"base"` or `"heter"`
- `receptor_type_mode`: optional; required only when inferring
  base->heter assignment from `neurons`
- `receptor_type_index`: receptor channel table including `receptor_index`

For `base -> heter`:

- no-split default:
  - provide `edge_receptor_assignment`, or
  - in `neuron` mode, provide `neurons` + `receptor_type_col`
- optional split mode:
  - set `allow_weight_split=True`
  - provide `edge_receptor_weight` with `weight_coeff`

### `convert_connection_layer_from_checkpoint`

```python
from btorch.models.connection_conversion import (
    convert_connection_layer_from_checkpoint,
)
```

Converts from serialized weights (`state_dict`) plus caller-supplied topology.

Topology is intentionally provided by higher-level APIs:

- sparse sources: `conn` is required
- constrained sparse sources: `constraint` is required
- constrained sparse sources:
  - if `conn` is provided, it is authoritative for initial edge weights
  - if `conn` is omitted, `state_dict` must provide both `initial_weight`
    and `indices`
- `source_class` is passed as a class object (`DenseConn`, `SparseConn`,
  or `SparseConstrainedConn`)

The function reconstructs the source layer, loads `state_dict`, then applies the
same conversion logic as `convert_connection_layer`.

## Minimal Example

```python
import pandas as pd
from btorch.models.connection_conversion import convert_connection_layer
from btorch.models.linear import SparseConn

layer = SparseConn(conn=conn_base, enforce_dale=False)
layer_heter = convert_connection_layer(
    layer,
    target_layout="heter",
    receptor_type_mode="connection",
    receptor_type_index=pd.DataFrame(
        [(0, "E"), (1, "I")],
        columns=["receptor_index", "receptor_type"],
    ),
    edge_receptor_assignment=edge_receptor_assignment,
)
```

## Notes

- Passing split tables while `allow_weight_split=False` raises `ValueError`.
- For constrained conversion, `group_policy` controls whether receptor-expanded
  groups are independent (`"independent"`) or shared (`"shared"`).
- Cross-family output override (`DenseConn` -> sparse, sparse -> `DenseConn`) is
  intentionally not supported.
