# Record recurrent states

`update_state_names` controls recording in `make_rnn`. Plain dotted strings
retain the existing `(outputs, stacked_states)` interface. Expressions from
`btorch.monitor` record traces, derived quantities, or temporal reductions,
available through `net.get_records()` after a forward call.

```python
from btorch.models.rnn import make_rnn
from btorch.monitor import col

# cell is a MemoryModule with a hidden memory named h.
net = make_rnn(
    cell,
    update_state_names=[
        "h",
        col("h").mean().alias("mean_h"),
        col("h").diff().alias("delta_h"),
    ],
)
outputs, states = net(inputs)  # inputs: [time, batch, features]
trace = states["h"]
mean_h = net.get_records()["mean_h"]  # [batch, hidden]
delta_h = net.get_records()["delta_h"]  # differences along time
```

A top-level mapping such as `{"mean_h": col("h").mean()}` routes all entries
through the recorder and disables `stacked_states`. Use a mixed list to retain
both channels. `None` records all memories through the existing states interface.
Plain module buffers require `allow_buffer=True`.

Temporal reductions (`mean`, `sum`, `min`, `max`, and custom `Reducer` folds)
thread a functional carry across unroll blocks and outer chunks. Streaming
reductions avoid materializing a full time trace. Raw traces still retain their
time dimension. Reductions compose with gradient checkpointing, including sparse
recurrence. Reduction and derived results are detached diagnostics; raw
`col("h")` traces retain their autograd connection.

Use `grad("h")` from `btorch.monitor` to record hidden-state gradients. Read its
history from `get_records()` after backward; this replaces `save_grad_history`,
`grad_state_names`, and `get_grad_history`. Gradient monitors require eager
execution. Value monitors support `torch.compile` and CPU offloading; CUDA graph
recording supports value monitors only when the trajectory fits in one chunk.
Call `clear_records()` to release the previous result, and reset network state
before starting an independent trajectory.
