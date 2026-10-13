"""Inspect scientific responses of existing stateful Torch models."""

import copy
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor, nn

from btorch.models import environ
from btorch.models.base import MemoryModule
from btorch.models.functional import named_hidden_states, set_hidden_states
from btorch.models.rnn import RecurrentNNAbstract


@dataclass(frozen=True)
class ModelSnapshot:
    """Hold diagnostic continuation state, including explicit extra attributes.

    This is an inference snapshot, not an autograd checkpoint. Python or
    external RNG generators and unregistered state must be supplied
    explicitly by callers. Parameters are restored in place, preserving
    parameter sharing.
    """

    hidden_states: dict[str, Any]
    state_dict: dict[str, Any]
    extra_states: dict[str, Any]
    buffers: dict[str, Tensor]
    parameters: dict[str, Tensor]
    cpu_rng_state: Tensor
    cuda_rng_states: list[Tensor] | None


def _clone_value(value: Any, memo: dict, storages: dict) -> Any:
    if id(value) in memo:
        return memo[id(value)]
    if isinstance(value, Tensor):
        if value.layout != torch.strided or value.is_conj() or value.is_neg():
            raise ValueError(
                "Provide a snapshot adapter for non-strided/conjugate state"
            )
        key = (value.device, value.dtype, value.untyped_storage().data_ptr())
        if key not in storages:
            size = value.untyped_storage().nbytes() // value.element_size()
            storages[key] = value.detach().as_strided((size,), (1,), 0).clone()
        result = storages[key].as_strided(
            value.shape, value.stride(), value.storage_offset()
        )
        memo[id(value)] = result
        return result
    if isinstance(value, dict):
        result = {
            key: _clone_value(item, memo, storages) for key, item in value.items()
        }
    elif isinstance(value, (list, tuple)):
        result = type(value)(_clone_value(item, memo, storages) for item in value)
    else:
        result = copy.deepcopy(value, memo)
    memo[id(value)] = result
    return result


def snapshot_model(
    model: nn.Module, *, extra_state_names: Sequence[str] = ()
) -> ModelSnapshot:
    """Capture memories, parameters, buffers and Torch RNG without advancing
    them.

    Args:
        model: Stateful model to inspect.
        extra_state_names: Dotted attributes not registered as memories or buffers.

    Returns:
        Independent diagnostic snapshot. CUDA RNG is captured only when initialized.
    """
    extra = {}
    for name in extra_state_names:
        parent, _, leaf = name.rpartition(".")
        extra[name] = getattr(model.get_submodule(parent), leaf)
    # Keep the whole source tree alive while cloning: temporary dict IDs can
    # otherwise be reused and accidentally mistaken for an existing alias.
    source = {
        "hidden_states": named_hidden_states(model),
        "state_dict": model.state_dict(),
        "extra_states": extra,
        "buffers": dict(model.named_buffers(remove_duplicate=False)),
        "parameters": dict(model.named_parameters(remove_duplicate=False)),
    }
    cloned = _clone_value(source, {}, {})
    return ModelSnapshot(
        **cloned,
        cpu_rng_state=torch.get_rng_state().clone(),
        cuda_rng_states=(
            torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
        ),
    )


def restore_model(model: nn.Module, snapshot: ModelSnapshot) -> None:
    """Restore an inference snapshot without replacing parameter objects.

    Args:
        model: Model with the same structure as the captured model.
        snapshot: Snapshot from :func:`snapshot_model`.
    """
    model.load_state_dict(snapshot.state_dict)
    memo, storages = {}, {}
    live = dict(model.named_buffers(remove_duplicate=False))
    for name, saved in snapshot.parameters.items():
        value = model.get_parameter(name)
        memo[id(saved)] = value
        key = (saved.device, saved.dtype, saved.untyped_storage().data_ptr())
        size = value.untyped_storage().nbytes() // value.element_size()
        storages[key] = value.detach().as_strided((size,), (1,), 0)
    with torch.no_grad():
        for name, saved in snapshot.buffers.items():
            value = live[name]
            if value.shape != saved.shape or value.dtype != saved.dtype:
                raise ValueError(
                    f"Buffer {name!r} changed shape/dtype; use a state adapter"
                )
            value.copy_(saved)
            memo[id(saved)] = value
            key = (saved.device, saved.dtype, saved.untyped_storage().data_ptr())
            size = value.untyped_storage().nbytes() // value.element_size()
            storages[key] = value.detach().as_strided((size,), (1,), 0)
    set_hidden_states(model, _clone_value(snapshot.hidden_states, memo, storages))
    for name, value in snapshot.extra_states.items():
        parent, _, leaf = name.rpartition(".")
        setattr(model.get_submodule(parent), leaf, _clone_value(value, memo, storages))
    torch.set_rng_state(snapshot.cpu_rng_state)
    if snapshot.cuda_rng_states is not None:
        torch.cuda.set_rng_state_all(snapshot.cuda_rng_states)


@dataclass(frozen=True)
class ResponseTrace:
    """Hold input, output and post-step state traces on CPU.

    Time is the leading axis. State trailing axes are retained without dropping
    receptor or adaptation channels. ``dt`` uses the model's time unit.
    """

    x_seq: Tensor
    spikes: Tensor
    states: dict[str, Tensor]
    final_state: ModelSnapshot
    dt: float


def simulate_response(
    model: nn.Module,
    x_seq: Tensor,
    *,
    dt: float = 1.0,
    state_names: Sequence[str] | None = None,
    extra_state_names: Sequence[str] = (),
    sequence: Callable[[Tensor], tuple[Tensor, dict[str, Tensor]]] | None = None,
) -> ResponseTrace:
    """Run the existing btorch sequence path and collect diagnostic traces.

    Args:
        model: Initialized model. Its state advances; training mode is unchanged.
        x_seq: Time-leading input, conventionally ``[time, batch, neuron]``.
        dt: Positive finite step duration in model time units.
        state_names: Exact dotted memory names. None selects all tensor memories
            for RecurrentNN or an explicit adapter; standalone primitives default
            to no histories. An empty sequence explicitly requests no histories.
        extra_state_names: Additional snapshot attributes, such as an external cursor.
        sequence: Optional adapter returning outputs and post-step state histories.
            Without an adapter, RecurrentNN uses its native sequence entry and
            standalone MemoryModule uses its own multi_step_forward. Primitive
            state histories require an explicit adapter; no wrapper replaces its
            continuous sequence kernel or changes its continuation semantics.

    Returns:
        CPU traces and a final continuation snapshot. No backward graph is retained.

    Notes:
        The primitive owns its sequence semantics, including empty inputs and
        state continuation. For example, a Torch PSC convolution path can be
        stateless while its GPU sequence kernel advances PSC state; this helper
        does not substitute a single-step loop to hide that difference. An adapter
        collecting single-step reference histories is diagnostic evidence for
        that reference path, not parity evidence for the primitive sequence op.
        RecurrentNN without an adapter retains the diagnostic empty-input
        convention of returning an empty input-shaped output without advancing.
    """
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be positive and finite")
    if x_seq.ndim < 2:
        raise ValueError("x_seq must have time and feature axes")
    states = named_hidden_states(model)
    selected = (
        tuple(name for name, value in states.items() if isinstance(value, Tensor))
        if state_names is None
        and (sequence is not None or isinstance(model, RecurrentNNAbstract))
        else tuple(state_names or ())
    )
    unknown = set(selected).difference(states)
    if unknown:
        raise ValueError(f"Unknown memory names: {sorted(unknown)}")
    if sequence is None and not isinstance(model, RecurrentNNAbstract):
        if not isinstance(model, MemoryModule):
            raise TypeError("Provide a sequence adapter for a non-btorch model")
        if selected:
            raise ValueError(
                "Primitive state histories require an explicit sequence adapter"
            )
    if (
        not x_seq.shape[0]
        and isinstance(model, RecurrentNNAbstract)
        and sequence is None
    ):
        stacked = {
            name: states[name].detach().cpu().new_empty((0, *states[name].shape))
            for name in selected
        }
        outputs = x_seq.detach().cpu().clone()
    else:
        modes = [
            (module, module.step_mode)
            for module in model.modules()
            if hasattr(module, "step_mode")
        ]
        previous_names = (
            model.update_state_names if isinstance(model, RecurrentNNAbstract) else None
        )
        try:
            if sequence is None:
                if isinstance(model, RecurrentNNAbstract):
                    runner = model
                    previous_names = model.update_state_names
                    model.update_state_names = selected
                else:
                    runner = model
                sequence = runner.multi_step_forward
            with torch.no_grad(), environ.context(dt=float(dt)):
                result = sequence(x_seq)
                if isinstance(result, Tensor):
                    outputs, histories = result, {}
                else:
                    outputs, histories = result
            stacked = {
                name: histories[name].detach().cpu().clone() for name in selected
            }
            outputs = outputs.detach().cpu().clone()
        finally:
            if isinstance(model, RecurrentNNAbstract):
                model.update_state_names = previous_names
            for module, mode in modes:
                module.step_mode = mode
    return ResponseTrace(
        x_seq.detach().cpu().clone(),
        outputs,
        stacked,
        snapshot_model(model, extra_state_names=extra_state_names),
        float(dt),
    )


def validate_response(
    trace: ResponseTrace,
    *,
    expectations: Mapping[str, Callable[[ResponseTrace], bool]] | None = None,
) -> list[str]:
    """Return numerical failures without imposing attractor or spike
    assumptions.

    Args:
        trace: Response to inspect.
        expectations: Named model-specific numerical predicates.

    Returns:
        Failures identifying the first nonfinite time and index, or expectation.
    """
    failures = []
    for name, values in {
        "input": trace.x_seq,
        "spikes": trace.spikes,
        **trace.states,
    }.items():
        bad = (~torch.isfinite(values)).nonzero()
        if bad.numel():
            index = tuple(bad[0].tolist())
            failures.append(
                f"{name}: nonfinite at index={index}, time={index[0] * trace.dt:g}"
            )
    for name, predicate in (expectations or {}).items():
        if not predicate(trace):
            failures.append(f"expectation failed: {name}")
    return failures


def describe_response(
    trace: ResponseTrace,
    *,
    max_values: int = 128,
    units: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Describe named response tensors with directly readable diagnostic
    values.

    Args:
        trace: Collected response. Reading it never changes states or RNG.
        max_values: Largest tensor whose complete nested values are included.
        units: Explicit units keyed by input, spikes, dotted state name or ``dt``.
            No physical units or batch/neuron axis meanings are inferred.

    Returns:
        JSON-ready metadata, all small values and finite-value summaries of
        large tensors. Nonfinite values use string labels instead of JSON NaN.
    """
    if max_values < 0:
        raise ValueError("max_values must be nonnegative")
    units = units or {}
    descriptions = {}
    for name, value in {
        "input": trace.x_seq,
        "spikes": trace.spikes,
        **trace.states,
    }.items():
        path = name if name in ("input", "spikes") else f"states.{name}"
        finite = torch.isfinite(value)
        finite_values = value[finite].double()
        bad = (~finite).nonzero()
        first = bad[0].tolist() if bad.numel() else None
        item = {
            "path": path,
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "device": str(value.device),
            "axes": ["time", *[f"dimension_{i}" for i in range(1, value.ndim)]],
            "units": units.get(name),
            "numel": value.numel(),
            "nonfinite_count": int((~finite).sum()),
            "first_nonfinite": first,
            "first_nonfinite_time": first[0] * trace.dt if first else None,
            "finite_summary": {
                "min": finite_values.min().item() if finite_values.numel() else None,
                "max": finite_values.max().item() if finite_values.numel() else None,
                "mean": finite_values.mean().item() if finite_values.numel() else None,
            },
        }
        if value.numel() <= max_values:

            def json_value(values):
                if isinstance(values, list):
                    return [json_value(element) for element in values]
                if isinstance(values, float) and not math.isfinite(values):
                    return str(values)
                return values

            item["values"] = json_value(value.tolist())
        descriptions[name] = item
    return {
        "dt": trace.dt,
        "dt_unit": units.get("dt", "model time unit"),
        "state_phase": "after the completed logical timestep",
        "tensors": descriptions,
    }


@dataclass(frozen=True)
class ResponsePlotOptions:
    """Set response figure units and optional explicit batch selection.

    Attributes:
        units: State-path units plus ``input``; omitted units remain unspecified.
        time_unit: Unit of dt; no numerical time conversion is performed.
        batch_index: Select a batch for conventional [time, batch, neuron]
            traces. None displays every batch. All state channels are retained.
    """

    units: Mapping[str, str] = field(default_factory=dict)
    time_unit: str = "model units"
    batch_index: int | None = None


def plot_response(
    trace: ResponseTrace,
    *,
    title: str = "",
    reference: ResponseTrace | None = None,
    options: ResponsePlotOptions | None = None,
):
    """Plot aligned input, colored spike raster and separate state channels.

    Args:
        trace: Candidate diagnostic trace.
        title: Case/model identifiers.
        reference: Optional golden trace, using matching colors and dashed lines.
        options: Units and explicit batch selection. Every recorded channel is
            plotted; unspecified layouts retain their complete flattened values.

    Returns:
        Matplotlib figure. The caller saves and closes it. Upstream tests use
        ``save_fig(..., file=__file__)`` to preserve repository figure layout.
    """
    import itertools

    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.lines import Line2D

    options = options or ResponsePlotOptions()
    if reference is not None and (
        reference.dt != trace.dt or reference.spikes.shape != trace.spikes.shape
    ):
        raise ValueError(
            "Reference and candidate must have matching time axes and outputs"
        )
    if options.batch_index is not None and (
        trace.spikes.ndim != 3 or not 0 <= options.batch_index < trace.spikes.shape[1]
    ):
        raise ValueError(
            "batch_index requires [time, batch, neuron] output and a valid batch"
        )
    count = trace.spikes.shape[0]
    time = np.arange(count) * trace.dt

    def select(values):
        if (
            options.batch_index is not None
            and values.ndim >= 3
            and values.shape[1] == trace.spikes.shape[1]
        ):
            return values[:, options.batch_index]
        return values

    spikes = select(trace.spikes)
    actor_shape = tuple(spikes.shape[1:])
    actor_count = math.prod(actor_shape)
    actor_indices = list(itertools.product(*(range(size) for size in actor_shape)))
    actor_labels = [
        (
            f"N{index[0]}"
            if len(index) == 1
            else f"B{index[0]} · N{index[1]}"
            if len(index) == 2
            else str(index)
        )
        for index in actor_indices
    ]
    colors = [plt.get_cmap("tab10")(index % 10) for index in range(actor_count)]
    panels = [
        ("input", select(trace.x_seq), None, trace.x_seq.shape == trace.spikes.shape)
    ]
    for name, recorded in trace.states.items():
        # Ordinary membrane/PSC/adaptation traces have the spike layout followed
        # by channel axes. Keep each component in its own panel, all neurons intact.
        if tuple(recorded.shape[: trace.spikes.ndim]) == tuple(trace.spikes.shape):
            values = select(recorded)
            channels = tuple(values.shape[spikes.ndim :])
            for component in itertools.product(*(range(size) for size in channels)):
                suffix = " [" + ",".join(map(str, component)) + "]" if component else ""
                panels.append(
                    (
                        name + suffix,
                        values[(..., *component)] if component else values,
                        (name, component),
                        True,
                    )
                )
        else:
            # Unknown axes are not inferred as neurons or batch. Keep every
            # original index and use neutral lines outside the neuron legend.
            panels.append((name + "\n(all indices)", recorded, (name, ()), False))
    heights = [1.0, 1.1] + [1.5] * (len(panels) - 1)
    fig, axes = plt.subplots(
        len(panels) + 1,
        1,
        sharex=True,
        figsize=(11.5, 1.7 + sum(heights) * 1.35),
        gridspec_kw={"height_ratios": heights},
    )
    line_axes = [axes[0], *axes[2:]]
    stimulus = select(trace.x_seq)
    active = (
        stimulus.reshape(count, math.prod(stimulus.shape[1:])).ne(0).any(dim=1).numpy()
    )
    boundaries = np.diff(np.r_[False, active, False].astype(int))
    intervals = list(
        zip(np.flatnonzero(boundaries == 1), np.flatnonzero(boundaries == -1))
    )
    for axis in axes:
        for start, stop in intervals:
            axis.axvspan(
                start * trace.dt,
                stop * trace.dt,
                color="#f0b44d",
                alpha=0.10,
                linewidth=0,
            )
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", color="#e5e7eb", linewidth=0.6)
        axis.set_axisbelow(True)
        axis.tick_params(labelsize=9)
        axis.margins(x=0)
    for axis, (label, values, source, actor_layout) in zip(
        line_axes, panels, strict=True
    ):
        name = source[0] if source else "input"
        unit = options.units.get(name)
        axis.set_ylabel(label + (f"\n({unit})" if unit else ""), fontsize=9)
        flattened = values.reshape(count, math.prod(values.shape[1:])).numpy()
        line_colors = [
            colors[index % actor_count] if actor_layout and actor_count else "#64748b"
            for index in range(flattened.shape[1])
        ]
        for index in range(flattened.shape[1]):
            axis.plot(
                time,
                flattened[:, index],
                color=line_colors[index],
                linewidth=1.1,
                drawstyle="steps-post" if source is None else "default",
            )
        if reference is not None:
            other = reference.x_seq if source is None else reference.states.get(name)
            if other is not None:
                if actor_layout or source is None:
                    other = select(other)
                if source and source[1]:
                    other = other[(..., *source[1])]
                if other.shape != values.shape:
                    raise ValueError(f"Reference state layout differs for {name}")
                other = other.reshape(count, flattened.shape[1]).numpy()
                for index in range(other.shape[1]):
                    axis.plot(
                        time,
                        other[:, index],
                        color=line_colors[index],
                        linestyle="--",
                        drawstyle="steps-post" if source is None else "default",
                        alpha=0.75,
                        linewidth=1.0,
                    )
        if name.rsplit(".", 1)[-1] == "v":
            for parameter in ("v_threshold", "v_peak"):
                threshold = trace.final_state.state_dict.get(name[:-1] + parameter)
                if isinstance(threshold, Tensor):
                    for index, level in enumerate(threshold.detach().cpu().unique()):
                        axis.axhline(
                            level.item(),
                            color="#64748b",
                            linestyle=":",
                            linewidth=0.9,
                            label=parameter if index == 0 else None,
                        )
                    axis.legend(loc="upper right", fontsize=8, frameon=False)
    raster = axes[1]
    for response, marker in ((trace, "|"), (reference, "x")):
        if response is None:
            continue
        events = select(response.spikes).reshape(count, actor_count).nonzero()
        raster.scatter(
            time[events[:, 0]],
            events[:, 1],
            marker=marker,
            s=28 if marker == "|" else 12,
            linewidths=0.9,
            c=[colors[index] for index in events[:, 1].tolist()],
            zorder=3,
        )
    raster.set_ylabel("Spikes", fontsize=9)
    tick_indices = (
        list(range(actor_count))
        if actor_count <= 12
        else np.unique(np.linspace(0, actor_count - 1, 12, dtype=int)).tolist()
    )
    raster.set_yticks(
        tick_indices, [actor_labels[index] for index in tick_indices], fontsize=8
    )
    raster.set_ylim(-0.6, max(actor_count - 0.4, 0.6))
    axes[-1].set_xlabel(f"Time ({options.time_unit})", fontsize=10)
    axes[-1].set_xlim(0, max(count * trace.dt, trace.dt))
    handles = [
        Line2D([0], [0], color=color, linewidth=2, label=label)
        for color, label in zip(colors, actor_labels, strict=True)
    ]
    if len(handles) > 12:
        handles = handles[:12] + [
            Line2D([], [], color="none", label=f"+{actor_count - 12} traces shown")
        ]
    if reference is not None:
        handles.append(
            Line2D([0], [0], color="#475569", linestyle="--", label="Reference")
        )
    legend_rows = math.ceil(len(handles) / 8)
    header = 0.95 + legend_rows * 0.25
    fig.suptitle(
        title or "Model response",
        x=0.12,
        y=1 - 0.18 / fig.get_figheight(),
        ha="left",
        fontsize=14,
    )
    batch = f" · batch {options.batch_index}" if options.batch_index is not None else ""
    fig.text(
        0.12,
        1 - 0.49 / fig.get_figheight(),
        f"{count} steps · dt = {trace.dt:g} {options.time_unit}{batch}"
        " · shaded: nonzero input",
        fontsize=9,
        color="#64748b",
    )
    if handles:
        fig.legend(
            handles=handles,
            loc="upper left",
            bbox_to_anchor=(0.115, 1 - 0.63 / fig.get_figheight()),
            ncol=min(len(handles), 8),
            fontsize=8,
            frameon=False,
            handlelength=1.6,
        )
    fig.subplots_adjust(
        left=0.12,
        right=0.97,
        bottom=0.065,
        top=1 - header / fig.get_figheight(),
        hspace=0.32,
    )
    return fig
