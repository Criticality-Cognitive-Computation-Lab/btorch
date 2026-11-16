"""HELION D²CSR propagation and online STDP source fragments."""

from __future__ import annotations

import functools
from dataclasses import dataclass, field

import torch

from ..plasticity import _native_source, get_plasticity_template, stdp_scalar_statements
from .engine import _DoubleBufferedState
from .renderers import _TILE_ALL_LANES_MARKER


@dataclass(frozen=True)
class HelionPlasticityPhases:
    """Injected source for the two HELION plasticity phases."""

    propagation: tuple[str, ...]
    post_propagation: tuple[str, ...]
    local_dtypes: tuple[tuple[str, str], ...]
    pre_propagation: tuple[str, ...] = ()
    mutated_buffers: tuple[str, ...] = ()


def event_topology_bindings(topology, *, spike_list=False, shared=False):
    """Bind existing outgoing segments and incoming canonical edge IDs."""
    if len(topology) not in ((14, 15) if shared else (14,)):
        raise ValueError("Event learning requires prepared source/delay segments")
    bindings = dict(
        seg_pre=topology[10],
        seg_delay=topology[11],
        seg_ptr=topology[12],
        seg_edge_id=topology[13],
        seg_dst=topology[3],
        in_crow=topology[0],
        in_col=topology[1],
        in_edge_id=topology[8],
    )
    if spike_list or shared:
        bindings["src_seg_ptr"] = topology[9]
    if shared and len(topology) == 15:
        bindings["edge_segment"] = topology[14]
    return bindings


@functools.lru_cache(maxsize=128)
def _event_connection_source(
    backend,
    model,
    n,
    receptors,
    segments,
    edges,
    parameter_sizes,
    histories,
    selection,
    storage,
    sizes,
    scalar_dtype,
    row_partitions,
    state_domains,
):
    """Render the same event traversal with ordinary kernel boundaries."""
    from .engine import _Source

    source = _Source(
        backend,
        ((n + 127) // 128) * 128 * row_partitions,
        1,
        0,
        storage=storage,
        precise_exp=True,
        scalar_dtype=scalar_dtype,
    )
    for name, size in sizes:
        source.pointer(name, size)
    phases = prepare_stdp_phases(
        backend,
        model,
        n,
        receptors,
        1,
        segments,
        edges,
        parameter_sizes,
        history_names=histories,
        spike_selection=selection,
        edge_records=True,
        row_partitions=row_partitions,
        index="producer_index",
        state_domains=state_domains,
        state_destinations=tuple((name, "next_" + name) for name, _ in state_domains),
    )
    source.local_dtypes.update(dict(phases.local_dtypes))
    source.local_dtypes.update(producer_index="int64", event_partition="int64")
    source.statements(
        (
            "producer_index = "
            f"(tl.program_id(0) // {row_partitions}) * BLOCK + tl.arange(0, BLOCK)",
            f"event_partition = tl.program_id(0) % {row_partitions}",
        )
        if backend == "triton"
        else (
            f"producer_index = (index // 128 // {row_partitions}) * 128 + index % 128",
            f"event_partition = (index // 128) % {row_partitions}",
        )
    )
    if backend == "triton":
        source.indexed_statements(
            (
                "neuron_output = tl.load(spike_ptr + producer_index, "
                f"mask=producer_index < {n}, other=0.0)",
            )
        )
    else:
        # Cooperative row traversal includes every physical lane, including
        # the partial final block; only neuron loads use the logical mask.
        source.indexed_statements(
            (
                "neuron_output = 0.0",
                f"if producer_index < {n}:",
                "    neuron_output = spike_ptr[producer_index]",
            )
        )
    source.indexed_statements(phases.propagation)
    source.indexed_statements(
        (
            "if event_partition == 0"
            + (f" and producer_index < {n}" if backend == "tilelang" else "")
            + ":",
            *("    " + line for line in phases.post_propagation),
        )
    )
    return (
        _native_source(source.render(), backend, model),
        tuple(source.pointers),
        phases.mutated_buffers,
    )


@dataclass
class PreparedEventConnection:
    """Own ordinary propagation/learning bindings for one recurrent chunk."""

    output: torch.Tensor
    spikes: object
    arguments: list
    prepare_launch: object
    spike_slot: int
    clock_slot: int
    clocks: object
    buffers: dict
    runtime: object
    model: str
    backend: str
    states: dict
    parameters: dict
    snapshots: _DoubleBufferedState | None = None
    state_bindings: tuple = ((), ())
    launch: object = field(default=None, init=False, repr=False)

    def step(self, source, time_index):
        """Propagate old weights and commit one event transition."""
        if self.launch is None:
            # The producer has now populated spikes and queue counts. Native
            # tuning must never execute a consumer against uninitialized lists.
            self.arguments[self.spike_slot] = self.runtime.argument(
                source, source.numel()
            )
            self.arguments[self.clock_slot] = self.clocks[time_index]
            prepared = self.prepare_launch()
            spike_slot, clock_slot = self.spike_slot, self.clock_slot
            pair_bindings = self.state_bindings
            snapshots = self.snapshots
            if self.runtime.supports_pointer_offsets:
                # The trajectory owns fixed layouts and all sequence storage.
                # Reuse the ordinary recurrent scheduler's public-stream binding
                # instead of revalidating those layouts for every event step.
                clocks = tuple(value.data_ptr() for value in self.clocks)
                submit, constants, stream = self.runtime.bind_submission(prepared)
                arguments = [*prepared.arguments, *constants]

                def run(source, time_index, _stream=stream):
                    prepared.owners[spike_slot] = source
                    arguments[spike_slot] = source.data_ptr()
                    arguments[clock_slot] = clocks[time_index]
                    for slot, value in pair_bindings[
                        snapshots.steps % 2 if snapshots is not None else 0
                    ]:
                        arguments[slot] = value
                    submit(*arguments)

            else:
                runtime, arguments, clocks = self.runtime, self.arguments, self.clocks

                def run(source, time_index):
                    arguments[spike_slot] = runtime.argument(source, source.numel())
                    arguments[clock_slot] = clocks[time_index]
                    for slot, value in pair_bindings[
                        snapshots.steps % 2 if snapshots is not None else 0
                    ]:
                        arguments[slot] = value
                    prepared(arguments)

            self.launch = run
        self.launch(source, time_index)
        if self.snapshots is not None:
            self.snapshots.steps += 1
        return self.output

    def finish(self):
        """Materialize persisted traces once at the chunk boundary."""
        if self.snapshots is not None:
            self.snapshots.finish()
            return
        from ..plasticity import materialize

        materialize(
            self.backend,
            self.model,
            self.states,
            self.buffers["trace_tick"],
            self.parameters,
            self.buffers["stdp_dt"],
            self.buffers["final_step"],
        )


def prepare_event_connection(
    source,
    weight,
    states,
    parameters,
    topology,
    dt,
    steps,
    spec,
    *,
    records=None,
):
    """Prepare shared sparse event learning for the ordinary paired schedule.

    The preceding neuron/PSC kernel publishes spikes and clears the same
    drive workspace. Kernel boundaries replace cooperative barriers; no
    model equations, canonical weights or connectivity are duplicated.
    """
    from ....sparse.runtime.spike_list import SPIKE_BLOCK, SpikeList, consumer_programs
    from .engine import _compile, _tensor_storage
    from .renderers import get_runtime

    model = spec["plasticity"]
    backend = spec["event_connection"]
    selection = spec["spike_selection"]
    receptors = max(int(spec["receptors"]), 1)
    segments = int(spec["n_segments"])
    enable_fp_fusion = bool(spec.get("enable_fp_fusion", True))
    precise_division = bool(spec.get("precise_division", False))
    n = source.numel()
    if topology[2].numel() != n + 1:
        raise NotImplementedError(
            "Two-kernel sparse event learning currently supports one "
            "zero-delay population"
        )
    domains = tuple(tuple(item) for item in spec.get("plasticity_domains", ()))
    template = get_plasticity_template(model, domains)
    shared = any(domain != "edge" for _, domain in template.state_domains)
    domains = domains if shared else ()
    if template.events is None:
        raise NotImplementedError(
            "Two-kernel sparse learning requires exact event-only silent evolution"
        )
    dtype = torch.promote_types(weight.dtype, source.dtype)
    # Reuse native spike-list scheduling, capped by row length so a CTA has
    # at least one useful edge tile at typical fanout. Partitions own disjoint
    # edge offsets; only partition zero materializes inactive edge histories.
    row_partitions = min(
        consumer_programs(weight.numel(), n),
        max(1, -(-weight.numel() // (n * SPIKE_BLOCK))),
    )
    output = torch.zeros((1, n * receptors), dtype=dtype, device=source.device)
    clocks = torch.arange(steps + 1, dtype=torch.int64, device=source.device)
    buffers = {
        **states,
        "weight": weight,
        "spike": source,
        "drive": output,
        "stdp_dt": dt,
        "stdp_step": clocks[:1],
        "edge_pre": topology[5],
        "edge_post": topology[6],
        **event_topology_bindings(
            topology, spike_list=selection == "spike_list", shared=shared
        ),
        **{"stdp_" + name: value for name, value in parameters.items()},
        **{name + "_history": value for name, value in (records or {}).items()},
    }
    snapshots = _DoubleBufferedState(states) if shared else None
    if snapshots is not None:
        buffers.update(snapshots.buffers())
    else:
        buffers["trace_tick"] = torch.zeros_like(weight, dtype=torch.int32)
    queue = None
    if selection == "spike_list":
        groups = (n + SPIKE_BLOCK - 1) // SPIKE_BLOCK
        queue = SpikeList(
            torch.empty(groups * SPIKE_BLOCK, dtype=torch.int32, device=source.device),
            torch.empty(groups * SPIKE_BLOCK, dtype=torch.int64, device=source.device),
            torch.empty(groups, dtype=torch.int32, device=source.device),
            topology[2][1:] - topology[2][:-1],
            output,
            n,
        )
        buffers.update(queue.bindings())
    storage = _tensor_storage(buffers, set(buffers))
    rendered, declarations, mutations = _event_connection_source(
        backend,
        model,
        n,
        receptors,
        segments,
        weight.numel(),
        tuple(parameters[name].numel() for name in template.parameters),
        tuple(records or ()),
        selection,
        storage,
        tuple((name, value.numel()) for name, value in buffers.items()),
        str(weight.dtype).removeprefix("torch."),
        row_partitions,
        domains,
    )
    kernel = _compile(
        backend, rendered, enable_fp_fusion, precise_division, device=source.device
    )
    runtime = get_runtime(backend)
    arguments = [runtime.argument(buffers[name], size) for name, size in declarations]
    slots = {name: index for index, (name, _) in enumerate(declarations)}
    writes = tuple(slots[name] for name in (*mutations, "drive") if name in slots)
    state_bindings = (
        snapshots.bindings(
            runtime, declarations, pointers=runtime.supports_pointer_offsets
        )
        if snapshots is not None
        else ((), ())
    )
    prepare_launch = functools.partial(
        runtime.prepare_launch,
        kernel,
        arguments,
        ((n + 127) // 128) * 128 * row_partitions,
        changing=(
            slots["spike"],
            slots["stdp_step"],
            *(slot for slot, _value in state_bindings[0]),
        ),
        enable_fp_fusion=enable_fp_fusion,
        writes=writes,
        read_writes=tuple(
            slots[name]
            for name in ("weight", "drive", "trace_tick", *states)
            if name in slots and slots[name] in writes
        ),
    )
    buffers["final_step"] = clocks[-1:]
    return PreparedEventConnection(
        output,
        queue,
        arguments,
        prepare_launch,
        slots["spike"],
        slots["stdp_step"],
        runtime.bind_rows(clocks[:-1, None], 1),
        buffers,
        runtime,
        model,
        backend,
        states,
        parameters,
        snapshots,
        state_bindings,
    )


def continuous_stages(template):
    """Partition continuous writers around the two existing grid fences.

    Post-owned filters may run alongside their neuron before
    publication. One canonical edge owner then transmits the old weight
    and applies its transition. Compact post/segment writers commit
    afterwards, when every edge reader has finished consuming the
    preceding trace snapshot.
    """
    edges = [
        index for index, stage in enumerate(template.stages) if stage.domain == "edge"
    ]
    if template.events is not None or len(edges) != 1:
        raise NotImplementedError("Continuous HELION requires one canonical edge stage")
    index = edges[0]
    before, edge, after = (
        template.stages[:index],
        template.stages[index],
        template.stages[index + 1 :],
    )
    domains = {
        **dict(template.state_domains),
        **dict.fromkeys(template.parameters, "edge"),
        **dict(template.parameter_domains),
        **dict(template.signals),
        "weight": "edge",
        "dt": "scalar",
        "arrival": "segment",
        "post_event": "post",
    }
    if set(domains.values()) - {"edge", "post", "segment", "scalar"}:
        raise NotImplementedError(
            "Continuous HELION requires declared scalar/post/segment/edge domains"
        )
    if "weight" not in edge.reads or "weight" not in dict(edge.outputs):
        raise NotImplementedError(
            "Continuous HELION requires edge-owned weight reads and updates"
        )
    if any(
        stage.domain != "post"
        or any(domains[name] not in ("post", "scalar") for name in stage.reads)
        for stage in before
    ):
        raise NotImplementedError(
            "Before propagation, continuous HELION supports only "
            "post-owned local filters"
        )
    if any(
        stage.domain not in ("post", "segment")
        or any(domains[name] not in (stage.domain, "scalar") for name in stage.reads)
        for stage in after
    ):
        raise NotImplementedError(
            "After propagation, continuous HELION requires independent "
            "post/segment writers"
        )
    return before, edge, after


def prepare_continuous_phases(
    backend,
    model,
    n,
    n_receptor,
    n_delay,
    n_edge,
    parameter_sizes,
    signals,
    history_names=(),
    *,
    has_delay=False,
    has_receptor=False,
    n_segments=0,
    state_domains=(),
    signal_domains=(),
):
    """Compose continuous learning with one canonical-edge propagation pass.

    Neuron owners publish any post filters before the existing grid
    barrier. Each canonical edge then propagates its old local weight
    and advances continuous learning exactly once, even without either
    spike event. The second existing barrier publishes drive before PSC
    consumption and permits compact trace commits. No lazy clocks or
    incoming-edge writers participate in this strategy.
    """
    template = get_plasticity_template(model, state_domains, signal_domains)
    before, edge_stage, after = continuous_stages(template)
    if backend not in ("triton", "tilelang"):
        raise ValueError("Continuous HELION requires Triton or TileLang")
    sizes = dict(zip(template.parameters, parameter_sizes, strict=True))
    domains = {
        **dict(template.state_domains),
        **dict.fromkeys(template.parameters, "edge"),
        **dict(template.parameter_domains),
        **dict(template.signals),
        "weight": "edge",
        "dt": "scalar",
        "post_event": "post",
    }
    segment_reads = "segment" in domains.values()
    if segment_reads and n_segments < 1 and n_edge:
        raise ValueError("Segment-owned continuous learning requires prepared segments")
    triton = backend == "triton"
    capacity = ((n + 127) // 128) * 128 if triton else n

    def load(name, pointer, index, mask):
        if triton:
            expression = (
                f"tl.load({pointer}_ptr)"
                if index == "0"
                else f"tl.load({pointer}_ptr + {index}, mask={mask}, other=0)"
            )
            return f"{name} = {expression}"
        return f"{name} = {pointer}_ptr[{index}]"

    def store(pointer, index, value, mask):
        return (
            f"tl.store({pointer}_ptr + {index}, {value}, mask={mask})"
            if triton
            else f"{pointer}_ptr[{index}] = {value}"
        )

    def stage_lines(stage, index, mask):
        lines = []
        for name in stage.reads:
            if name == "arrival":
                if stage.domain == "segment":
                    lines.extend(
                        (
                            load("trace_pre", "seg_pre", index, mask),
                            load("trace_delay", "seg_delay", index, mask),
                        )
                    )
                    lines.extend(
                        (
                            *_triton_arrival(
                                "trace", "trace_pre", "trace_delay", mask, n, n_delay
                            ),
                            "arrival = trace_arrival",
                        )
                        if triton
                        else _tile_arrival("trace_pre", "trace_delay", n, n_delay)
                    )
                continue  # Edge arrival is resolved once by its canonical owner.
            if name == "dt":
                lines.append("dt = stdp_dt")
                continue
            domain = domains[name]
            item = (
                "0"
                if domain == "scalar" or sizes.get(name) == 1
                else domain
                if domain in ("post", "segment") and stage.domain == "edge"
                else index
            )
            pointer = (
                "stdp_dt"
                if name == "dt"
                else "spike"
                if name == "post_event"
                else "stdp_" + name
                if name in sizes
                else signals.get(name, name)
            )
            lines.append(load(name, pointer, item, mask))
        lines.extend(stage.statements)
        for name, value in stage.outputs:
            lines.append(store(name, index, value, mask))
            if name in history_names:
                count = {"post": n, "segment": n_segments, "edge": n_edge}[stage.domain]
                lines.append(
                    store(
                        name + "_history", f"stdp_step * {count} + {index}", value, mask
                    )
                )
        return lines

    pre = [
        load("stdp_step", "stdp_step", "0", "mask"),
        load("stdp_dt", "stdp_dt", "0", "mask"),
        *(line for stage in before for line in stage_lines(stage, "index", "mask")),
    ]
    passes = (n_edge + capacity - 1) // capacity
    propagation = [
        f"for edge_pass in {'tl.range' if triton else 'T.serial'}(0, {passes}):",
        f"    edge = index + edge_pass * {capacity}",
    ]
    body = []
    if triton:
        body.append(f"edge_mask = edge < {n_edge}")
        indent, edge_mask = "    ", "edge_mask"
    else:
        body.append(f"if edge < {n_edge}:")
        indent, edge_mask = "        ", "mask"
    edge_lines = [
        load("pre", "edge_pre", "edge", edge_mask),
        load("post", "edge_post", "edge", edge_mask),
    ]
    if segment_reads:
        edge_lines.append(
            load("segment", "edge_segment", "edge", edge_mask)
            if n_delay > 1
            else load("segment", "src_seg_ptr", "pre", edge_mask)
        )
    if has_delay:
        edge_lines.append(load("delay", "edge_delay", "edge", edge_mask))
    else:
        edge_lines.append("delay = 0")
    edge_lines.extend(
        _triton_arrival("edge", "pre", "delay", edge_mask, n, n_delay)
        if triton
        else _tile_arrival("pre", "delay", n, n_delay)
    )
    if triton:
        edge_lines.append("arrival = edge_arrival")
    if not has_receptor:
        edge_lines.append("receptor = 0")
    # The loaded weight remains the transmitted value even after its buffer
    # is updated. No other traversal writes this canonical edge.
    transition = stage_lines(edge_stage, "edge", edge_mask)
    weight_load = next(line for line in transition if line.startswith("weight ="))
    transition.remove(weight_load)
    edge_lines.append(weight_load)
    target = f"post * {n_receptor} + receptor"
    if triton:
        if has_receptor:
            edge_lines.append(
                load(
                    "receptor",
                    "edge_receptor",
                    "edge",
                    f"{edge_mask} & (arrival != 0.0)",
                )
            )
        edge_lines.append(
            f"tl.atomic_add(drive_ptr + {target}, weight * arrival, "
            f"mask={edge_mask} & (arrival != 0.0), sem='relaxed')"
        )
    else:
        edge_lines.append("if arrival != 0.0:")
        if has_receptor:
            edge_lines.append(
                "    " + load("receptor", "edge_receptor", "edge", edge_mask)
            )
        edge_lines.extend((f"    T.atomic_add(drive_ptr[{target}], weight * arrival)",))
    edge_lines.extend(transition)
    propagation.extend("    " + line for line in body)
    propagation.extend(indent + line for line in edge_lines)
    post = []
    for stage in after:
        count = n_segments if stage.domain == "segment" else n
        passes = (count + capacity - 1) // capacity
        post.extend(
            (
                f"for trace_pass in {'tl.range' if triton else 'T.serial'}"
                f"(0, {passes}):",
                f"    trace_index = index + trace_pass * {capacity}",
            )
        )
        if triton:
            post.append(f"    trace_valid = trace_index < {count}")
            indent = "    "
        else:
            post.append(f"    if trace_index < {count}:")
            indent = "        "
        post.extend(
            indent + line for line in stage_lines(stage, "trace_index", "trace_valid")
        )
    return HelionPlasticityPhases(
        tuple(propagation),
        tuple(post),
        tuple(
            (name, dtype)
            for name, dtype in (
                ("stdp_step", "int64"),
                ("edge", "int64"),
                ("pre", "int64"),
                ("post", "int64"),
                ("delay", "int64"),
                ("receptor", "int64"),
                ("segment", "int64"),
                ("trace_index", "int64"),
                ("trace_pass", "int64"),
                ("trace_pre", "int64"),
                ("trace_delay", "int64"),
                ("trace_valid", "bool"),
                ("trace_past", "bool"),
                ("edge_past", "bool"),
                ("edge_mask", "bool"),
            )
        ),
        tuple(pre),
        tuple(
            dict.fromkeys(
                name for stage in template.stages for name, _ in stage.outputs
            )
        )
        + tuple(name + "_history" for name in history_names),
    )


def _parameter_loads(backend, parameter_sizes, edge, mask, model):
    lines = []
    for name, size in zip(
        get_plasticity_template(model).parameters, parameter_sizes, strict=True
    ):
        index = "0" if int(size) == 1 else edge
        if backend == "triton":
            lines.append(
                f"{name} = tl.load(stdp_{name}_ptr)"
                if int(size) == 1
                else f"{name} = tl.load(stdp_{name}_ptr + {index}, "
                f"mask={mask}, other=0.0)"
            )
        else:
            lines.append(f"{name} = stdp_{name}_ptr[{index}]")
    return lines


def _triton_arrival(prefix, pre, delay, edge_mask, n, n_delay):
    """Emit a masked Triton arrival query for a vector of canonical edges."""
    if n_delay == 1:
        return (
            f"{prefix}_arrival = tl.load(spike_ptr + {pre}, "
            f"mask={edge_mask}, other=0.0)",
        )
    return (
        f"{prefix}_past = stdp_step >= {delay}",
        f"{prefix}_current = tl.load(spike_ptr + {pre}, mask={edge_mask}, other=0.0)",
        f"{prefix}_history = tl.load(spike_history_ptr + "
        f"(stdp_step - {delay}) * {n} + {pre}, "
        f"mask={edge_mask} & {prefix}_past, other=0.0)",
        f"{prefix}_initial = tl.load(delay_history_ptr + "
        f"({delay} - stdp_step - 1) * {n} + {pre}, "
        f"mask={edge_mask} & ~{prefix}_past, other=0.0)",
        f"{prefix}_arrival = tl.where({delay} == 0, {prefix}_current, "
        f"tl.where({prefix}_past, {prefix}_history, {prefix}_initial))",
    )


def _tile_arrival(pre, delay, n, n_delay, output="arrival"):
    if n_delay == 1:
        return (f"{output} = spike_ptr[{pre}]",)
    return (
        f"if {delay} == 0:",
        f"    {output} = spike_ptr[{pre}]",
        "else:",
        f"    if stdp_step >= {delay}:",
        f"        {output} = spike_history_ptr[(stdp_step - {delay}) * {n} + {pre}]",
        "    else:",
        f"        {output} = delay_history_ptr[({delay} - stdp_step - 1) * "
        f"{n} + {pre}]",
    )


def _shared_edge_update(
    backend, edge, mask, template, segment, post, n_delay, histories, n_edges
):
    """Read strict-old domain traces while unique edge owners update
    weights."""
    triton = backend == "triton"

    def load(name, pointer, index):
        return (
            f"{name} = tl.load({pointer}_ptr + ({index}) + "
            f"tl.zeros((BLOCK,), tl.int64), mask={mask}, other=0.0)"
            if triton
            else f"{name} = {pointer}_ptr[{index}]"
        )

    lines = [load("weight", "weight", edge)]
    if segment is None:
        # Incoming traversal already knows the source. Without delays each
        # source has at most one segment; delayed layouts use the cached inverse.
        lines.append(
            load("shared_segment", "src_seg_ptr", "pre")
            if n_delay == 1
            else load("shared_segment", "edge_segment", edge)
        )
        segment = "shared_segment"
    for name, domain in template.state_domains:
        lines.append(load(name, name, segment if domain == "segment" else post))
    # The incoming owner is restricted to arrival==0, so the resolved full
    # edge transition already suppresses LTD. Reuse the baseline's stage,
    # rather than independently expanding a second version of its equations.
    lines.append("dt = stdp_dt")
    lines.extend(template.stages[0].statements)
    lines.append(
        f"tl.store(weight_ptr + {edge}, next_weight, mask={mask})"
        if triton
        else f"weight_ptr[{edge}] = next_weight"
    )
    if "weight" in histories:
        lines.append(f"record_index = stdp_step * {n_edges} + {edge}")
        lines.append(
            f"tl.store(weight_history_ptr + record_index, next_weight, mask={mask})"
            if triton
            else "weight_history_ptr[record_index] = next_weight"
        )
    return lines


def _shared_trace_commit(
    backend,
    template,
    n,
    n_segments,
    n_delay,
    parameter_sizes,
    histories,
    *,
    destinations=(),
    index="index",
):
    """Commit compact event stages only after all edge readers reach the
    fence."""
    triton = backend == "triton"
    grid_capacity = ((n + 127) // 128) * 128 if triton else n
    sizes = dict(zip(template.parameters, parameter_sizes, strict=True))
    lines = []

    def load(name, pointer, index):
        if triton and index == "0":
            return f"{name} = tl.load({pointer}_ptr)"
        return (
            f"{name} = tl.load({pointer}_ptr + {index}, mask=shared_valid, other=0.0)"
            if triton
            else f"{name} = {pointer}_ptr[{index}]"
        )

    def store(pointer, index, value):
        return (
            f"tl.store({pointer}_ptr + {index}, {value}, mask=shared_valid)"
            if triton
            else f"{pointer}_ptr[{index}] = {value}"
        )

    # These are precisely the independent after-edge stages already validated
    # in the separate baseline, including their scalar coefficient contracts.
    cardinalities = {"segment": n_segments, "post": n}
    destinations = dict(destinations)
    for stage in template.stages[1:]:
        count = cardinalities[stage.domain]
        passes = (count + grid_capacity - 1) // grid_capacity
        loop = (
            f"for shared_pass in tl.range(0, {passes}):"
            if triton
            else f"for shared_pass in T.serial(0, {passes}):"
        )
        body = [
            f"shared_index = {index} + shared_pass * {grid_capacity}",
            f"shared_valid = shared_index < {count}",
        ]
        inner = []
        for name in stage.reads:
            if name == "dt":
                inner.append("dt = stdp_dt")
            elif name == "arrival":
                inner.extend(
                    (
                        load("shared_pre", "seg_pre", "shared_index"),
                        load("shared_delay", "seg_delay", "shared_index"),
                    )
                )
                inner.extend(
                    _triton_arrival(
                        "shared",
                        "shared_pre",
                        "shared_delay",
                        "shared_valid",
                        n,
                        n_delay,
                    )
                    if triton
                    else _tile_arrival(
                        "shared_pre",
                        "shared_delay",
                        n,
                        n_delay,
                        output="shared_arrival",
                    )
                )
                inner.append("arrival = shared_arrival")
            elif name == "post_event":
                inner.append(load("post_event", "spike", "shared_index"))
            elif name in sizes:
                inner.append(
                    load(
                        name,
                        "stdp_" + name,
                        "0" if sizes[name] == 1 else "shared_index",
                    )
                )
            else:
                inner.append(load(name, name, "shared_index"))
        inner.extend(stage.statements)
        for name, output in stage.outputs:
            inner.append(store(destinations.get(name, name), "shared_index", output))
            if name in histories:
                inner.append(
                    store(
                        name + "_history", f"stdp_step * {count} + shared_index", output
                    )
                )
        if triton:
            body.extend(inner)
        else:
            body.extend(["if shared_valid:", *("    " + line for line in inner)])
        lines.extend([loop, *("    " + line for line in body)])
    return lines


def _triton_edge_update(
    edge,
    edge_mask,
    *,
    model,
    ltp=True,
    ltd=True,
    histories=(),
    n_edges=None,
    state_domains=(),
    segment="active_segment",
    post="post",
    n_delay=1,
):
    template = get_plasticity_template(model, state_domains)
    if state_domains:
        return _shared_edge_update(
            "triton",
            edge,
            edge_mask,
            template,
            segment,
            post,
            n_delay,
            histories,
            n_edges,
        )
    lines = [
        f"weight = tl.load(weight_ptr + {edge}, mask={edge_mask}, other=0.0)",
        *(
            f"{name} = tl.load({name}_ptr + {edge}, mask={edge_mask}, other=0.0)"
            for name in template.states
        ),
        f"trace_tick = tl.load(trace_tick_ptr + {edge}, mask={edge_mask}, other=0)",
        "elapsed = tick - trace_tick",
        "elapsed_dt = elapsed * stdp_dt",
    ]
    lines.extend(
        stdp_scalar_statements(
            decay="elapsed_dt",
            include_ltp=ltp,
            include_ltd=ltd,
            model=model,
        )
    )
    lines.extend(
        (
            f"tl.store(weight_ptr + {edge}, next_weight, mask=update_mask)",
            *(
                f"tl.store({name}_ptr + {edge}, {output}, mask=update_mask)"
                for name, output in zip(
                    template.states, template.events.state_outputs, strict=True
                )
            ),
            f"tl.store(trace_tick_ptr + {edge}, tick, mask=update_mask)",
        )
    )
    if histories:
        outputs = {
            "weight": "next_weight",
            **dict(zip(template.states, template.events.state_outputs, strict=True)),
        }
        lines.append(f"record_index = stdp_step * {n_edges} + {edge}")
        lines.extend(
            f"tl.store({name}_history_ptr + record_index, {outputs[name]}, "
            "mask=update_mask)"
            for name in histories
        )
    return lines


def _tile_edge_update(
    edge,
    *,
    model,
    ltp=True,
    ltd=True,
    histories=(),
    n_edges=None,
    state_domains=(),
    segment="active_segment",
    post="post",
    n_delay=1,
):
    template = get_plasticity_template(model, state_domains)
    if state_domains:
        return _shared_edge_update(
            "tilelang",
            edge,
            "update_mask",
            template,
            segment,
            post,
            n_delay,
            histories,
            n_edges,
        )
    lines = [
        f"weight = weight_ptr[{edge}]",
        *(f"{name} = {name}_ptr[{edge}]" for name in template.states),
        f"trace_tick = trace_tick_ptr[{edge}]",
        "elapsed = tick - trace_tick",
        "elapsed_dt = elapsed * stdp_dt",
    ]
    lines.extend(
        stdp_scalar_statements(
            decay="elapsed_dt",
            include_ltp=ltp,
            include_ltd=ltd,
            model=model,
        )
    )
    lines.extend(
        (
            f"weight_ptr[{edge}] = next_weight",
            *(
                f"{name}_ptr[{edge}] = {output}"
                for name, output in zip(
                    template.states, template.events.state_outputs, strict=True
                )
            ),
            f"trace_tick_ptr[{edge}] = tick",
        )
    )
    if histories:
        outputs = {
            "weight": "next_weight",
            **dict(zip(template.states, template.events.state_outputs, strict=True)),
        }
        lines.append(f"record_index = stdp_step * {n_edges} + {edge}")
        lines.extend(
            f"{name}_history_ptr[record_index] = {outputs[name]}" for name in histories
        )
    return lines


def _triton_group_min(value, width):
    """Broadcast each traversal group's next active row to its lanes."""
    if width == 128:
        return f"tl.min({value}, axis=0)"
    if width == 1:
        return value
    return (
        f"tl.reshape(tl.broadcast_to(tl.min(tl.reshape({value}, (4, 32)), "
        "axis=1)[:, None], (4, 32)), (128,))"
    )


def spike_list_candidates(backend, index="index"):
    """Read the shared producer's compact entries without scanning spikes."""
    if backend == "triton":
        group = "tl.program_id(0)" if index == "index" else f"({index} // 128)"
        return (
            f"list_count = tl.load(spike_counts_ptr + {group})",
            "list_valid = tl.arange(0, BLOCK) < list_count",
            f"listed_source = tl.load(spike_indices_ptr + {index}, "
            "mask=list_valid, other=0)",
        )
    if backend == "tilelang":
        return (
            f"list_count = spike_counts_ptr[{index} // 128]",
            f"list_valid = {index} % 128 < list_count",
            "listed_source = 0",
            "if list_valid:",
            f"    listed_source = spike_indices_ptr[{index}]",
        )
    raise ValueError("Spike-list discovery requires Triton or TileLang")


def _triton_propagation(
    n,
    n_receptor,
    n_delay,
    n_segments,
    n_edges,
    parameter_sizes,
    model,
    *,
    parallelism="block",
    spike_list=False,
    histories=(),
    row_partitions=1,
    index="index",
    state_domains=(),
):
    """Walk disjoint active rows using thread, warp or block cooperation.

    Group selection changes traversal only. Outgoing owners still update
    both event terms; incoming owners update exclusively arrival-
    inactive edges. Empty groups participate in CTA-uniform loops with
    all memory masked, so a warp with no event cannot read the sentinel
    row or skip a collective.
    """
    width = {"thread": 1, "warp": 32, "block": 128}[parallelism]
    # Drive accumulation publishes no other data: the next consumer follows
    # either a stream kernel boundary or the fused path's explicit grid fence.
    # Per-edge acquire/release fences would serialize unrelated trace stores.
    grid_capacity = ((int(n) + 127) // 128) * 128
    segment_passes = (int(n_segments) + grid_capacity - 1) // grid_capacity

    def selected(value):
        return _triton_group_min(value, width)

    def any_active(value, limit):
        condition = f"{value} < {limit}"
        return (
            condition
            if width == 128
            else f"tl.sum(({condition}).to(tl.int32), axis=0) > 0"
        )

    def row_bounds(pointer, row, valid, indent):
        return [
            f"{indent}row_start = tl.load({pointer}_ptr + {row}, "
            f"mask={valid}, other=0)",
            f"{indent}row_end = tl.load({pointer}_ptr + {row} + 1, "
            f"mask={valid}, other=0)",
        ]

    row_length = (
        "row_end - row_start" if width == 128 else "tl.max(row_end - row_start, axis=0)"
    )
    lines = [
        "stdp_step = tl.load(stdp_step_ptr)",
        "tick = stdp_step + 1",
        "stdp_dt = tl.load(stdp_dt_ptr)",
        f"lane = tl.arange(0, BLOCK) % {width}",
    ]
    if spike_list:
        lines.extend(spike_list_candidates("triton", index))
    for segment_pass in range(segment_passes):
        if spike_list:
            # With no delays, a source owns at most one nonempty segment.
            # The queue's source IDs map through the existing segment index;
            # no inverse edge map or separate connectivity is allocated.
            lines.extend(
                (
                    "segment_id = tl.load(src_seg_ptr_ptr + listed_source, "
                    "mask=list_valid, other=0)",
                    "segment_end = tl.load(src_seg_ptr_ptr + listed_source + 1, "
                    "mask=list_valid, other=0)",
                    "segment_mask = list_valid & (segment_id < segment_end)",
                )
            )
        else:
            lines.extend(
                (
                    f"segment_id = {index} + {segment_pass} * {grid_capacity}",
                    f"segment_mask = segment_id < {n_segments}",
                )
            )
        lines.extend(
            (
                "segment_pre = tl.load(seg_pre_ptr + segment_id, "
                "mask=segment_mask, other=0)",
                "segment_delay = tl.load(seg_delay_ptr + segment_id, "
                "mask=segment_mask, other=0)",
            )
        )
        lines.extend(
            _triton_arrival(
                "segment", "segment_pre", "segment_delay", "segment_mask", n, n_delay
            )
        )
        lines.extend(
            (
                "remaining_segment = tl.where(segment_mask & (segment_arrival != 0.0), "
                f"segment_id, {n_segments})",
                f"active_segment = {selected('remaining_segment')}",
                f"while {any_active('active_segment', n_segments)}:",
                f"    group_active = active_segment < {n_segments}",
                "    active_pre = tl.load(seg_pre_ptr + active_segment, "
                "mask=group_active, other=0)",
                "    active_delay = tl.load(seg_delay_ptr + active_segment, "
                "mask=group_active, other=0)",
            )
        )
        lines.extend(
            "    " + line
            for line in _triton_arrival(
                "active", "active_pre", "active_delay", "group_active", n, n_delay
            )
        )
        lines.extend(row_bounds("seg_ptr", "active_segment", "group_active", "    "))
        lines.extend(
            (
                "    for edge_offset in tl.range("
                + (f"event_partition * {width}" if row_partitions > 1 else "0")
                + f", {row_length}, {width * row_partitions}):",
                "        edge = row_start + edge_offset + lane",
                "        update_mask = group_active & (edge < row_end)",
                "        canonical_edge = tl.load(seg_edge_id_ptr + edge, "
                "mask=update_mask, other=0)",
                "        dst = tl.load(seg_dst_ptr + edge, mask=update_mask, other=0)",
                f"        post = dst // {n_receptor}",
                "        post_event = tl.load(spike_ptr + post, "
                "mask=update_mask, other=0.0)",
                "        arrival = active_arrival",
            )
        )
        lines.extend(
            "        " + line
            for line in _parameter_loads(
                "triton", parameter_sizes, "canonical_edge", "update_mask", model
            )
        )
        lines.extend(
            "        " + line
            for line in _triton_edge_update(
                "canonical_edge",
                "update_mask",
                model=model,
                histories=histories,
                n_edges=n_edges,
                state_domains=state_domains,
                n_delay=n_delay,
            )
        )
        lines.extend(
            (
                "        tl.atomic_add(drive_ptr + dst, weight * active_arrival, "
                "mask=update_mask, sem='relaxed')",
                "    remaining_segment = tl.where(remaining_segment == active_segment, "
                f"{n_segments}, remaining_segment)",
                f"    active_segment = {selected('remaining_segment')}",
            )
        )
    post_candidates = (
        f"tl.where(list_valid, listed_source, {n})"
        if spike_list
        else f"tl.where(({index} < {n}) & (neuron_output != 0.0), {index}, {n})"
    )
    lines.extend(
        (
            f"remaining_post = {post_candidates}",
            f"active_post = {selected('remaining_post')}",
            f"while {any_active('active_post', n)}:",
            f"    group_active = active_post < {n}",
            "    post_event = tl.load(spike_ptr + active_post, "
            "mask=group_active, other=0.0)",
            f"    for receptor in tl.range(0, {n_receptor}):",
            f"        row_index = active_post * {n_receptor} + receptor",
        )
    )
    lines.extend(row_bounds("in_crow", "row_index", "group_active", "        "))
    lines.extend(
        (
            "        for edge_offset in tl.range("
            + (f"event_partition * {width}" if row_partitions > 1 else "0")
            + f", {row_length}, {width * row_partitions}):",
            "            edge = row_start + edge_offset + lane",
            "            edge_mask = group_active & (edge < row_end)",
            "            canonical_edge = tl.load(in_edge_id_ptr + edge, "
            "mask=edge_mask, other=0)",
            "            source_delay = tl.load(in_col_ptr + edge, "
            "mask=edge_mask, other=0)",
            f"            pre = source_delay // {max(n_delay, 1)}",
            f"            delay = source_delay % {max(n_delay, 1)}",
        )
    )
    lines.extend(
        "            " + line
        for line in _triton_arrival("incoming", "pre", "delay", "edge_mask", n, n_delay)
    )
    lines.extend(
        (
            "            update_mask = edge_mask & (incoming_arrival == 0.0)",
            "            arrival = incoming_arrival",
        )
    )
    lines.extend(
        "            " + line
        for line in _parameter_loads(
            "triton", parameter_sizes, "canonical_edge", "update_mask", model
        )
    )
    lines.extend(
        "            " + line
        for line in _triton_edge_update(
            "canonical_edge",
            "update_mask",
            model=model,
            ltp=True,
            ltd=False,
            histories=histories,
            n_edges=n_edges,
            state_domains=state_domains,
            segment=None,
            post="active_post",
            n_delay=n_delay,
        )
    )
    lines.extend(
        (
            "    remaining_post = tl.where(remaining_post == active_post, "
            f"{n}, remaining_post)",
            f"    active_post = {selected('remaining_post')}",
        )
    )
    return lines


def _tile_propagation(
    n,
    n_receptor,
    n_delay,
    n_segments,
    parameter_sizes,
    model,
    *,
    parallelism="warp",
    spike_list=False,
    histories=(),
    n_edges=None,
    row_partitions=1,
    index="index",
    state_domains=(),
):
    """Share event equations across thread, warp and full-CTA row traversal."""
    width = {"thread": 1, "warp": 32, "block": 128}[parallelism]
    block = width == 128
    grid_capacity = ((int(n) + 127) // 128) * 128
    segment_passes = (int(n_segments) + grid_capacity - 1) // grid_capacity
    lines = [
        "stdp_step = stdp_step_ptr[0]",
        "tick = stdp_step + 1",
        "stdp_dt = stdp_dt_ptr[0]",
        _TILE_ALL_LANES_MARKER,
        "lane = T.get_thread_binding()",
        f"valid_index = {index} < {n}",
    ]
    if block:
        # This binding is a CTA-wide selector. The TileLang renderer must keep
        # it as shared memory instead of lowering it as a scalar local.
        lines.append('active_row_shared = T.alloc_shared((4,), "int64")')
    if spike_list:
        lines.extend(spike_list_candidates("tilelang", index))

    def select(kind, candidate, sentinel, body):
        if width == 1:
            return [
                f"active_{kind} = {candidate}",
                f"if active_{kind} < {sentinel}:",
                *("    " + line for line in body),
            ]
        if width == 32:
            return [
                f"active_{kind} = T.warp_reduce_min({candidate})",
                f"while active_{kind} < {sentinel}:",
                *("    " + line for line in body),
                f"    {candidate} = {candidate} if {candidate} != active_{kind} "
                f"else {sentinel}",
                f"    active_{kind} = T.warp_reduce_min({candidate})",
            ]

        def reduce_rows():
            # Each warp owns one shared slot. A single lane combines four
            # minima, then the CTA reads the published row after the barrier.
            # This also avoids TileLang's signed-int64 shared atomic limitation.
            return [
                f"warp_candidate = T.warp_reduce_min({candidate})",
                "if lane % 32 == 0:",
                "    active_row_shared[lane // 32] = warp_candidate",
                "T.sync_threads()",
                "if lane == 0:",
                "    active_row_shared[0] = minimum(minimum(active_row_shared[0], "
                "active_row_shared[1]), minimum(active_row_shared[2], "
                "active_row_shared[3]))",
                "T.sync_threads()",
                f"active_{kind} = active_row_shared[0]",
            ]

        return [
            *reduce_rows(),
            f"while active_{kind} < {sentinel}:",
            *("    " + line for line in body),
            # All readers finish before the four shared slots are reused.
            "    T.sync_threads()",
            f"    {candidate} = {candidate} if {candidate} != active_{kind} "
            f"else {sentinel}",
            *("    " + line for line in reduce_rows()),
        ]

    def outgoing_body():
        lane_expr = f"lane % {width}"
        edge_step = width * row_partitions
        body = [
            "active_pre = seg_pre_ptr[active_segment]",
            "active_delay = seg_delay_ptr[active_segment]",
            "active_arrival = 0.0",
        ]
        body.extend(
            line
            for line in _tile_arrival(
                "active_pre", "active_delay", n, n_delay, output="active_arrival"
            )
        )
        body.extend(
            (
                "row_start = seg_ptr_ptr[active_segment]",
                "row_end = seg_ptr_ptr[active_segment + 1]",
                "for out_edge_offset in T.serial("
                + (f"event_partition * {width}" if row_partitions > 1 else "0")
                + ", row_end - row_start, "
                f"{edge_step}):",
                f"    out_edge = row_start + out_edge_offset + {lane_expr}",
                "    out_edge_valid = out_edge < row_end",
                "    if out_edge_valid:",
                "        canonical_edge = seg_edge_id_ptr[out_edge]",
                "        dst = seg_dst_ptr[out_edge]",
                f"        post = dst // {n_receptor}",
                "        post_event = spike_ptr[post]",
                "        arrival = active_arrival",
            )
        )
        body.extend(
            "        " + line
            for line in _parameter_loads(
                "tilelang", parameter_sizes, "canonical_edge", "update_mask", model
            )
        )
        body.extend(
            "        " + line
            for line in _tile_edge_update(
                "canonical_edge",
                model=model,
                histories=histories,
                n_edges=n_edges,
                state_domains=state_domains,
                n_delay=n_delay,
            )
        )
        body.append("        T.atomic_add(drive_ptr[dst], weight * arrival)")
        return body

    for segment_pass in range(segment_passes):
        if spike_list:
            lines.extend(
                (
                    "segment_id = 0",
                    "segment_valid = False",
                    "if list_valid:",
                    "    segment_id = src_seg_ptr_ptr[listed_source]",
                    "    segment_valid = segment_id < "
                    "src_seg_ptr_ptr[listed_source + 1]",
                )
            )
        else:
            lines.extend(
                (
                    f"segment_id = {index} + {segment_pass} * {grid_capacity}",
                    f"segment_valid = segment_id < {n_segments}",
                )
            )
        lines.extend(
            (
                "segment_arrival = 0.0",
                "if segment_valid:",
                "    segment_pre = seg_pre_ptr[segment_id]",
                "    segment_delay = seg_delay_ptr[segment_id]",
            )
        )
        lines.extend(
            "    " + line
            for line in _tile_arrival(
                "segment_pre", "segment_delay", n, n_delay, output="segment_arrival"
            )
        )
        candidate = (
            f"segment_id if segment_valid and segment_arrival != 0.0 else {n_segments}"
        )
        lines.extend(
            (
                f"remaining_segment = {candidate}",
                *select("segment", "remaining_segment", n_segments, outgoing_body()),
            )
        )

    def incoming_body():
        lane_expr = f"lane % {width}"
        edge_step = width * row_partitions
        body = ["post_event = spike_ptr[active_post]"]
        body.extend(
            (
                f"for receptor in T.serial(0, {n_receptor}):",
                f"    row_index = active_post * {n_receptor} + receptor",
                "    row_start = in_crow_ptr[row_index]",
                "    row_end = in_crow_ptr[row_index + 1]",
                "    for in_edge_offset in T.serial("
                + (f"event_partition * {width}" if row_partitions > 1 else "0")
                + ", row_end - row_start, "
                f"{edge_step}):",
                f"        in_edge = row_start + in_edge_offset + {lane_expr}",
                "        in_edge_valid = in_edge < row_end",
                "        if in_edge_valid:",
                "            canonical_edge = in_edge_id_ptr[in_edge]",
                "            source_delay = in_col_ptr[in_edge]",
                f"            pre = source_delay // {max(n_delay, 1)}",
                f"            delay = source_delay % {max(n_delay, 1)}",
                "            incoming_arrival = 1.0",
            )
        )
        body.extend(
            "            " + line
            for line in _tile_arrival(
                "pre", "delay", n, n_delay, output="incoming_arrival"
            )
        )
        body.extend(
            (
                "            if incoming_arrival == 0.0:",
                "                arrival = incoming_arrival",
            )
        )
        body.extend(
            "                " + line
            for line in _parameter_loads(
                "tilelang", parameter_sizes, "canonical_edge", "update_mask", model
            )
        )
        body.extend(
            "                " + line
            for line in _tile_edge_update(
                "canonical_edge",
                model=model,
                ltp=True,
                ltd=False,
                histories=histories,
                n_edges=n_edges,
                state_domains=state_domains,
                segment=None,
                post="active_post",
                n_delay=n_delay,
            )
        )
        return body

    lines.extend(
        (
            "post_active = False",
            "if list_valid:" if spike_list else "if valid_index:",
            "    post_active = True"
            if spike_list
            else "    post_active = neuron_output != 0.0",
        )
    )
    lines.extend(
        (
            f"remaining_post = {'listed_source' if spike_list else index} "
            f"if post_active else {n}",
            *select("post", "remaining_post", n, incoming_body()),
        )
    )
    return lines


def _post_propagation(
    backend,
    n,
    n_edges,
    parameter_sizes,
    history_names,
    clip_initial,
    model,
    *,
    inactive_only=False,
    index="index",
):
    """Clip once and materialize only explicitly requested edge histories."""
    template = get_plasticity_template(model)
    sizes = dict(zip(template.parameters, parameter_sizes, strict=True))
    triton = backend == "triton"
    grid_capacity = ((int(n) + 127) // 128) * 128 if triton else int(n)
    edge_passes = (int(n_edges) + grid_capacity - 1) // grid_capacity
    loop = (
        f"for edge_pass in tl.range(0, {edge_passes}):"
        if triton
        else f"for edge_pass in T.serial(0, {edge_passes}):"
    )
    edge_lines = [f"edge = {index} + edge_pass * {grid_capacity}"]
    edge_lines.append(
        f"edge_mask = edge < {n_edges}" if triton else f"if edge < {n_edges}:"
    )
    indent = "    " if triton else "        "
    if inactive_only:
        if triton:
            edge_lines.extend(
                (
                    "edge_pre = tl.load(edge_pre_ptr + edge, mask=edge_mask, other=0)",
                    "edge_post = tl.load(edge_post_ptr + edge, "
                    "mask=edge_mask, other=0)",
                    "edge_arrival = tl.load(spike_ptr + edge_pre, "
                    "mask=edge_mask, other=0)",
                    "edge_event = tl.load(spike_ptr + edge_post, "
                    "mask=edge_mask, other=0)",
                    "edge_mask = edge_mask & (edge_arrival == 0.0) "
                    "& (edge_event == 0.0)",
                )
            )
        else:
            edge_lines.extend(
                (
                    "    edge_pre = edge_pre_ptr[edge]",
                    "    edge_post = edge_post_ptr[edge]",
                    "    if spike_ptr[edge_pre] == 0.0 "
                    "and spike_ptr[edge_post] == 0.0:",
                )
            )
            indent = "            "

    def load(name, *, parameter=False):
        pointer = "stdp_" + name if parameter else name
        index = "0" if parameter and sizes[name] == 1 else "edge"
        if triton and index == "0":
            return f"{name} = tl.load({pointer}_ptr)"
        return (
            f"{name} = tl.load({pointer}_ptr + {index}, mask=edge_mask, other=0)"
            if triton
            else f"{name} = {pointer}_ptr[{index}]"
        )

    def store(name, value, *, history=False):
        pointer = name + "_history" if history else name
        index = "record_index" if history else "edge"
        return (
            f"tl.store({pointer}_ptr + {index}, {value}, mask=edge_mask)"
            if triton
            else f"{pointer}_ptr[{index}] = {value}"
        )

    def edge_pass(body):
        return [
            loop,
            *("    " + line for line in edge_lines),
            *(indent + line for line in body),
        ]

    lines = []
    if clip_initial:
        body = [
            load("weight"),
            load("w_min", parameter=True),
            load("w_max", parameter=True),
            "clipped = maximum(w_min, minimum(w_max, weight))",
            store("weight", "clipped"),
        ]
        lines.extend(
            ["if stdp_step == 0:", *("    " + line for line in edge_pass(body))]
        )
    if history_names:
        body = [f"record_index = stdp_step * {n_edges} + edge"]
        trace_times = dict(
            zip(template.states, template.events.trace_times, strict=True)
        )
        if any(name in trace_times for name in history_names):
            body.extend(
                [load("trace_tick"), "elapsed_dt = (tick - trace_tick) * stdp_dt"]
            )
        for name in history_names:
            body.append(load(name))
            if name == "weight":
                value = name
            else:
                tau = trace_times[name]
                body.extend(
                    [
                        load(tau, parameter=True),
                        f"record_{name} = {name} * exp(-elapsed_dt / {tau})",
                    ]
                )
                value = "record_" + name
            body.append(store(name, value, history=True))
        lines.extend(edge_pass(body))
    return lines


@functools.lru_cache(maxsize=128)
def prepare_stdp_phases(
    backend,
    model,
    n_neuron,
    n_receptor,
    n_delay,
    n_segments,
    n_edge,
    parameter_sizes,
    history_names=(),
    *,
    clip_initial=True,
    parallelism="auto",
    spike_selection="scan",
    edge_records=False,
    row_partitions=1,
    index="index",
    state_domains=(),
    state_destinations=(),
):
    """Build D²CSR propagation/STDP source for one fused timestep.

    Prepare connection traversal once. HELION selects warp cooperation
    below 32 edges per neuron and block cooperation otherwise; thread
    traversal is available for controlled comparisons. This policy
    applies to event rows, not continuous rules that must advance every
    edge on silent steps.
    """
    if backend not in ("triton", "tilelang"):
        raise ValueError("HELION STDP requires Triton or TileLang")
    template = get_plasticity_template(model, state_domains)
    if template.events is None:
        raise NotImplementedError(
            f"HELION's event traversal cannot execute continuous rule {model!r}"
        )
    shared = any(domain != "edge" for _, domain in template.state_domains)
    if shared and state_destinations:
        destinations = dict(state_destinations)
        if (
            len(destinations) != len(state_destinations)
            or set(destinations) != set(template.states)
            or len(set(destinations.values())) != len(destinations)
            or set(destinations.values()) & {*template.states, "weight"}
        ):
            raise ValueError(
                "Shared trace destinations must map every state to a distinct "
                "buffer outside the current states and weight"
            )
    if shared and (
        (edge_records and not state_destinations)
        or any(domain == "edge" for _, domain in template.state_domains)
        or template.stages[0].domain != "edge"
        or tuple(name for name, _ in template.stages[0].outputs) != ("weight",)
        or any(stage.domain not in ("segment", "post") for stage in template.stages[1:])
    ):
        raise NotImplementedError(
            "Shared fused event traces require domain-owned states and the "
            "cooperative timestep; mixed edge/shared states are not implemented"
        )
    if len(parameter_sizes) != len(template.parameters):
        raise ValueError("HELION plasticity parameters disagree with its template")
    if parallelism == "auto":
        parallelism = "warp" if n_edge < 32 * n_neuron else "block"
    if parallelism not in ("thread", "warp", "block"):
        raise ValueError(f"Unknown HELION connection parallelism: {parallelism!r}")
    n_delay = max(int(n_delay), 1)
    if row_partitions < 1 or (row_partitions > 1 and not edge_records):
        raise ValueError("Row partitions require the ordinary event-owner schedule")
    if edge_records and n_delay != 1:
        raise NotImplementedError(
            "Event-owned history recording currently requires zero-delay connections"
        )
    if spike_selection not in ("scan", "spike_list"):
        raise ValueError(f"Unknown spike selection: {spike_selection!r}")
    if spike_selection == "spike_list" and n_delay != 1:
        raise NotImplementedError(
            "Fused spike-list STDP requires a delayed-queue history to select "
            "source/delay segments; currently only zero-delay connections are supported"
        )
    if backend == "triton":
        propagation = _triton_propagation(
            n_neuron,
            n_receptor,
            n_delay,
            n_segments,
            n_edge,
            parameter_sizes,
            model,
            parallelism=parallelism,
            spike_list=spike_selection == "spike_list",
            histories=tuple(history_names) if edge_records else (),
            row_partitions=row_partitions,
            index=index,
            state_domains=state_domains,
        )
    else:
        propagation = _tile_propagation(
            n_neuron,
            n_receptor,
            n_delay,
            n_segments,
            parameter_sizes,
            model,
            parallelism=parallelism,
            spike_list=spike_selection == "spike_list",
            histories=tuple(history_names) if edge_records else (),
            n_edges=n_edge,
            row_partitions=row_partitions,
            index=index,
            state_domains=state_domains,
        )
    post = _post_propagation(
        backend,
        n_neuron,
        n_edge,
        parameter_sizes,
        tuple(name for name in history_names if not shared or name == "weight"),
        clip_initial,
        model,
        inactive_only=edge_records,
        index=index,
    )
    if shared:
        post.extend(
            _shared_trace_commit(
                backend,
                template,
                n_neuron,
                n_segments,
                n_delay,
                parameter_sizes,
                history_names,
                destinations=state_destinations,
                index=index,
            )
        )
    local_names = {
        "shared_segment": "int64",
        "shared_index": "int64",
        "shared_pre": "int64",
        "shared_delay": "int64",
        "shared_pass": "int64",
        "shared_valid": "bool",
        "shared_past": "bool",
        "tick": "int64",
        "stdp_step": "int64",
        "trace_tick": "int64",
        "elapsed": "int64",
        "segment_id": "int64",
        "segment_valid": "bool",
        "segment_pre": "int64",
        "segment_delay": "int64",
        "remaining_segment": "int64",
        "active_segment": "int64",
        "active_pre": "int64",
        "active_delay": "int64",
        "warp_lane": "int64",
        "warp_candidate": "int64",
        "lane": "int64",
        "valid_index": "bool",
        "row_start": "int64",
        "row_end": "int64",
        "edge": "int64",
        "out_edge": "int64",
        "out_edge_valid": "bool",
        "in_edge": "int64",
        "in_edge_valid": "bool",
        "canonical_edge": "int64",
        "dst": "int64",
        "post": "int64",
        "source_delay": "int64",
        "pre": "int64",
        "delay": "int64",
        "receptor": "int64",
        "row_index": "int64",
        "record_index": "int64",
        "post_active": "bool",
        "remaining_post": "int64",
        "active_post": "int64",
        "list_count": "int32",
        "list_valid": "bool",
        "listed_source": "int64",
        "segment_end": "int64",
        "edge_pre": "int64",
        "edge_post": "int64",
    }
    return HelionPlasticityPhases(
        tuple(propagation),
        tuple(post),
        tuple(sorted(local_names.items())),
        mutated_buffers=(
            "weight",
            *(dict(state_destinations).get(name, name) for name in template.states),
            *(() if shared else ("trace_tick",)),
        )
        + tuple(name + "_history" for name in history_names),
    )
