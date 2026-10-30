"""Replay functional recurrent stages with independent autograd tapes."""

from collections import OrderedDict
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

import torch

from ...utils.kernel import tensor_layout


_active: ContextVar = ContextVar("btorch_recurrent_training_graph", default=None)


def active_graphs():
    """Return the native training capture selected by the calling module."""
    return _active.get()


@dataclass
class _Stage:
    graph: torch.cuda.CUDAGraph
    inputs: list[torch.Tensor]
    outputs: list[torch.Tensor | None]


class RecurrentGraphs:
    """Own bounded forward/reverse graphs, never outstanding training tapes.

    Static states, drives and cotangents are staging buffers. Every
    result is copied out before another replay can overwrite it.
    Autograd therefore owns ordinary invocation-private tensors,
    including inter-chunk state adjoints. Parameters/topology stay bound
    by address and are read anew on replay.
    """

    def __init__(self, steps: int, warmup: int):
        self.steps = steps
        self.warmup = warmup
        self.stages = OrderedDict()
        self.version = ()
        self.n_captures = 0

    @contextmanager
    def activate(self):
        token = _active.set(self)
        try:
            yield
        finally:
            _active.reset(token)

    def run(self, function, arguments, *, fixed=()):
        """Replay one functional operator, binding fresh dynamic operands."""
        with torch.cuda.device(arguments[0].device):
            return self._run(function, arguments, fixed)

    def _run(self, function, arguments, fixed):
        # Operator arguments consist only of tensors, flat tensor lists, and
        # immutable metadata. Keep this packing local to the recurrent ABI.
        tensors = []
        layout = []
        metadata = []
        for index, value in enumerate(arguments):
            if isinstance(value, torch.Tensor):
                layout.append((len(tensors), 1, False))
                tensors.append(value)
                metadata.append(None)
            elif isinstance(value, list) and all(
                isinstance(t, torch.Tensor) for t in value
            ):
                layout.append((len(tensors), len(value), True))
                tensors.extend(value)
                metadata.append(None)
            else:
                layout.append(None)
                metadata.append(tuple(value) if isinstance(value, list) else value)
        bound = {
            position
            for index in fixed
            for position in range(layout[index][0], sum(layout[index][:2]))
            # AMP casts and differentiable weight transforms are invocation
            # values, not persistent parameters. Refresh their staging buffers
            # instead of capturing a temporary allocation's address.
            if tensors[position].is_leaf
        }
        key = (
            function,
            self.version,
            (
                torch.is_autocast_enabled(tensors[0].device.type),
                torch.get_autocast_dtype(tensors[0].device.type),
            ),
            torch.cuda.current_stream(tensors[0].device).cuda_stream,
            tuple(metadata),
            tuple(layout),
            tuple(
                (tensor_layout(t), t.data_ptr() if i in bound else None)
                for i, t in enumerate(tensors)
            ),
        )
        stage = self.stages.get(key)
        if stage is None:
            inputs = [
                t if i in bound else torch.empty_like(t) for i, t in enumerate(tensors)
            ]
            for i, (target, source) in enumerate(zip(inputs, tensors)):
                if i not in bound:
                    target.copy_(source)
            captured_arguments = []
            for value, item in zip(arguments, layout):
                if item is None:
                    captured_arguments.append(value)
                else:
                    offset, count, sequence = item
                    captured_arguments.append(
                        inputs[offset : offset + count] if sequence else inputs[offset]
                    )
            current = torch.cuda.current_stream(tensors[0].device)
            side = torch.cuda.Stream(device=tensors[0].device)
            side.wait_stream(current)
            with torch.cuda.stream(side):
                for _ in range(max(1, self.warmup)):
                    function(*captured_arguments)
            current.wait_stream(side)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                outputs = function(*captured_arguments)
            self.n_captures += 1
            stage = _Stage(graph, inputs, outputs)
            self.stages[key] = stage
            # Replaced layouts/structures must not accumulate graph pools.
            while len(self.stages) > 4:
                self.stages.popitem(last=False)
        else:
            self.stages.move_to_end(key)
            for i, (target, source) in enumerate(zip(stage.inputs, tensors)):
                if i not in bound:
                    target.copy_(source)
        stage.graph.replay()
        # The custom-op schema is functional: no result may alias the replay
        # pool, another output, or a caller-owned tensor. Saved-tensor hooks see
        # these copies normally, including checkpointing and CPU offload.
        return [value.clone() if value is not None else None for value in stage.outputs]
