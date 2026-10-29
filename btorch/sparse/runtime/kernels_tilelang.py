"""TileLang implementation of the shared sparse CUDA algorithms.

Select with ``btorch.sparse.runtime.use_backend("tilelang")``. Preparation and autograd
are shared with the Triton routes; compilation and launch use TileLang's public
JIT kernel interface exclusively.
"""

import importlib.util

import torch

from ...utils.kernel import storage_span
from . import csr_execution, csr_layout as csr
from .backend import KernelCache


def is_available():
    """Return whether TileLang and a CUDA device are available."""
    return (
        importlib.util.find_spec("tilelang") is not None and torch.cuda.is_available()
    )


class _Execution:
    """Bind TileLang stages to one invocation's shapes and owned views."""

    def __init__(self):
        self.stages = {}
        self.inputs = {}
        self.scalar_positions = {}

    def _stage(self, name, grid, args, const_names, consts, warps):
        del const_names
        # Static stage metadata is resolved once. Repeated submissions only
        # refresh changed tensor views and invoke the public compiled kernel.
        tensors = tuple(arg for arg in args if isinstance(arg, torch.Tensor))
        scalars = tuple(arg for arg in args if not isinstance(arg, torch.Tensor))
        metadata = (grid, consts, warps, scalars)
        stage = self.stages.get(name)
        if stage is None:
            from . import tilelang_csr

            factory = {
                "csr_matvec": tilelang_csr.matvec,
                "hub_combine": tilelang_csr.combine,
                "edge_grad": tilelang_csr.edge_grad,
                "listed_output": tilelang_csr.listed_output,
                "pack_spikes": tilelang_csr.pack_spikes,
                "push_packed": tilelang_csr.push_packed,
                "pack_tasks": tilelang_csr.pack_tasks,
                "direct_tasks": tilelang_csr.direct_tasks,
                "hash_tasks": tilelang_csr.hash_tasks,
                "push_dense": tilelang_csr.push_dense,
            }[name]
            views = [storage_span(arg) for arg in tensors]
            shapes = tuple(tuple(view.shape) for view in views)
            dtypes = tuple(str(arg.dtype).removeprefix("torch.") for arg in tensors)
            device_grid = (*grid, *(1 for _ in range(3 - len(grid))))
            from ...utils.autotune import tilelang_kernel

            # Queue consumers are bound before discovery has filled their
            # indices. Never benchmark them on uninitialized queue storage.
            tunable = name in (
                "csr_matvec",
                "hub_combine",
                "edge_grad",
                "listed_output",
            )
            positions = [
                i for i, arg in enumerate(args) if isinstance(arg, torch.Tensor)
            ]
            writes = tuple(
                positions.index(i) for i in csr_execution.stage_outputs(name)
            )
            kernel = tilelang_kernel(
                factory,
                (shapes, dtypes, scalars, consts, device_grid),
                views,
                writes,
                warps,
                work=csr_execution.stage_work(name, args, grid) if tunable else 0,
                overwrite=name != "csr_matvec" or not consts[6],
            )
            layouts = tuple(
                (arg.shape, arg.stride(), arg.dtype, arg.device) for arg in tensors
            )
            stage = (
                kernel,
                list(tensors),
                views,
                layouts,
                metadata,
                [arg.data_ptr() for arg in tensors],
            )
            self.stages[name] = stage
            self.inputs[name] = tuple(
                (position, positions.index(position))
                for position in csr_execution.stage_inputs(name, consts)
            )
            self.scalar_positions[name] = tuple(
                i for i, arg in enumerate(args) if not isinstance(arg, torch.Tensor)
            )
        if metadata != stage[4]:
            raise ValueError("Sparse launch metadata changed; prepare a new execution")
        return stage

    @staticmethod
    def _argument(stage, index, tensor):
        _, owners, views, layouts, _, pointers = stage
        pointer = tensor.data_ptr()
        if owners[index] is not tensor or pointers[index] != pointer:
            if (tensor.shape, tensor.stride(), tensor.dtype, tensor.device) != layouts[
                index
            ]:
                raise ValueError(
                    "Sparse tensor layout/dtype changed; prepare a new execution"
                )
            owners[index] = tensor
            views[index] = storage_span(tensor)
            pointers[index] = pointer

    def launch(self, name, grid, args, const_names, consts, warps):
        stage = self.stages.get(name)
        if stage is None:
            stage = self._stage(name, grid, args, const_names, consts, warps)
        elif (
            grid,
            consts,
            warps,
            tuple(args[i] for i in self.scalar_positions[name]),
        ) != stage[4]:
            raise ValueError("Sparse launch metadata changed; prepare a new execution")
        for position, index in self.inputs[name]:
            self._argument(stage, index, args[position])
        stage[0](*stage[2])

    def prepare(self, name, grid, args, const_names, consts, warps, *, changing):
        """Bind static ABI once; only declared source slots may change.

        Static tensor storage is invocation-owned and must not be
        rebound during execution. Mutable slots accept new addresses,
        including a same-object storage rebind, while preserving
        shape/stride/dtype/device.
        """
        stage = self._stage(name, grid, args, const_names, consts, warps)
        tensor_positions = {
            position: index
            for index, position in enumerate(
                position
                for position, arg in enumerate(args)
                if isinstance(arg, torch.Tensor)
            )
        }
        updates = tuple((position, tensor_positions[position]) for position in changing)
        update = self._argument
        kernel, views = stage[0], stage[2]

        def submit(arguments):
            for position, index in updates:
                update(stage, index, arguments[position])
            kernel(*views)

        return submit


def _prepare_matvec(cache, crow, col, values, x):
    return csr_execution._prepare_matvec(_Execution(), cache, crow, col, values, x)


def _prepare_backward(cache, *args, **kwargs):
    return csr_execution._prepare_backward(_Execution(), cache, *args, **kwargs)


def csr_matvec(crow, col, values, x):
    """Multiply destination-major CSR using the shared split-row algorithm."""
    return _prepare_matvec(KernelCache(), crow, col, values, x)(x)


def edge_grad(crow, col, grad, x, n_vb):
    """Reduce sample products for each stored edge without an edge/sample
    tape."""
    return csr_execution._prepare_edge_grad(
        _Execution(), KernelCache(), crow, col, grad, x, n_vb
    )(grad, x)


def csr_backward(
    crow, col, t_crow, t_col, t_perm, values, x, grad, need_values=True, need_x=True
):
    """Compute edge and input adjoints with shared gather and reduction
    stages."""
    return _prepare_backward(
        KernelCache(),
        crow,
        col,
        t_crow,
        t_col,
        t_perm,
        values,
        x,
        grad,
        need_values,
        need_x,
    )(x, grad)


def register(registry, priority=5):
    """Register TileLang sparse kernels without changing automatic defaults."""
    cache = registry.kernels

    def matvec(crow, col, values, x):
        return _prepare_matvec(cache, crow, col, values, x)(x)

    def gradient(crow, col, grad, x, n_vb):
        return csr_execution._prepare_edge_grad(
            _Execution(), cache, crow, col, grad, x, n_vb
        )(grad, x)

    def backward(*args):
        return _prepare_backward(cache, *args)(args[6], args[7])

    def prepare_forward(crow, col, values, x):
        return _prepare_matvec(cache, crow, col, values, x)

    def prepare_backward(*args, **kwargs):
        return _prepare_backward(cache, *args, **kwargs)

    def prepare_pull(crow, col, t_crow, t_col, t_perm):
        csr._prepare(cache, crow, col, t_crow, t_col, t_perm)

    def epoch(crow, col, t_crow, t_col, t_perm):
        from . import source_layout

        return max(
            csr._layout_epoch(cache, crow, col, t_crow, t_col, t_perm),
            source_layout.layout_epoch(cache, t_crow, t_col, t_perm),
        )

    def prepare_push(crow, col, t_crow, t_col, t_perm):
        from . import source_layout

        prepare_pull(crow, col, t_crow, t_col, t_perm)
        source_layout._layout(cache, t_crow, t_col, t_perm)

    def prepare_values(*args, **kwargs):
        from .spike_list import prepare_values

        return prepare_values(*args, **kwargs, cache=cache)

    def prepare_list(*args, **kwargs):
        return prepare_spike_list(*args, **kwargs, cache=cache)

    def push(*args, **kwargs):
        execute = prepare_list(*args, **kwargs)
        return execute(args[4])

    def prepare_tasks(t_crow, t_col, t_perm, values, batch_size, *, route_key=None):
        from . import task_layout

        state = task_layout.prepare(
            cache, t_crow, t_col, t_perm, batch_size, values.dtype, route_key=route_key
        )
        if state is None:
            return None, 0
        return task_layout.bind_values(cache, state, values), state.layout_version

    def packed_push(*args, **kwargs):
        return csr_execution.packed_push(_Execution(), *args, cache=cache, **kwargs)

    def prepare_dense(*args, **kwargs):
        return csr_execution.prepare_task_push(_Execution(), cache, *args, **kwargs)

    def dense_push(*args, **kwargs):
        return task_push(*args, **kwargs, cache=cache)

    for name, function, prepared in (
        ("csr_matvec", matvec, prepare_forward),
        ("edge_grad", gradient, None),
        ("csr_backward", backward, prepare_backward),
        ("prepare", prepare_push, None),
        ("prepare_pull", prepare_pull, None),
        ("prepare_push", prepare_push, None),
        ("prepare_spike_list_values", prepare_values, None),
        ("spike_push_list", push, prepare_list),
        ("prepare_task_values", prepare_tasks, None),
        ("spike_push_dense", dense_push, prepare_dense),
        ("spike_push", packed_push, None),
        ("layout_epoch", epoch, None),
    ):
        registry.register(
            name,
            "tilelang",
            function,
            device="cuda",
            priority=priority,
            available=is_available,
            prepare_execution=prepared,
        )


def _bind_spike_list(execution, x):
    from . import source_layout as push
    from .spike_list import SPIKE_BLOCK

    runtime = _Execution()
    queue = execution.spikes
    n_batch, n_in = x.shape
    n_out = execution.output.shape[-1]
    programs = execution.programs
    permute = execution.task_values is None
    execution.arguments = [
        execution.layout.crow,
        execution.layout.col,
        execution.layout.perm,
        execution.values if execution.task_values is None else execution.task_values,
        x,
        queue.indices,
        queue.cumulative,
        queue.counts,
        queue.output,
        SPIKE_BLOCK,
        n_in,
        n_out,
        n_batch,
        SPIKE_BLOCK.bit_length(),
        0,
        programs,
        (
            execution.values if execution.task_values is None else execution.task_values
        ).stride(0),
        x.stride(0),
        x.stride(1),
    ]

    pack_arguments = [
        x,
        queue.degree,
        queue.indices,
        queue.cumulative,
        queue.counts,
        queue.output,
    ]
    pack_grid = (-(-max(x.numel(), queue.output.numel()) // SPIKE_BLOCK),)
    pack_constants = (
        queue.n_sources,
        x.numel(),
        queue.output.numel(),
        SPIKE_BLOCK,
        x.stride(0),
        x.stride(1),
    )
    pack = None
    consume = runtime.prepare(
        "push_packed",
        (programs * queue.counts.numel(),),
        execution.arguments,
        (),
        (push.EDGE_BLOCK, permute, SPIKE_BLOCK),
        push.BLOCK_WARPS,
        changing=(4,),
    )

    def emit(input):
        nonlocal pack
        # Direct neuron producers never submit dense packing. Bind that
        # separate operation only on its first use, as the Triton packer does.
        if pack is None:
            pack = runtime.prepare(
                "pack_spikes",
                pack_grid,
                pack_arguments,
                (),
                pack_constants,
                4,
                changing=(0,),
            )
        pack_arguments[0] = input
        pack(pack_arguments)

    execution.pack = emit
    execution.launch = consume


def prepare_spike_list(*args, **kwargs):
    """Prepare the same local queues and persistent fanout as Triton push."""
    from .spike_list import prepare_spike_list as prepare

    return prepare(*args, **kwargs, bind=_bind_spike_list)


def task_push(
    t_crow,
    t_col,
    t_perm,
    values,
    x,
    n_out,
    *,
    task_values=None,
    require_task=False,
    cache,
):
    """Consume the same direct/hash task queues as Triton's dense-source
    route."""
    from . import task_layout

    binding = (
        None
        if task_values is None
        else task_layout.state_for_values(cache, task_values, t_crow, t_col, t_perm)
    )
    if binding is None:
        if require_task:
            raise RuntimeError("TileLang task push requires prepared task-order values")
        from . import source_layout

        layout = source_layout._layout(cache, t_crow, t_col, t_perm)
        if layout is None:
            raise RuntimeError("Unsupported source-major sparse layout")
        return csr_execution._launch_dense(
            _Execution(),
            cache,
            layout,
            values,
            x,
            n_out,
            True,
        )
    return csr_execution.run(
        _Execution(),
        binding.task_state,
        task_values,
        x,
        n_out,
        cache,
        workspace=binding.workspace,
    )
