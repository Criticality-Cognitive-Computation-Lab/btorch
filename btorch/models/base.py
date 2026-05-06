from abc import abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from numbers import Number
from typing import Any

import numpy as np
import torch
from jaxtyping import Float
from torch import Tensor

from ..types import TensorLike
from .shape import expand_leading_dims
from .surrogate import Sigmoid


class StepModule:
    """Mixin that provides step_mode dispatch (ported from spikingjelly).

    Subclasses must implement ``single_step_forward``. Optionally override
    ``multi_step_forward`` for a more efficient batched implementation.
    """

    def supported_step_mode(self) -> tuple[str, ...]:
        return ("s", "m")

    @property
    def step_mode(self) -> str:
        return self._step_mode

    @step_mode.setter
    def step_mode(self, value: str):
        if value not in self.supported_step_mode():
            raise ValueError(
                f"step_mode can only be {self.supported_step_mode()}, "
                f'but got "{value}"!'
            )
        self._step_mode = value

    def single_step_forward(self, x: torch.Tensor, *args, **kwargs):
        raise NotImplementedError

    def multi_step_forward(self, x_seq: torch.Tensor, *args, **kwargs):
        raise NotImplementedError

    def forward(self, *args, **kwargs):
        if self.step_mode == "s":
            return self.single_step_forward(*args, **kwargs)
        elif self.step_mode == "m":
            return self.multi_step_forward(*args, **kwargs)
        else:
            raise ValueError(self.step_mode)


def is_broadcastable(shape_from, shape_to):
    try:
        _ = torch.empty(shape_from) + torch.empty(shape_to)
        return True
    except RuntimeError:
        return False


@dataclass
class PreparedParam:
    """Intermediate parameter definition before module registration."""

    name: str
    value: torch.Tensor
    sizes: tuple[int, ...]
    is_trainable: bool
    trainable_shape: str
    allow_compact: bool


class ParamBufferMixin(torch.nn.Module):
    """Standard parameter/buffer definition and load-shape behavior.

    This mixin allows defining parameters/buffers that can be stored in their
    minimal broadcastable form (to save memory) or as full arrays. Supports:
    - easy trainable definition via one argument: `trainable_param`
    - optional trainable shape policy (`trainable_shape="scalar"|"full"|"auto"`)
    - extend `load_state_dict` for loading non-uniform full tensors
        on uniform scalar buffer
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # name -> "auto" | "scalar" | "full"
        self._param_shape_mode: dict[str, str] = {}
        # name -> whether compact scalar save/load is allowed
        self._param_allow_compact: dict[str, bool] = {}

    def _resolve_is_trainable(
        self,
        name: str,
        trainable_param: bool | set[str] | None = None,
    ) -> bool:
        if isinstance(trainable_param, bool):
            return trainable_param
        if isinstance(trainable_param, set):
            return name in trainable_param
        if hasattr(self, "trainable_param"):
            return name in getattr(self, "trainable_param")
        return False

    @staticmethod
    def _resolve_sizes_spec_for_value(
        val, sizes_spec: tuple[int | None, ...]
    ) -> tuple[int, ...]:
        none_axes = [idx for idx, dim in enumerate(sizes_spec) if dim is None]
        if len(none_axes) > 1:
            raise ValueError(
                f"At most one inferred dimension is supported, got sizes={sizes_spec}."
            )

        if len(none_axes) == 0:
            resolved: list[int] = []
            for dim in sizes_spec:
                if dim is None:
                    raise ValueError(
                        f"Unexpected unresolved dimension in sizes={sizes_spec}."
                    )
                resolved.append(int(dim))
            return tuple(resolved)

        infer_axis = none_axes[0]
        inferred_dim = 1
        val_tensor = torch.as_tensor(val)

        if val_tensor.ndim == len(sizes_spec):
            inferred_dim = int(val_tensor.shape[infer_axis])
        elif infer_axis == len(sizes_spec) - 1 and val_tensor.ndim > infer_axis:
            inferred_dim = int(val_tensor.shape[-1])
        elif infer_axis == len(sizes_spec) - 1:
            known_prefix = tuple(int(dim) for dim in sizes_spec[:-1] if dim is not None)
            prefix_prod = int(np.prod(known_prefix)) if known_prefix else 1
            if (
                prefix_prod > 0
                and val_tensor.ndim == 1
                and val_tensor.numel() % prefix_prod == 0
            ):
                inferred_dim = max(1, int(val_tensor.numel() // prefix_prod))
            elif val_tensor.ndim > 0:
                inferred_dim = int(val_tensor.shape[-1])

        resolved_sizes: list[int] = []
        for dim in sizes_spec:
            if dim is None:
                resolved_sizes.append(inferred_dim)
            else:
                resolved_sizes.append(int(dim))
        return tuple(resolved_sizes)

    def def_param_resolve_sizes(
        self,
        *vals,
        sizes: tuple[int | None, ...] | None = None,
    ) -> tuple[int, ...]:
        """Resolve a concrete size tuple from one or more values.

        If ``sizes`` contains one ``None`` axis, this method infers that axis
        per value and returns the broadcast-compatible maximum across values.
        """
        if sizes is None:
            if not hasattr(self, "n_neuron"):
                raise ValueError("sizes is required when module has no n_neuron.")
            sizes = tuple(getattr(self, "n_neuron"))

        sizes_spec = tuple(sizes)
        if len(vals) == 0:
            return self._resolve_sizes_spec_for_value(0.0, sizes_spec)

        resolved_list = [
            self._resolve_sizes_spec_for_value(v, sizes_spec) for v in vals
        ]
        target = list(resolved_list[0])
        for resolved in resolved_list[1:]:
            if len(resolved) != len(target):
                raise ValueError("Resolved sizes must have the same rank.")
            for axis in range(len(target)):
                d0, d1 = target[axis], resolved[axis]
                dmax = max(d0, d1)
                if d0 not in (1, dmax) or d1 not in (1, dmax):
                    raise ValueError(
                        f"Incompatible resolved sizes at axis {axis}: "
                        f"{tuple(target)} vs {resolved}."
                    )
                target[axis] = dmax
        return tuple(target)

    def def_param_prepare(
        self,
        name: str,
        val,
        *,
        sizes: tuple[int | None, ...] | None = None,
        trainable_param: bool | set[str] | None = None,
        trainable_shape: str = "auto",
        normalize_to_sizes: bool = False,
        **kwargs,
    ) -> PreparedParam:
        """Build a parameter definition without registering it.

        Args:
            name: Attribute name.
            val: Initial value.
            sizes: Intended tensor shape. If None, uses ``self.n_neuron``.
            trainable_param: Trainable selector:
                - ``True``: trainable parameter
                - ``False``: buffer
                - ``set[str]``: trainable when ``name in set``
                - ``None``: fallback to ``self.trainable_param`` if present
            trainable_shape: Shape policy for trainable values:
                - ``"auto"``: keep provided shape
                - ``"scalar"``: require uniform value and store as scalar
                - ``"full"``: store as full tensor with ``sizes``
            normalize_to_sizes: If True, broadcast and materialize value to
                ``sizes`` (ignored for ``trainable_shape="scalar"``).
            **kwargs: Passed to :func:`torch.as_tensor`.

        Raises:
            ValueError: If the value shape is not broadcastable to ``sizes``.

        Returns:
            Prepared parameter metadata and tensor value.
        """
        if sizes is None:
            if not hasattr(self, "n_neuron"):
                raise ValueError("sizes is required when module has no n_neuron.")
            sizes = tuple(getattr(self, "n_neuron"))

        sizes_spec = tuple(sizes)

        is_trainable = self._resolve_is_trainable(name, trainable_param)

        if trainable_shape not in {"auto", "scalar", "full"}:
            raise ValueError(
                f"Invalid trainable_shape={trainable_shape!r}. "
                "Use 'auto', 'scalar', or 'full'."
            )

        sizes = self._resolve_sizes_spec_for_value(val, sizes_spec)
        val = torch.as_tensor(val, **kwargs)

        if val.ndim == 1 and val.numel() == int(np.prod(sizes)):
            val = val.reshape(sizes)

        if (
            len(sizes) > 1
            and val.ndim == len(sizes) - 1
            and tuple(val.shape) == sizes[:-1]
        ):
            val = val[..., None]
        elif (
            val.ndim == 1 and len(sizes) > 1 and val.numel() == int(np.prod(sizes[:-1]))
        ):
            val = val.reshape(sizes[:-1] + (1,))

        if not is_broadcastable(val.shape, sizes):
            raise ValueError(
                f"{name} shape {tuple(val.shape)} is not broadcastable to {sizes}."
            )

        # Keep compacting only for neuron-sized parameters. For explicitly
        # larger shapes (e.g., neuron + extra axes), preserve full shape.
        allow_compact = sizes == tuple(getattr(self, "n_neuron", sizes))

        if is_trainable and trainable_shape == "full" and val.shape != sizes:
            val = expand_leading_dims(val, sizes, match_full_shape=True)
        if is_trainable and trainable_shape == "scalar":
            if not self._is_uniform(val):
                raise ValueError(
                    f"{name} with trainable_shape='scalar' must be uniform, "
                    f"but got shape {tuple(val.shape)} with non-uniform values."
                )
            val = val.reshape(-1)[:1].reshape(())

        if normalize_to_sizes and trainable_shape != "scalar" and val.shape != sizes:
            val = torch.broadcast_to(val, sizes).clone()

        return PreparedParam(
            name=name,
            value=val,
            sizes=sizes,
            is_trainable=is_trainable,
            trainable_shape=trainable_shape,
            allow_compact=allow_compact,
        )

    def def_param_register(self, prepared: PreparedParam) -> None:
        """Register a parameter from :meth:`def_param_prepare`."""
        if hasattr(self, prepared.name):
            delattr(self, prepared.name)

        self._param_shape_mode[prepared.name] = (
            prepared.trainable_shape if prepared.is_trainable else "auto"
        )
        self._param_allow_compact[prepared.name] = prepared.allow_compact

        if prepared.is_trainable:
            self.register_parameter(prepared.name, torch.nn.Parameter(prepared.value))
        else:
            self.register_buffer(prepared.name, prepared.value, persistent=True)

    def def_param(
        self,
        name: str,
        val,
        *,
        sizes: tuple[int | None, ...] | None = None,
        trainable_param: bool | set[str] | None = None,
        trainable_shape: str = "auto",
        normalize_to_sizes: bool = False,
        **kwargs,
    ):
        """Define a trainable parameter or persistent buffer.

        Convenience wrapper equivalent to:

        1. :meth:`def_param_prepare`
        2. :meth:`def_param_register`
        """
        prepared = self.def_param_prepare(
            name,
            val,
            sizes=sizes,
            trainable_param=trainable_param,
            trainable_shape=trainable_shape,
            normalize_to_sizes=normalize_to_sizes,
            **kwargs,
        )
        self.def_param_register(prepared)

    @staticmethod
    def _is_uniform(tensor: torch.Tensor, atol: float = 1e-6) -> bool:
        if tensor.numel() <= 1:
            return True
        if tensor.dtype.is_floating_point:
            return bool(torch.allclose(tensor, tensor.reshape(-1)[0], atol=atol))
        return bool((tensor == tensor.reshape(-1)[0]).all())

    def _replace_registered_tensor(self, name: str, value: torch.Tensor) -> None:
        current = getattr(self, name)
        value = value.to(device=current.device, dtype=current.dtype)
        if name in self._parameters:
            requires_grad = bool(self._parameters[name].requires_grad)
            delattr(self, name)
            self.register_parameter(
                name,
                torch.nn.Parameter(value, requires_grad=requires_grad),
            )
            return
        if name in self._buffers:
            persistent = name not in self._non_persistent_buffers_set
            delattr(self, name)
            self.register_buffer(name, value, persistent=persistent)

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        super()._save_to_state_dict(destination, prefix, keep_vars)
        for name, mode in self._param_shape_mode.items():
            if mode == "full":
                continue
            if not self._param_allow_compact.get(name, True):
                continue
            key = prefix + name
            if key not in destination:
                continue
            value = destination[key]
            if torch.is_tensor(value) and value.numel() > 1 and self._is_uniform(value):
                destination[key] = value.reshape(-1)[:1].reshape(())

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        for name, mode in self._param_shape_mode.items():
            key = prefix + name
            if key not in state_dict or not hasattr(self, name):
                continue

            loaded = state_dict[key]
            if not torch.is_tensor(loaded):
                continue

            current = getattr(self, name)
            current_shape = tuple(current.shape)
            loaded_shape = tuple(loaded.shape)

            # Parameters with trailing dimensions should preserve their full
            # shape. We only broadcast incoming compact tensors to the current
            # shape but never collapse to scalar.
            if not self._param_allow_compact.get(name, True):
                if loaded_shape != current_shape and is_broadcastable(
                    loaded_shape, current_shape
                ):
                    state_dict[key] = torch.broadcast_to(loaded, current_shape).clone()
                elif loaded_shape != current_shape:
                    self._replace_registered_tensor(name, loaded.detach().clone())
                continue

            # Mode full: always keep full tensor shape.
            if mode == "full":
                if loaded_shape != current_shape and is_broadcastable(
                    loaded_shape, current_shape
                ):
                    state_dict[key] = torch.broadcast_to(loaded, current_shape).clone()
                elif loaded_shape != current_shape:
                    self._replace_registered_tensor(name, loaded.detach().clone())
                continue

            if mode == "scalar":
                if loaded.numel() == 1:
                    scalar = loaded.reshape(-1)[:1].reshape(())
                    state_dict[key] = scalar
                    if current_shape != ():
                        self._replace_registered_tensor(name, scalar)
                    continue

                # Non-scalar checkpoint for scalar mode:
                # - trainable parameter: bail out (avoid silent layout changes)
                # - non-trainable buffer: promote to loaded shape
                if name in self._parameters:
                    error_msgs.append(
                        f"{key}: received non-scalar checkpoint tensor for "
                        "trainable_shape='scalar' trainable parameter."
                    )
                    continue

                self._replace_registered_tensor(name, loaded.detach().clone())
                continue

            # Mode auto:
            # - uniform loaded value -> scalar
            # - non-uniform loaded value -> full tensor
            if self._is_uniform(loaded):
                scalar = loaded.reshape(-1)[:1].reshape(())
                state_dict[key] = scalar
                if current_shape != ():
                    self._replace_registered_tensor(name, scalar)
            elif loaded_shape != current_shape:
                self._replace_registered_tensor(name, loaded.detach().clone())

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )


def normalize_n_neuron(
    n_neuron: int | Sequence[int],
) -> tuple[tuple[int, ...], int]:
    if isinstance(n_neuron, int):
        n_neuron = (n_neuron,)
    else:
        n_neuron = tuple(n_neuron)
    if len(n_neuron) == 0:
        raise ValueError("n_neuron must contain at least one dimension.")
    size = int(np.prod(n_neuron))
    return n_neuron, size


def flatten_neuron(
    x: torch.Tensor, n_neuron: tuple[int, ...], size: int
) -> tuple[torch.Tensor, tuple[int, ...]]:
    """Flatten trailing neuron dimensions for linear transformation.

    Args:
        x: Input tensor with trailing neuron dimensions.
        n_neuron: Neuron dimension sizes.
        size: Flattened size (product of n_neuron).

    Returns:
        Tuple of (flattened_tensor, leading_shape).
    """
    if len(n_neuron) == 1:
        return x, x.shape[:-1]
    leading = x.shape[: -len(n_neuron)]
    return x.reshape(*leading, size), leading


def unflatten_neuron(
    x: torch.Tensor, leading_shape: tuple[int, ...], n_neuron: tuple[int, ...]
) -> torch.Tensor:
    """Restore neuron dimensions after linear transformation.

    Args:
        x: Flattened tensor.
        leading_shape: Leading batch dimensions.
        n_neuron: Neuron dimension sizes.

    Returns:
        Tensor with restored neuron dimensions.
    """
    if len(n_neuron) == 1:
        return x
    return x.reshape(*leading_shape, *n_neuron)


ResetValueType = Callable | np.ndarray | torch.Tensor | None


@dataclass
class ResetValue:
    # None inits to torch.empty
    # never in-place modify this
    value: ResetValueType
    # sizes of the memory to be reset
    sizes: tuple[int, ...]
    # dtype of the memory to be reset
    # don't set unless you want to force this dtype (higer priority over reset(dtype))
    # typically used for boolean memories
    dtype: torch.dtype | None = None
    # persistent buffer
    persistent: bool | None = None
    # mark self.value stores batch axis additionally.
    # intended for e.g. different init values for each batch axis
    has_batch: bool = False
    # use case for this?
    # extra_kwargs: dict[str, Any] = field(default_factory=dict)


def _validate_sizes(reset_val: ResetValue | dict) -> None:
    if isinstance(reset_val, ResetValue):
        reset_val = reset_val.__dict__

    assert "sizes" in reset_val
    if reset_val["value"] is None or isinstance(reset_val["value"], Callable):
        return
    assert is_broadcastable(reset_val["value"].shape, reset_val["sizes"])


def _validate_has_batch(reset_val: ResetValue | dict) -> None:
    """Validate that the value tensor shape is consistent with the has_batch
    flag."""
    if isinstance(reset_val, ResetValue):
        reset_val = reset_val.__dict__

    value = reset_val["value"]
    has_batch = reset_val["has_batch"]
    sizes = reset_val["sizes"]
    sizes_str = ", ".join(str(s) for s in sizes)

    # Callable values are always valid
    if isinstance(value, Callable):
        return

    ndim_sizes = len(sizes)

    if has_batch:
        # With batch: must have tensor with extra dimension(s)
        if value is None:
            raise ValueError(
                f"has_batch=True requires an array value, but got None. "
                f"Expected (*batch_dims, {sizes_str})\n"
            )

        ndim_value = len(value.shape)
        if ndim_value <= ndim_sizes:
            raise ValueError(
                f"has_batch=True but array lacks batch dimensions. "
                f"Expected (*batch_dims, {sizes_str}) but got {value.shape}\n"
            )

    else:
        # Without batch: tensor should not have extra dimensions
        if value is not None:
            ndim_value = len(value.shape)
            if ndim_value > ndim_sizes:
                raise ValueError(
                    f"has_batch=False but array has too many dimensions."
                    f"Expected {sizes} but got {value.shape}\n"
                )


def _memory_var(
    reset_val: ResetValue,
    batch_size: tuple[int, ...] | int | None = None,
    **format_args,
):
    """Helper to initialize or reset memory var."""
    sizes = reset_val.sizes
    if batch_size is not None:
        if isinstance(batch_size, int):
            batch_size = (batch_size,)
        sizes = batch_size + sizes

    if reset_val.dtype is not None:
        format_args["dtype"] = reset_val.dtype

    if isinstance(reset_val.value, Callable):
        # sizes now contain batch axis if specified
        # callable can determine whether batch axis exists originally, from batch_size
        v = torch.as_tensor(reset_val.value(sizes, batch_size=batch_size)).to(
            **format_args
        )
        assert is_broadcastable(v.shape, sizes)
    elif reset_val.value is None:
        v = torch.empty(sizes, **format_args)
    else:
        # avoid accidentally carrying grad from old v
        v = torch.as_tensor(reset_val.value, **format_args).detach().clone()
    if v.shape != sizes:
        v = expand_leading_dims(v, sizes, match_full_shape=True, view=False)
    return v


class MemoryModule(StepModule, torch.nn.Module):
    """Base class for all stateful modules with managed memory buffers.

    MemoryModule provides infrastructure for managing stateful tensors
    (memories) in neuromorphic models. Unlike SpikingJelly's MemoryModule,
    this implementation:

    1. Stores all memories as torch.Tensor buffers (enables ONNX export)
    2. Does not support list/tuple memories (override reset/init for history)
    3. Uses fixed memory sizes, with variable batch size

    Memories are registered via register_memory() and automatically
    initialized/reset via init_state() and reset(). Each memory has a
    ResetValue configuration controlling its initialization behavior.

    Example:
        >>> class MyNeuron(MemoryModule):
        ...     def __init__(self, n_neuron):
        ...         super().__init__()
        ...         self.register_memory("v", 0.0, n_neuron)
        ...
        ...     def forward(self, x):
        ...         self.v = self.v + x  # simple integration
        ...         return self.v
        >>>
        >>> neuron = MyNeuron(10)
        >>> neuron.init_state(batch_size=2)  # init with batch dim
        >>> out = neuron(torch.randn(2, 10))
    """

    @property
    def supported_backends(self) -> tuple[str, ...]:
        return ("torch",)

    @property
    def backend(self) -> str:
        return self._backend

    @backend.setter
    def backend(self, value: str):
        if value not in self.supported_backends:
            raise NotImplementedError(
                f"{value} is not a supported backend of {self._get_name()}!"
            )
        self._backend = value

    @abstractmethod
    def single_step_forward(self, x: torch.Tensor, *args, **kwargs):
        pass

    def multi_step_forward(self, x_seq: torch.Tensor, *args, **kwargs):
        T = x_seq.shape[0]
        y_seq = []
        for t in range(T):
            y = self.single_step_forward(x_seq[t], *args, **kwargs)
            y_seq.append(y.unsqueeze(0))
        return torch.cat(y_seq, 0)

    def __init__(self):
        super().__init__()
        self._memories_rv: dict[str, ResetValue] = {}
        self._backend = "torch"
        self._step_mode = "s"

    @staticmethod
    def _format_repr_value(value: Any) -> str:
        if value is None:
            return "None"
        if isinstance(value, Callable):
            return "callable"
        if isinstance(value, torch.Tensor):
            if value.numel() == 1:
                return f"{value.item():.4g}"
            return f"shape={tuple(value.shape)}"
        if isinstance(value, np.ndarray):
            if value.size == 1:
                return f"{float(value.reshape(-1)[0]):.4g}"
            return f"shape={value.shape}"
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            array = np.asarray(value)
            if array.size == 1:
                return f"{float(array.reshape(-1)[0]):.4g}"
            return f"shape={array.shape}"
        return str(value)

    def _memories_repr(self) -> str:
        if not self._memories_rv:
            return ""
        entries = []
        for name, reset_val in self._memories_rv.items():
            extra = []
            if reset_val.dtype is not None:
                extra.append(str(reset_val.dtype).replace("torch.", ""))
            if reset_val.persistent is not None:
                extra.append("persistent" if reset_val.persistent else "non_persistent")
            if reset_val.has_batch:
                extra.append("has_batch")
            extra_str = f" [{', '.join(extra)}]" if extra else ""
            value_str = ""
            if reset_val.value is not None:
                value_str = f" init={self._format_repr_value(reset_val.value)}"
            entries.append(f"{name}={reset_val.sizes}{value_str}{extra_str}")
        return f"memories=({', '.join(entries)})"

    def extra_repr(self):
        return self._memories_repr()

    def register_memory(
        self,
        name: str,
        value: Any,
        sizes: int | Sequence[int],
        dtype=None,
        persistent=None,
    ):
        assert not hasattr(self, name), f"{name} has been set as a member variable!"
        if isinstance(sizes, int):
            sizes = (sizes,)
        else:
            sizes = tuple(sizes)
        if isinstance(value, Sequence):
            assert (
                len(value) != 0
            ), f"Memory cannot be empty list or sequence: got {value}"
            value = np.asarray(value)

        self.set_reset_value(
            name, value, sizes=sizes, dtype=dtype, persistent=persistent
        )

    def set_reset_value(
        self,
        name: str,
        value: ResetValueType | Number | ResetValue,
        *,
        strict: bool = True,
        **reset_kwargs,
    ):
        if isinstance(value, Number):
            value = np.asarray(value)
        if isinstance(value, ResetValueType):
            reset_kwargs["value"] = value
        else:
            if name not in self._memories_rv and len(reset_kwargs) == 0:
                if not strict:
                    self._memories_rv[name] = value
                    return
            reset_kwargs = {**value.__dict__, **reset_kwargs}

        if "value" not in reset_kwargs:
            raise ValueError(f"'value' is required for register_memory_rv('{name}')")
        if name not in self._memories_rv:
            if "sizes" not in reset_kwargs:
                raise ValueError(
                    f"'sizes' is required when registering new memory_rv '{name}'"
                )
            if "has_batch" not in reset_kwargs:
                reset_kwargs["has_batch"] = False

            _validate_sizes(reset_kwargs)
            _validate_has_batch(reset_kwargs)
            self._memories_rv[name] = ResetValue(**reset_kwargs)
            return

        if "sizes" in reset_kwargs:
            if strict and self._memories_rv[name].sizes != reset_kwargs["sizes"]:
                raise ValueError(
                    f"Memory_rv '{name}' sizes mismatch: "
                    f"existing={self._memories_rv[name].sizes}, "
                    f"new={reset_kwargs['sizes']}"
                )
        else:
            reset_kwargs["sizes"] = self._memories_rv[name].sizes

        _validate_sizes(reset_kwargs)

        if "has_batch" not in reset_kwargs:
            reset_kwargs["has_batch"] = self._memories_rv[name].has_batch

        _validate_has_batch(reset_kwargs)
        for k, v in self._memories_rv[name].__dict__.items():
            # if k not in ("has_batch", "sizes"):
            reset_kwargs.setdefault(k, v)

        self._memories_rv[name] = ResetValue(**reset_kwargs)

    @torch.no_grad()
    def init_state(
        self,
        batch_size=None,
        dtype=None,
        device=None,
        persistent=False,
        skip_mem_name: tuple[str, ...] = (),
    ):
        skip_mem_name_set = set(skip_mem_name)
        for key, reset_val in self._memories_rv.items():
            if key in skip_mem_name_set:
                continue
            dtype = reset_val.dtype or dtype
            v = _memory_var(reset_val, batch_size, dtype=dtype, device=device)
            persistent = reset_val.persistent or persistent
            self.register_buffer(key, v, persistent=persistent)

    def _batch_dim_detect(self, mem_name):
        sizes = self._memories_rv[mem_name].sizes
        buffer = self._buffers[mem_name]
        if (buffer is not None) and (buffer.shape != sizes):
            return buffer.shape[: -len(sizes)]
        return None

    def _batch_dim_exist(self, mem_name: str):
        return self._batch_dim_detect(mem_name) is not None

    @torch.no_grad()
    def reset(
        self,
        batch_size=None,
        dtype=None,
        device=None,
        skip_mem_name: tuple[str, ...] = (),
    ):
        skip_mem_name_set = set(skip_mem_name)
        for key, reset_val in self._memories_rv.items():
            if key in skip_mem_name_set:
                continue
            if reset_val.dtype is not None:
                dtype = reset_val.dtype
            buffer = self._buffers[key]
            format_args = {
                "device": device or buffer.device,
                "dtype": dtype or buffer.dtype,
            }
            if batch_size is None:
                batch_size = self._batch_dim_detect(key)

            v = _memory_var(reset_val, batch_size, **format_args)
            setattr(self, key, v)

    def __getattr__(self, name: str):
        return torch.nn.Module.__getattr__(self, name)

    def __setattr__(self, name: str, value) -> None:
        torch.nn.Module.__setattr__(self, name, value)

    def __delattr__(self, name):
        if name in self._memories_rv:
            del self._memories_rv[name]
        torch.nn.Module.__delattr__(self, name)

    def __dir__(self):
        return torch.nn.Module.__dir__(self)

    def memories(self):
        for name in self._memories_rv.keys():
            yield self._buffers[name]

    def named_memories(self):
        for name in self._memories_rv.keys():
            yield name, self._buffers[name]

    def detach(self):
        for key in self._memories_rv.keys():
            self._buffers[key].detach_()

    def _apply(self, fn):
        return torch.nn.Module._apply(self, fn)

    def _replicate_for_data_parallel(self):
        replica = torch.nn.Module._replicate_for_data_parallel(self)
        return replica

    @property
    def _memories(self):
        return {name: self._buffers[name] for name in self._memories_rv.keys()}

    @_memories.setter
    def _memories(self, value: dict):
        for k, v in value.items():
            assert k in self._memories_rv
            setattr(self, k, v)

    @property
    def memories_rv(self):
        return self._memories_rv

    def set_memories_rv(self, value: dict, strict: bool = False):
        for k, v in value.items():
            assert k in self._memories_rv
            self.set_reset_value(k, v, strict=strict)

    @memories_rv.setter
    def memories_rv(self, value: dict):
        for k, v in value.items():
            assert k in self._memories_rv
            self.set_reset_value(k, v, strict=False)


# TODO: pre_spike_v should be merged with v to avoid double memory consumption
# TODO: ODE integration method should be configurable
class BaseNode(ParamBufferMixin, MemoryModule):
    """Base class for differentiable spiking neurons.

    Implements the spiking neuron lifecycle: charge -> adapt -> fire -> reset.
    Subclasses implement neuronal_charge() and neuronal_adaptation().

    Args:
        n_neuron: Number of neurons (int or tuple).
        v_threshold: Firing threshold. Default: 1.0.
        v_reset: Reset voltage. Default: 0.0.
        trainable_param: Trainable parameter names. Default: ().
        surrogate_function: Surrogate for backprop. Default: Sigmoid().
        detach_reset: Detach reset signal. Default: False.
        hard_reset: Hard vs soft reset. Default: False.
        pre_spike_v: Store pre-spike voltage. Default: False.
        step_mode: "s" or "m". Default: "s".
        backend: Compute backend. Default: "torch".
        device: Tensor device. Default: None.
        dtype: Tensor dtype. Default: None.
    """

    n_neuron: tuple[int, ...]
    size: int
    v: torch.Tensor
    v_pre_spike: torch.Tensor
    v_threshold: torch.Tensor | torch.nn.Parameter
    v_reset: torch.Tensor | torch.nn.Parameter

    def __init__(
        self,
        n_neuron: int | Sequence[int],
        v_threshold: float | Float[TensorLike, " n_neuron"] = 1.0,
        v_reset: float | Float[TensorLike, " n_neuron"] = 0.0,
        trainable_param: set[str] = set(),
        surrogate_function: Callable = Sigmoid(),
        detach_reset: bool = False,
        hard_reset: bool = False,
        pre_spike_v: bool = False,
        step_mode="s",
        backend="torch",
        device=None,
        dtype=None,
    ):
        """Modified spikingjelly BaseNode.

        * :ref:`API in English <BaseNode.__init__-en>`

        This class is the base class of differentiable spiking neurons.
        """

        # override neuron.BaseNode's __init__ method to remove unnecessary checks
        # call neuron.BaseNode's parent MemoryModule directly
        super().__init__()

        self.n_neuron, self.size = normalize_n_neuron(n_neuron)
        self.register_memory("v", v_reset, self.n_neuron)
        self.pre_spike_v = pre_spike_v

        _factory_kwargs: dict[str, Any] = {"device": device, "dtype": dtype}
        if pre_spike_v:
            self.register_memory(
                "v_pre_spike", v_reset, self.n_neuron, persistent=False
            )

        self.trainable_param = set(trainable_param)
        self.def_param(
            "v_threshold",
            v_threshold,
            sizes=self.n_neuron,
            trainable_param=self.trainable_param,
            **_factory_kwargs,
        )
        self.def_param(
            "v_reset",
            v_reset,
            sizes=self.n_neuron,
            trainable_param=self.trainable_param,
            **_factory_kwargs,
        )

        self.detach_reset = detach_reset
        self.surrogate_function = surrogate_function
        self.hard_reset = hard_reset

        self.step_mode = step_mode
        self.backend = backend

    def extra_repr(self):
        parts = [
            f"n_neuron={self.n_neuron}",
            f"v_threshold={self._format_repr_value(self.v_threshold)}",
            f"v_reset={self._format_repr_value(self.v_reset)}",
            f"step_mode={self.step_mode}",
            f"backend={self.backend}",
            f"surrogate={self.surrogate_function.__class__.__name__}",
        ]
        if self.detach_reset:
            parts.append("detach_reset=True")
        if self.hard_reset:
            parts.append("hard_reset=True")
        if self.pre_spike_v:
            parts.append("pre_spike_v=True")
        mem_repr = super().extra_repr()
        if mem_repr:
            parts.append(mem_repr)
        return ", ".join(parts)

    @abstractmethod
    def neuronal_charge(self, x: torch.Tensor):
        """
         * :ref:`API in English <BaseNode.neuronal_charge-en>`

        .. _BaseNode.neuronal_charge-cn:

        定义神经元的充电差分方程。子类必须实现这个函数。

        * :ref:`中文API <BaseNode.neuronal_charge-cn>`

        .. _BaseNode.neuronal_charge-en:


        Define the charge difference equation.
        The sub-class must implement this function.
        """
        raise NotImplementedError

    def neuronal_fire(self):
        """
        * :ref:`API in English <BaseNode.neuronal_fire-en>`

        .. _BaseNode.neuronal_fire-cn:

        根据当前神经元的电压、阈值，计算输出脉冲。

        * :ref:`中文API <BaseNode.neuronal_fire-cn>`

        .. _BaseNode.neuronal_fire-en:


        Calculate out spikes of neurons by their current membrane potential
        and threshold voltage.
        """

        return self.surrogate_function(self.v - self.v_threshold)

    def neuronal_reset(self, spike):
        """
        * :ref:`API in English <BaseNode.neuronal_reset-en>`

        .. _BaseNode.neuronal_reset-cn:

        根据当前神经元释放的脉冲，对膜电位进行重置。

        * :ref:`中文API <BaseNode.neuronal_reset-cn>`

        .. _BaseNode.neuronal_reset-en:


        Reset the membrane potential according to neurons' output spikes.
        """
        if self.detach_reset:
            spike_d = spike.detach()
        else:
            spike_d = spike

        if self.pre_spike_v:
            self.v_pre_spike = self.v.clone()

        if self.hard_reset:
            # hard reset
            self.v = self.v - (self.v - self.v_reset) * spike_d
        else:
            # soft reset
            self.v = self.v - (self.v_threshold - self.v_reset) * spike_d

    def neuronal_adaptation(self):
        raise NotImplementedError()

    def single_step_forward(self, x: Float[Tensor, "*batch n_neuron"]):
        """
        * :ref:`API in English <BaseNode.single_step_forward-en>`
        """
        self.neuronal_charge(x)
        self.neuronal_adaptation()
        spike = self.neuronal_fire()
        self.neuronal_reset(spike)
        return spike

    def multi_step_forward(self, x_seq: Float[Tensor, "T *batch n_neuron"]):
        s_seq = []
        for t, x in enumerate(x_seq):
            s = self.single_step_forward(x)
            s_seq.append(s)

        return torch.stack(s_seq)
