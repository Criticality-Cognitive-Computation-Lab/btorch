from abc import abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from numbers import Number
from typing import Any

import numpy as np
import torch
from jaxtyping import Float
from spikingjelly.activation_based import base
from torch import Tensor

from ..types import TensorLike
from .shape import expand_leading_dims
from .surrogate import Sigmoid


def is_broadcastable(shape_from, shape_to):
    try:
        _ = torch.empty(shape_from) + torch.empty(shape_to)
        return True
    except RuntimeError:
        return False


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


class MemoryModule(base.MemoryModule):
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

    def __init__(self):
        super().__init__()
        self._memories_rv: dict[str, ResetValue] = {}

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

        assert "value" in reset_kwargs
        if name not in self._memories_rv:
            assert "sizes" in reset_kwargs
            if "has_batch" not in reset_kwargs:
                reset_kwargs["has_batch"] = False

            _validate_sizes(reset_kwargs)
            _validate_has_batch(reset_kwargs)
            self._memories_rv[name] = ResetValue(**reset_kwargs)
            return

        if "sizes" in reset_kwargs:
            assert not strict or (
                self._memories_rv[name].sizes == reset_kwargs["sizes"]
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


class BaseNode(MemoryModule):
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
        self._def_param("v_threshold", v_threshold, **_factory_kwargs)
        self._def_param("v_reset", v_reset, **_factory_kwargs)

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

    # TODO: improve
    def _def_param(self, name, val, *, allow_trailing_dims: bool = False, **kwargs):
        val = torch.as_tensor(val, **kwargs)
        if hasattr(self, name):
            delattr(self, name)
        neuron_shape = self.n_neuron
        if val.ndim == 0:
            val = expand_leading_dims(val, neuron_shape, match_full_shape=True)
        elif val.shape[: len(neuron_shape)] != neuron_shape:
            if val.ndim == 1 and val.numel() == self.size:
                val = val.reshape(neuron_shape)
            elif allow_trailing_dims:
                val = expand_leading_dims(val, neuron_shape)
            elif is_broadcastable(val.shape, neuron_shape):
                val = expand_leading_dims(val, neuron_shape, match_full_shape=True)
            else:
                raise ValueError(
                    f"{name} shape {tuple(val.shape)} is not broadcastable to "
                    f"{neuron_shape}."
                )
        if name in self.trainable_param:
            self.register_parameter(name, torch.nn.Parameter(val))
        else:
            self.register_buffer(name, val, persistent=True)

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
