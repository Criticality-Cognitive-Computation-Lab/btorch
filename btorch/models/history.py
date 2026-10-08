"""Spike history buffer for delayed synapse models.

This module provides a rolling buffer for spike history to support
heterogeneous synaptic delays. The buffer maintains the last N timesteps
of spike activity, allowing delayed connections to access past spikes.

Example:
    >>> history = SpikeHistory(n_neuron=100, max_delay_steps=5)
    >>> history.init_state(batch_size=2)
    >>> for t in range(T):
    ...     z = neuron(v)  # current spikes
    ...     history.update(z)  # stores spike and advances buffer
    ...     # Get flattened history for matrix multiply
    ...     z_delayed = history.get_flattened(5)
    ...     psc = linear(z_delayed)

The flattened format interleaves neurons and delays:
    [n0_d0, n0_d1, n0_d2, ..., n1_d0, n1_d1, ...]
where nX_dY = neuron X at delay Y.
"""

import torch
from torch import Tensor

from .base import MemoryModule, flatten_neuron, normalize_n_neuron


def expand_delay_sequence(z_seq: Tensor, n_delays: int) -> Tensor:
    """Delay-expand a whole spike sequence that starts from an empty history.

    Stateless, vectorised counterpart of stepping
    ``history.update(z_seq[t]); history.get_flattened(n_delays)`` from a freshly
    reset :class:`SpikeHistory`: entry ``n * n_delays + d`` of step ``t`` is
    ``z_seq[t - d, ..., n]`` (zero for ``t < d``).

    Args:
        z_seq: Spike sequence of shape ``(T, *batch, size)``.
        n_delays: Number of delay bins.

    Returns:
        Tensor of shape ``(T, *batch, size * n_delays)``.
    """
    n_step = z_seq.shape[0]
    if n_delays > 1:
        pad = z_seq.new_zeros((n_delays - 1, *z_seq.shape[1:]))
        z_seq = torch.cat([pad, z_seq], dim=0)
    # Column d holds the sequence shifted d steps into the future.
    shifted = [
        z_seq[n_delays - 1 - d : n_delays - 1 - d + n_step] for d in range(n_delays)
    ]
    return torch.stack(shifted, dim=-1).flatten(-2)


class SpikeHistory(MemoryModule):
    """Rolling spike history buffer for delayed synaptic connections.

    Maintains a buffer of the last `max_delay_steps` spike patterns.
    Supports batch dimensions and provides flattened output for matrix
    multiplication with delay-expanded connection matrices.

    This implementation uses the MemoryModule buffer system for consistent
    state management with other btorch components like BasePSC.

    Supports two update modes:
    - Circular buffer (default): Memory-efficient for simulation, uses
      O(max_delay * n_neuron) memory with no copying.
    - torch.cat: torch.compile compatible for backpropagation training,
      but requires O(max_delay * n_neuron) memory copying each step.

    The history buffer has shape (max_delay_steps, *batch, *n_neuron) where
    batch dimensions are handled internally.

    Args:
        n_neuron: Number of neurons (int or tuple of dimensions).
        max_delay_steps: Maximum delay to store (default: 5).
        use_circular_buffer: If True (default), use circular buffer for
            memory-efficient simulation. Set to False for torch.compile
            compatibility during training.

    Attributes:
        max_delay: Maximum delay steps stored.
        use_circular_buffer: Whether circular buffer mode is enabled.

    Example:
        >>> # For simulation (efficient, default)
        >>> history = SpikeHistory(n_neuron=10, max_delay_steps=3)
        >>> history.init_state(batch_size=2)
        >>>
        >>> # For training with torch.compile
        >>> history = SpikeHistory(n_neuron=10, max_delay_steps=3,
        ...                        use_circular_buffer=False)
        >>> history.init_state(batch_size=2)
    """

    def __init__(
        self,
        n_neuron: int | tuple[int, ...],
        max_delay_steps: int = 5,
        use_circular_buffer: bool = True,
    ):
        super().__init__()
        self.n_neuron, self.size = normalize_n_neuron(n_neuron)
        self.max_delay = max_delay_steps
        self.use_circular_buffer = use_circular_buffer

        # Register history buffer with shape (max_delay_steps, *n_neuron)
        # The memory system will prepend batch dimensions:
        #   (*batch, max_delay, *n_neuron)
        # Use torch.tensor(0.0, dtype=torch.float32) to avoid Python's float64 default
        self.register_memory(
            "history",
            torch.tensor(0.0, dtype=torch.float32),
            (max_delay_steps, *self.n_neuron),
        )

        if use_circular_buffer:
            self.register_buffer(
                "_cursor",
                torch.tensor(0, dtype=torch.long),
                persistent=False,
            )

    def init_state(
        self,
        batch_size: int | tuple[int, ...] | None = None,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
        persistent: bool = False,
        skip_mem_name: tuple[str, ...] = (),
    ) -> None:
        """Initialize or reset the history buffer.

        Args:
            batch_size: Batch dimensions. If None, no batch dim.
            dtype: Data type for buffer.
            device: Device to place buffer on.
            persistent: Whether buffer should be persistent in stateDict.
            skip_mem_name: Names of memories to skip initialization.
        """
        if self.use_circular_buffer:
            super().init_state(
                batch_size, dtype, device, persistent, skip_mem_name + ("_cursor",)
            )
            self._cursor = torch.tensor(0, dtype=torch.long, device=device)
        else:
            super().init_state(batch_size, dtype, device, persistent, skip_mem_name)

    def reset(
        self,
        batch_size: int | tuple[int, ...] | None = None,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
        skip_mem_name: tuple[str, ...] = (),
        inplace: bool = False,
    ) -> None:
        """Reset the history buffer to initial state.

        Args:
            batch_size: Batch dimensions. If None, uses existing batch size.
            dtype: Data type for buffer.
            device: Device to place buffer on.
            skip_mem_name: Names of memories to skip reset.
            inplace: If True, zero the existing buffer (and cursor) instead of
                rebinding them; see :meth:`MemoryModule.reset`.
        """
        super().reset(batch_size, dtype, device, skip_mem_name, inplace=inplace)
        if self.use_circular_buffer:
            if inplace:
                self._cursor.zero_()
            else:
                self._cursor = torch.tensor(
                    0, dtype=torch.long, device=self._cursor.device
                )

    def detach(self) -> None:
        """Detach the buffered spikes from the autograd graph.

        With batch dimensions the stored buffer is a permuted view (see
        :meth:`_set_history_with_delay_first`), which cannot be detached in
        place, so the buffer is rebound to a detached tensor instead.
        """
        self.history = self.history.detach()

    def _get_history_with_delay_first(self) -> Tensor:
        """Get history tensor with delay dimension moved to front.

        The memory system stores history as (*batch, max_delay,
        *n_neuron). This moves it to (max_delay, *batch, *n_neuron) for
        easier indexing.
        """
        history = self.history
        n_batch_dims = history.ndim - 1 - len(self.n_neuron)
        if n_batch_dims <= 0:
            return history

        perm = (
            [n_batch_dims]
            + list(range(n_batch_dims))
            + list(range(n_batch_dims + 1, history.ndim))
        )
        return history.permute(*perm)

    def _set_history_with_delay_first(self, value: Tensor) -> None:
        """Set history tensor from delay-first format.

        Converts from (max_delay, *batch, *n_neuron) to (*batch,
        max_delay, *n_neuron).
        """
        history = self.history
        n_batch_dims = history.ndim - 1 - len(self.n_neuron)
        if n_batch_dims <= 0:
            self.history = value
            return

        perm = (
            list(range(1, n_batch_dims + 1))
            + [0]
            + list(range(n_batch_dims + 1, value.ndim))
        )
        self.history = value.permute(*perm)

    def update(self, spike: Tensor) -> "SpikeHistory":
        """Push new spike pattern into history buffer.

        Args:
            spike: Spike tensor of shape (*batch, *n_neuron).

        Returns:
            Self for method chaining.

        Raises:
            RuntimeError: If buffer not initialized.
        """
        if not hasattr(self, "history") or self.history is None:
            raise RuntimeError(
                "SpikeHistory buffer not initialized. "
                "Call init_state() before first update()."
            )

        if spike.dtype != self.history.dtype:
            spike = spike.to(self.history.dtype)

        if self.use_circular_buffer:
            self._update_circular(spike)
        else:
            self._update_cat(spike)

        return self

    def _update_circular(self, spike: Tensor) -> None:
        """Update using circular buffer (efficient for simulation)."""
        history = self._get_history_with_delay_first()

        history[self._cursor] = spike

        self._cursor.fill_((self._cursor + 1) % self.max_delay)

        if history is not self.history:
            self._set_history_with_delay_first(history)

    def _update_cat(self, spike: Tensor) -> None:
        """Update using torch.cat (torch.compile compatible for training)."""
        history = self._get_history_with_delay_first()

        # (*batch, *n_neuron) -> (1, *batch, *n_neuron)
        spike = spike.unsqueeze(0)

        new_history = torch.cat([spike, history[:-1]], dim=0)

        if history is not self.history:
            self._set_history_with_delay_first(new_history)
        else:
            self.history = new_history

    def get_delay(self, delay_steps: int) -> Tensor:
        """Get spike pattern at specific delay from history.

        Args:
            delay_steps: Number of timesteps in the past (0 = current).

        Returns:
            Spike tensor from `delay_steps` ago, shape (*batch, *n_neuron).

        Raises:
            IndexError: If delay_steps >= max_delay.
        """
        if delay_steps >= self.max_delay:
            raise IndexError(
                f"delay_steps ({delay_steps}) >= max_delay ({self.max_delay}). "
                "Increase max_delay_steps or use smaller delays."
            )

        history = self._get_history_with_delay_first()

        if self.use_circular_buffer:
            # Most recent entry is at cursor - 1; the index stays a tensor to
            # avoid a GPU/CPU sync.
            idx = (self._cursor - 1 - delay_steps) % self.max_delay
            return history[idx]
        else:
            # cat mode: most recent entry is at index 0
            return history[delay_steps]

    def get_recent(self, n_steps: int) -> list[Tensor]:
        """Get spikes for recent n_steps: [t-0, t-1, ..., t-(n_steps-1)].

        Args:
            n_steps: Number of recent timesteps to retrieve.

        Returns:
            List of spike tensors, each shape (*batch, *n_neuron).

        Raises:
            IndexError: If n_steps > max_delay.
        """
        if n_steps > self.max_delay:
            raise IndexError(f"n_steps ({n_steps}) > max_delay ({self.max_delay})")
        return [self.get_delay(d) for d in range(n_steps)]

    def get_flattened(self, n_delays: int) -> Tensor:
        """Get flattened history suitable for delay-expanded matrix multiply.

        The output format interleaves neurons and delays:
            [n0_d0, n0_d1, ..., n0_d{n-1}, n1_d0, n1_d1, ...]
        where nX_dY = neuron X at delay Y.

        This matches the row ordering of matrices expanded by
        `expand_conn_for_delays()`.

        Args:
            n_delays: Number of delay steps to include (must be <= max_delay).

        Returns:
            Flattened tensor of shape (*batch, size * n_delays).

        Example:
            >>> history = SpikeHistory(n_neuron=3, max_delay_steps=5)
            >>> history.init_state(batch_size=1)
            >>> # After some updates...
            >>> flat = history.get_flattened(3)
            >>> flat.shape
            torch.Size([1, 9])  # 3 neurons * 3 delays
        """
        if n_delays > self.max_delay:
            raise IndexError(f"n_delays ({n_delays}) > max_delay ({self.max_delay})")

        delays = [self.get_delay(d) for d in range(n_delays)]
        stacked = torch.stack(delays, dim=0)

        # stacked: (n_delays, *batch, *n_neuron); size = product of n_neuron dims
        delays_flat, _ = flatten_neuron(stacked, self.n_neuron, self.size)

        # Permute to move delay dimension to the end: (*batch, size, n_delays)
        perm = list(range(1, delays_flat.ndim)) + [0]
        permuted = delays_flat.permute(*perm)

        return permuted.reshape(*permuted.shape[:-2], -1)

    def extra_repr(self) -> str:
        batch_str = ""
        if hasattr(self, "history") and self.history is not None:
            expected_ndim = 1 + len(self.n_neuron)
            if self.history.ndim > expected_ndim:
                n_batch_dims = self.history.ndim - expected_ndim
                batch_shape = self.history.shape[:n_batch_dims]
                batch_str = f", batch={batch_shape}"
        mode_str = ", circular" if self.use_circular_buffer else ", cat"
        return (
            f"n_neuron={self.n_neuron}, max_delay={self.max_delay}{mode_str}{batch_str}"
        )
