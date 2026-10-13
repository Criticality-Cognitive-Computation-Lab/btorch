"""Tests for Delay buffer."""

import pytest
import torch

from btorch.models.history import Delay


class TestDelayCircular:
    """Test Delay with circular buffer mode (default)."""

    def test_basic_initialization(self):
        """Test basic initialization without state."""
        history = Delay(n_neuron=10, max_delay_steps=5)
        assert history.n_neuron == (10,)
        assert history.size == 10
        assert history.max_delay == 5
        assert history.use_circular_buffer is True
        assert hasattr(history, "_cursor")

    def test_init_state_creates_buffer(self):
        """Test that init_state creates the history buffer."""
        history = Delay(n_neuron=10, max_delay_steps=5)
        history.init_state(batch_size=2, dtype=torch.float32)

        assert hasattr(history, "history")
        assert history.history is not None
        # Memory system stores as (batch, delay, neuron)
        assert history.history.shape == (2, 5, 10)
        assert history.history.dtype == torch.float32

    def test_update_requires_init(self):
        """Test that update requires init_state first."""
        history = Delay(n_neuron=10, max_delay_steps=5)
        spike = torch.randn(2, 10, dtype=torch.float32)

        # Should raise RuntimeError if not initialized
        with pytest.raises(RuntimeError, match="not initialized"):
            history.update(spike)

        # After init_state, should work
        history.init_state(batch_size=2, dtype=torch.float32)
        history.update(spike)
        # Memory stores as (batch, delay, neuron), so spike is at [:, 0, :]
        assert torch.allclose(history.history[:, 0, :], spike)

    def test_circular_buffer_behavior(self):
        """Test that circular buffer wraps around correctly."""
        history = Delay(n_neuron=3, max_delay_steps=3)
        history.init_state(batch_size=1, dtype=torch.float32)

        # Push 5 spikes (more than max_delay)
        spikes = [
            torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32),
            torch.tensor([[0.0, 2.0, 0.0]], dtype=torch.float32),
            torch.tensor([[0.0, 0.0, 3.0]], dtype=torch.float32),
            torch.tensor([[4.0, 0.0, 0.0]], dtype=torch.float32),
            torch.tensor([[0.0, 5.0, 0.0]], dtype=torch.float32),
        ]

        for s in spikes:
            history.update(s)

        # After 5 updates with max_delay=3:
        # With circular buffer, cursor wraps around
        # Most recent (t=4) at cursor-1, then t=3, t=2
        # Get them via get_delay to verify correct retrieval
        assert history.get_delay(0)[0, 0].item() == 0.0  # spike 4: [0, 5, 0]
        assert history.get_delay(0)[0, 1].item() == 5.0
        assert history.get_delay(1)[0, 0].item() == 4.0  # spike 3: [4, 0, 0]
        assert history.get_delay(2)[0, 2].item() == 3.0  # spike 2: [0, 0, 3]

    def test_get_delay_returns_correct_spike(self):
        """Test get_delay returns spike from correct timestep."""
        history = Delay(n_neuron=3, max_delay_steps=5)
        history.init_state(batch_size=1, dtype=torch.float32)

        # Push distinct spikes
        for i in range(5):
            spike = torch.tensor([[float(i), 0.0, 0.0]], dtype=torch.float32)
            history.update(spike)

        # Current (delay=0) should be spike 4
        assert history.get_delay(0)[0, 0].item() == 4.0
        # delay=1 should be spike 3
        assert history.get_delay(1)[0, 0].item() == 3.0
        # delay=4 should be spike 0
        assert history.get_delay(4)[0, 0].item() == 0.0

    def test_get_delay_bounds_check(self):
        """Test that get_delay raises error for out-of-bounds delays."""
        history = Delay(n_neuron=3, max_delay_steps=5)
        history.init_state(batch_size=1, dtype=torch.float32)
        history.update(torch.randn(1, 3, dtype=torch.float32))

        with pytest.raises(IndexError, match="delay_steps .* >= max_delay"):
            history.get_delay(5)

        with pytest.raises(IndexError, match="n_delays .* > max_delay"):
            history.get_flattened(6)

    def test_get_recent_returns_list(self):
        """Test get_recent returns list of recent spikes."""
        history = Delay(n_neuron=3, max_delay_steps=5)
        history.init_state(batch_size=1, dtype=torch.float32)

        for i in range(5):
            spike = torch.tensor([[float(i), 0.0, 0.0]], dtype=torch.float32)
            history.update(spike)

        recent = history.get_recent(3)
        assert len(recent) == 3
        assert recent[0][0, 0].item() == 4.0  # Most recent
        assert recent[1][0, 0].item() == 3.0
        assert recent[2][0, 0].item() == 2.0

    def test_get_recent_bounds_check(self):
        """Test that get_recent raises error for too many steps."""
        history = Delay(n_neuron=3, max_delay_steps=5)
        history.init_state(batch_size=1, dtype=torch.float32)

        with pytest.raises(IndexError, match="n_steps .* > max_delay"):
            history.get_recent(6)

    def test_get_flattened_format(self):
        """Test get_flattened returns correct interleaved format."""
        history = Delay(n_neuron=2, max_delay_steps=3)
        history.init_state(batch_size=1, dtype=torch.float32)

        # Push 3 spikes with distinct patterns
        history.update(torch.tensor([[1.0, 10.0]], dtype=torch.float32))  # t=0
        history.update(torch.tensor([[2.0, 20.0]], dtype=torch.float32))  # t=1
        history.update(torch.tensor([[3.0, 30.0]], dtype=torch.float32))  # t=2

        # At t=2 (current), flattened should be:
        # [n0_d0, n0_d1, n0_d2, n1_d0, n1_d1, n1_d2]
        # = [3, 2, 1, 30, 20, 10]
        flat = history.get_flattened(3)

        assert flat.shape == (1, 6)
        expected = torch.tensor(
            [[3.0, 2.0, 1.0, 30.0, 20.0, 10.0]], dtype=torch.float32
        )
        assert torch.allclose(flat, expected)

    def test_get_flattened_batch_handling(self):
        """Test get_flattened handles batch dimensions correctly."""
        history = Delay(n_neuron=2, max_delay_steps=3)
        history.init_state(batch_size=(2, 3), dtype=torch.float32)

        spike = torch.randn(2, 3, 2, dtype=torch.float32)
        history.update(spike)

        flat = history.get_flattened(3)
        # Shape should be (2, 3, 2*3) = (2, 3, 6)
        assert flat.shape == (2, 3, 6)

    def test_no_batch_dimension(self):
        """Test Delay works without batch dimension."""
        history = Delay(n_neuron=3, max_delay_steps=3)
        history.init_state(dtype=torch.float32)

        spike = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
        history.update(spike)

        # Memory system stores as (delay, neuron) when no batch
        assert history.history.shape == (3, 3)
        retrieved = history.get_delay(0)
        assert torch.allclose(retrieved, spike)

    def test_chaining_support(self):
        """Test that update returns self for method chaining."""
        history = Delay(n_neuron=3, max_delay_steps=3)
        history.init_state(dtype=torch.float32)

        result = history.update(torch.randn(3, dtype=torch.float32))
        assert result is history

        # Test chaining works
        history.update(torch.randn(3, dtype=torch.float32)).update(
            torch.randn(3, dtype=torch.float32)
        )

    def test_device_preservation(self):
        """Test that device is preserved through operations."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        history = Delay(n_neuron=3, max_delay_steps=3)
        history.init_state(batch_size=2, device="cuda")

        spike = torch.randn(2, 3, device="cuda")
        history.update(spike)
        assert history.history.device.type == "cuda"

        retrieved = history.get_delay(0)
        assert retrieved.device.type == "cuda"

    def test_dtype_preservation(self):
        """Test that dtype is preserved through operations."""
        history = Delay(n_neuron=3, max_delay_steps=3)
        history.init_state(batch_size=2, dtype=torch.float64)

        # Test with float64
        spike = torch.randn(2, 3, dtype=torch.float64)
        history.update(spike)
        assert history.history.dtype == torch.float64

        # Test with float32
        history2 = Delay(n_neuron=3, max_delay_steps=3)
        history2.init_state(batch_size=2, dtype=torch.float32)
        spike2 = torch.randn(2, 3, dtype=torch.float32)
        history2.update(spike2)
        assert history2.history.dtype == torch.float32

    def test_reset_clears_buffer(self):
        """Test that reset clears the buffer and cursor."""
        history = Delay(n_neuron=3, max_delay_steps=3)
        history.init_state(batch_size=1, dtype=torch.float32)

        # Add some data
        history.update(torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32))
        history.update(torch.tensor([[4.0, 5.0, 6.0]], dtype=torch.float32))

        # Reset
        history.reset()

        # Buffer should be zeros and cursor at 0
        assert torch.all(history.history == 0)
        assert history._cursor.item() == 0

    def test_tuple_n_neuron(self):
        """Test Delay with multi-dimensional neuron shape."""
        history = Delay(n_neuron=(4, 5), max_delay_steps=3)
        assert history.n_neuron == (4, 5)
        assert history.size == 20

        history.init_state(batch_size=2, dtype=torch.float32)
        # Memory system stores as (batch, delay, 4, 5)
        assert history.history.shape == (2, 3, 4, 5)

        spike = torch.randn(2, 4, 5, dtype=torch.float32)
        history.update(spike)
        assert torch.allclose(history.get_delay(0), spike)

        # Test get_flattened
        history.update(torch.randn(2, 4, 5, dtype=torch.float32))
        history.update(torch.randn(2, 4, 5, dtype=torch.float32))
        flat = history.get_flattened(3)
        # Shape: (batch, size * n_delays) = (2, 20 * 3) = (2, 60)
        assert flat.shape == (2, 60)


class TestDelayCat:
    """Test Delay with torch.cat mode (for torch.compile)."""

    def test_initialization_cat_mode(self):
        """Test initialization in cat mode."""
        history = Delay(n_neuron=10, max_delay_steps=5, use_circular_buffer=False)
        assert history.n_neuron == (10,)
        assert history.max_delay == 5
        assert history.use_circular_buffer is False
        assert not hasattr(history, "_cursor")

    def test_cat_buffer_behavior(self):
        """Test that cat mode shifts correctly."""
        history = Delay(n_neuron=3, max_delay_steps=3, use_circular_buffer=False)
        history.init_state(batch_size=1, dtype=torch.float32)

        # Push 5 spikes (more than max_delay)
        spikes = [
            torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32),
            torch.tensor([[0.0, 2.0, 0.0]], dtype=torch.float32),
            torch.tensor([[0.0, 0.0, 3.0]], dtype=torch.float32),
            torch.tensor([[4.0, 0.0, 0.0]], dtype=torch.float32),
            torch.tensor([[0.0, 5.0, 0.0]], dtype=torch.float32),
        ]

        for s in spikes:
            history.update(s)

        # With torch.cat: most recent is at index 0
        # history[:, 0, :] = most recent = spikes[4] = [0, 5, 0]
        # history[:, 1, :] = second = spikes[3] = [4, 0, 0]
        # history[:, 2, :] = third = spikes[2] = [0, 0, 3]

        assert torch.allclose(history.history[:, 0, :], spikes[4])
        assert torch.allclose(history.history[:, 1, :], spikes[3])
        assert torch.allclose(history.history[:, 2, :], spikes[2])

        # Verify get_delay works correctly
        assert torch.allclose(history.get_delay(0), spikes[4])
        assert torch.allclose(history.get_delay(1), spikes[3])
        assert torch.allclose(history.get_delay(2), spikes[2])

    def test_cat_get_flattened(self):
        """Test get_flattened in cat mode."""
        history = Delay(n_neuron=2, max_delay_steps=3, use_circular_buffer=False)
        history.init_state(batch_size=1, dtype=torch.float32)

        # Push 3 spikes
        history.update(torch.tensor([[1.0, 10.0]], dtype=torch.float32))  # t=0
        history.update(torch.tensor([[2.0, 20.0]], dtype=torch.float32))  # t=1
        history.update(torch.tensor([[3.0, 30.0]], dtype=torch.float32))  # t=2

        # With cat mode: most recent at index 0
        # [n0_d0, n0_d1, n0_d2, n1_d0, n1_d1, n1_d2]
        # = [3, 2, 1, 30, 20, 10]
        flat = history.get_flattened(3)

        assert flat.shape == (1, 6)
        expected = torch.tensor(
            [[3.0, 2.0, 1.0, 30.0, 20.0, 10.0]], dtype=torch.float32
        )
        assert torch.allclose(flat, expected)


@pytest.mark.parametrize("persistent", [False, True])
def test_circular_delay_continuation_state(persistent):
    """Restore the ring ordering, not just its stored tensor values.

    Wrapping a three-slot ring five times leaves a nonzero cursor.
    Restoring history without that cursor produces different delayed
    inputs; both the functional state interface and persistent
    checkpoints must preserve it.
    """
    from btorch.models.functional import named_hidden_states, set_hidden_states

    delay = Delay(2, max_delay_steps=3)
    delay.init_state(batch_size=2, dtype=torch.float64, persistent=persistent)
    for step in range(5):
        delay.update(torch.full((2, 2), float(step), dtype=torch.float64))
    snapshot = named_hidden_states(delay, clone=True)
    assert set(snapshot) == {"history", "_cursor"}
    assert snapshot["_cursor"].shape == ()
    assert snapshot["_cursor"].dtype == torch.long
    checkpoint = delay.state_dict()
    assert ("history" in checkpoint) == persistent
    assert ("_cursor" in checkpoint) == persistent

    restored = Delay(2, max_delay_steps=3)
    restored.init_state(batch_size=2, dtype=torch.float64, persistent=persistent)
    if persistent:
        restored.load_state_dict(checkpoint)
    else:
        set_hidden_states(restored, snapshot)
    for step in range(5, 9):
        value = torch.full((2, 2), float(step), dtype=torch.float64)
        delay.update(value)
        restored.update(value)
        torch.testing.assert_close(restored.get_flattened(3), delay.get_flattened(3))
    restored.reset(inplace=True)
    assert restored._cursor.item() == 0
    torch.testing.assert_close(
        restored.get_flattened(3), torch.zeros((2, 6), dtype=torch.float64)
    )


def test_circular_delay_reset_preserves_cursor_type_and_skipped_state():
    """Selective initialization and reset retain a scalar integer cursor.

    Float dtype changes apply to tensor history only. A caller
    preserving cursor/history retains their checkpoint persistence too,
    so checkpoints never contain only one half of the ring state.
    """
    delay = Delay(2, max_delay_steps=3)
    delay.init_state(batch_size=2)
    delay.update(torch.ones(2, 2))
    history, cursor = delay.history.clone(), delay._cursor.clone()
    delay.init_state(
        batch_size=2,
        dtype=torch.float64,
        persistent=True,
        skip_mem_name=("history", "_cursor"),
    )
    torch.testing.assert_close(delay.history, history)
    torch.testing.assert_close(delay._cursor, cursor)
    assert "_cursor" not in delay.state_dict()
    assert "history" not in delay.state_dict()
    delay.reset(dtype=torch.float64, skip_mem_name=("_cursor",))
    torch.testing.assert_close(delay._cursor, cursor)
    assert delay.history.dtype == torch.float64
    delay.reset(dtype=torch.float32)
    assert delay._cursor.dtype == torch.long and delay._cursor.shape == ()
    assert delay._cursor.device == delay.history.device
    assert delay._cursor.item() == 0
