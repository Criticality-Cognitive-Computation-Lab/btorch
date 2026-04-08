"""Tests for SpikeHistory buffer."""

import pytest
import torch

from btorch.models.history import SpikeHistory


class TestSpikeHistoryCircular:
    """Test SpikeHistory with circular buffer mode (default)."""

    def test_basic_initialization(self):
        """Test basic initialization without state."""
        history = SpikeHistory(n_neuron=10, max_delay_steps=5)
        assert history.n_neuron == (10,)
        assert history.size == 10
        assert history.max_delay == 5
        assert history.use_circular_buffer is True
        assert hasattr(history, "_cursor")

    def test_init_state_creates_buffer(self):
        """Test that init_state creates the history buffer."""
        history = SpikeHistory(n_neuron=10, max_delay_steps=5)
        history.init_state(batch_size=2, dtype=torch.float32)

        assert hasattr(history, "history")
        assert history.history is not None
        # Memory system stores as (batch, delay, neuron)
        assert history.history.shape == (2, 5, 10)
        assert history.history.dtype == torch.float32

    def test_update_requires_init(self):
        """Test that update requires init_state first."""
        history = SpikeHistory(n_neuron=10, max_delay_steps=5)
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
        history = SpikeHistory(n_neuron=3, max_delay_steps=3)
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
        history = SpikeHistory(n_neuron=3, max_delay_steps=5)
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
        history = SpikeHistory(n_neuron=3, max_delay_steps=5)
        history.init_state(batch_size=1, dtype=torch.float32)
        history.update(torch.randn(1, 3, dtype=torch.float32))

        with pytest.raises(IndexError, match="delay_steps .* >= max_delay"):
            history.get_delay(5)

        with pytest.raises(IndexError, match="n_delays .* > max_delay"):
            history.get_flattened(6)

    def test_get_recent_returns_list(self):
        """Test get_recent returns list of recent spikes."""
        history = SpikeHistory(n_neuron=3, max_delay_steps=5)
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
        history = SpikeHistory(n_neuron=3, max_delay_steps=5)
        history.init_state(batch_size=1, dtype=torch.float32)

        with pytest.raises(IndexError, match="n_steps .* > max_delay"):
            history.get_recent(6)

    def test_get_flattened_format(self):
        """Test get_flattened returns correct interleaved format."""
        history = SpikeHistory(n_neuron=2, max_delay_steps=3)
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
        history = SpikeHistory(n_neuron=2, max_delay_steps=3)
        history.init_state(batch_size=(2, 3), dtype=torch.float32)

        spike = torch.randn(2, 3, 2, dtype=torch.float32)
        history.update(spike)

        flat = history.get_flattened(3)
        # Shape should be (2, 3, 2*3) = (2, 3, 6)
        assert flat.shape == (2, 3, 6)

    def test_no_batch_dimension(self):
        """Test SpikeHistory works without batch dimension."""
        history = SpikeHistory(n_neuron=3, max_delay_steps=3)
        history.init_state(dtype=torch.float32)

        spike = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
        history.update(spike)

        # Memory system stores as (delay, neuron) when no batch
        assert history.history.shape == (3, 3)
        retrieved = history.get_delay(0)
        assert torch.allclose(retrieved, spike)

    def test_chaining_support(self):
        """Test that update returns self for method chaining."""
        history = SpikeHistory(n_neuron=3, max_delay_steps=3)
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

        history = SpikeHistory(n_neuron=3, max_delay_steps=3)
        history.init_state(batch_size=2, device="cuda")

        spike = torch.randn(2, 3, device="cuda")
        history.update(spike)
        assert history.history.device.type == "cuda"

        retrieved = history.get_delay(0)
        assert retrieved.device.type == "cuda"

    def test_dtype_preservation(self):
        """Test that dtype is preserved through operations."""
        history = SpikeHistory(n_neuron=3, max_delay_steps=3)
        history.init_state(batch_size=2, dtype=torch.float64)

        # Test with float64
        spike = torch.randn(2, 3, dtype=torch.float64)
        history.update(spike)
        assert history.history.dtype == torch.float64

        # Test with float32
        history2 = SpikeHistory(n_neuron=3, max_delay_steps=3)
        history2.init_state(batch_size=2, dtype=torch.float32)
        spike2 = torch.randn(2, 3, dtype=torch.float32)
        history2.update(spike2)
        assert history2.history.dtype == torch.float32

    def test_reset_clears_buffer(self):
        """Test that reset clears the buffer and cursor."""
        history = SpikeHistory(n_neuron=3, max_delay_steps=3)
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
        """Test SpikeHistory with multi-dimensional neuron shape."""
        history = SpikeHistory(n_neuron=(4, 5), max_delay_steps=3)
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


class TestSpikeHistoryCat:
    """Test SpikeHistory with torch.cat mode (for torch.compile)."""

    def test_initialization_cat_mode(self):
        """Test initialization in cat mode."""
        history = SpikeHistory(
            n_neuron=10, max_delay_steps=5, use_circular_buffer=False
        )
        assert history.n_neuron == (10,)
        assert history.max_delay == 5
        assert history.use_circular_buffer is False
        assert not hasattr(history, "_cursor")

    def test_cat_buffer_behavior(self):
        """Test that cat mode shifts correctly."""
        history = SpikeHistory(n_neuron=3, max_delay_steps=3, use_circular_buffer=False)
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
        history = SpikeHistory(n_neuron=2, max_delay_steps=3, use_circular_buffer=False)
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


class TestDelayedSynapse:
    """Test suite for DelayedSynapse class."""

    def test_delayed_synapse_init(self):
        """Test DelayedSynapse initialization."""
        from btorch.models.history import DelayedSynapse

        class MockLinear:
            def __call__(self, x):
                return x

        synapse = DelayedSynapse(n_neuron=10, linear=MockLinear(), max_delay_steps=5)
        assert synapse.n_neuron == (10,)
        assert synapse.history is not None
        assert synapse.history.max_delay == 5

    def test_delayed_synapse_init_state(self):
        """Test DelayedSynapse state initialization."""
        from btorch.models.history import DelayedSynapse

        class MockLinear:
            def __call__(self, x):
                return x

        synapse = DelayedSynapse(n_neuron=10, linear=MockLinear(), max_delay_steps=5)
        synapse.init_state(batch_size=2)

        assert hasattr(synapse, "psc")
        assert synapse.psc.shape == (2, 10)

    def test_delayed_synapse_forward_circular(self):
        """Test DelayedSynapse forward pass with circular buffer."""
        from btorch.models.history import DelayedSynapse

        # Mock linear that maps from flattened history (size * n_delays) to neuron space
        class MockLinear:
            def __init__(self, in_features, out_features):
                self.in_features = in_features
                self.out_features = out_features
                self.weight = torch.zeros(out_features, in_features)
                for i in range(out_features):
                    self.weight[i, i * 3 : (i + 1) * 3] = 1.0

            def __call__(self, x):
                return x @ self.weight.T

        linear = MockLinear(in_features=9, out_features=3)
        synapse = DelayedSynapse(
            n_neuron=3, linear=linear, max_delay_steps=3, use_circular_buffer=True
        )
        synapse.init_state(batch_size=1, dtype=torch.float32)

        spike = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32)
        psc = synapse(spike)

        # After one update, only delay=0 has the spike
        expected = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32)
        assert torch.allclose(psc, expected)

    def test_delayed_synapse_forward_cat(self):
        """Test DelayedSynapse forward pass with cat mode."""
        from btorch.models.history import DelayedSynapse

        class MockLinear:
            def __init__(self, in_features, out_features):
                self.in_features = in_features
                self.out_features = out_features
                self.weight = torch.zeros(out_features, in_features)
                for i in range(out_features):
                    self.weight[i, i * 3 : (i + 1) * 3] = 1.0

            def __call__(self, x):
                return x @ self.weight.T

        linear = MockLinear(in_features=9, out_features=3)
        synapse = DelayedSynapse(
            n_neuron=3, linear=linear, max_delay_steps=3, use_circular_buffer=False
        )
        synapse.init_state(batch_size=1, dtype=torch.float32)

        spike = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32)
        psc = synapse(spike)

        expected = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32)
        assert torch.allclose(psc, expected)
