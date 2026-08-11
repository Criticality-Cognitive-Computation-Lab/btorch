"""Tests for the standalone recording engine (:mod:`btorch.monitor`), no RNN.

The engine records a *stepped computation* whose named state buffers are the
"columns" and whose timesteps are the "rows".  A :class:`Recorder` compiles a
Polars-style spec (``col("neuron.v").mean()`` etc.) and is driven one frame per
timestep; ``run`` returns the finalised ``{name: tensor}`` records.

Below, ``StatefulNet`` stands in for a real model: two submodules each holding one
state buffer (``neuron.v`` and ``synapse.psc``).  The tests read as usage examples
(top) followed by engine-internal guarantees and rejected specs (bottom).
"""

import pytest
import torch
import torch.nn as nn

from btorch.monitor import (
    EagerFrame,
    Recorder,
    Reducer,
    Resolver,
    col,
    grad,
    lit,
    map_seq,
    validate_reducer,
)
from btorch.monitor.ir import AGG, STEP, build


class _EMA(Reducer):
    """Exponential moving average -- a stateful reduction as a functional
    carry."""

    def __init__(self, alpha):
        self.alpha = alpha

    def init(self, example):
        return torch.zeros_like(example)

    def update(self, carry, value, valid):
        return self.alpha * value + (1 - self.alpha) * carry

    def finalize(self, carry):
        return carry


class _WeightedMean(Reducer):
    """Mean that uses ``valid`` to exclude warmup (carry = (sum, count)
    tuple)."""

    def init(self, example):
        return example.new_zeros(example.shape), example.new_zeros(())

    def update(self, carry, value, valid):
        total, count = carry
        return total + value * valid, count + valid

    def finalize(self, carry):
        total, count = carry
        return total / count.clamp(min=1)


T, BATCH, FEATURES = 6, 2, 3


class _Neuron(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("v", torch.zeros(BATCH, FEATURES))


class _Synapse(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("psc", torch.zeros(BATCH, FEATURES))


class StatefulNet(nn.Module):
    """A minimal module exposing two dotted state buffers: neuron.v, synapse.psc."""

    def __init__(self):
        super().__init__()
        self.neuron = _Neuron()
        self.synapse = _Synapse()


def record_over(net, specs):
    """Build a :class:`Recorder` for ``specs`` against ``net``'s state
    buffers."""
    return Recorder(specs, resolver=Resolver(net))


def run_steps(recorder, v_seq, psc_seq):
    """Drive ``recorder`` over a sequence, feeding each step's buffers, and
    return the finalised records.

    Mirrors how the RNN loop feeds per-step state.
    """
    net = recorder.resolver.root
    frames = []
    for v, psc in zip(v_seq, psc_seq):
        net.neuron.v, net.synapse.psc = v, psc
        frames.append(recorder.resolver.frame())
    return recorder.run(frames)


@pytest.fixture
def sequence():
    """A random (voltage, psc) trajectory of length T."""
    torch.manual_seed(0)
    v_seq = [torch.randn(BATCH, FEATURES) for _ in range(T)]
    psc_seq = [torch.randn(BATCH, FEATURES) for _ in range(T)]
    return v_seq, psc_seq


# ===========================================================================
# Usage examples: what the DSL records
# ===========================================================================
def test_bare_string_records_the_raw_trace(sequence):
    v_seq, psc_seq = sequence
    recorder = record_over(StatefulNet(), ["neuron.v"])
    records = run_steps(recorder, v_seq, psc_seq)
    # a bare dotted string stacks the buffer over time -> [T, B, F]
    torch.testing.assert_close(records["neuron.v"], torch.stack(v_seq, 0))
    assert recorder.raw_columns == {"neuron.v"}


def test_alias_renames_the_output_key(sequence):
    v_seq, psc_seq = sequence
    recorder = record_over(StatefulNet(), [col("neuron.v").alias("membrane")])
    records = run_steps(recorder, v_seq, psc_seq)
    assert set(records) == {"membrane"}


def test_bare_tensor_reference_resolves_like_a_dotted_name(sequence):
    v_seq, psc_seq = sequence
    net = StatefulNet()
    # col(net.neuron.v) identifies the buffer by identity, then records like a name
    recorder = Recorder([col(net.neuron.v).alias("v")], resolver=Resolver(net))
    records = run_steps(recorder, v_seq, psc_seq)
    torch.testing.assert_close(records["v"], torch.stack(v_seq, 0))


def test_mean_reduces_over_time(sequence):
    v_seq, psc_seq = sequence
    recorder = record_over(StatefulNet(), {"v_mean": col("neuron.v").mean()})
    records = run_steps(recorder, v_seq, psc_seq)
    torch.testing.assert_close(records["v_mean"], torch.stack(v_seq, 0).mean(0))


def test_scalar_reductions_over_time(sequence):
    v_seq, psc_seq = sequence
    stacked = torch.stack(v_seq, 0)
    recorder = record_over(
        StatefulNet(),
        {
            "sum": col("neuron.v").sum(),
            "last": col("neuron.v").last(),
            "first": col("neuron.v").first(),
            "min": col("neuron.v").min(),
            "max": col("neuron.v").max(),
            "count": col("neuron.v").count(),
        },
    )
    records = run_steps(recorder, v_seq, psc_seq)
    torch.testing.assert_close(records["sum"], stacked.sum(0))
    torch.testing.assert_close(records["last"], v_seq[-1])
    torch.testing.assert_close(records["first"], v_seq[0])
    torch.testing.assert_close(records["min"], stacked.min(0).values)
    torch.testing.assert_close(records["max"], stacked.max(0).values)
    torch.testing.assert_close(records["count"], torch.tensor(float(T)))


def test_std_and_var_match_population_statistics(sequence):
    v_seq, psc_seq = sequence
    stacked = torch.stack(v_seq, 0)
    recorder = record_over(
        StatefulNet(),
        {
            "std": col("neuron.v").std(),
            "var": col("neuron.v").var(),
        },
    )
    records = run_steps(recorder, v_seq, psc_seq)
    # streaming Welford; population (unbiased=False) by convention
    torch.testing.assert_close(records["var"], stacked.var(0, unbiased=False))
    torch.testing.assert_close(records["std"], stacked.std(0, unbiased=False))


def test_per_step_expression_over_two_columns(sequence):
    v_seq, psc_seq = sequence
    # elementwise over two state buffers, evaluated per timestep, stacked over time
    recorder = record_over(
        StatefulNet(), {"drive": col("neuron.v") - col("synapse.psc")}
    )
    records = run_steps(recorder, v_seq, psc_seq)
    torch.testing.assert_close(
        records["drive"], torch.stack(v_seq, 0) - torch.stack(psc_seq, 0)
    )


def test_col_plus_scalar_is_per_step(sequence):
    v_seq, psc_seq = sequence
    recorder = record_over(StatefulNet(), {"shifted": col("neuron.v") + 1})
    records = run_steps(recorder, v_seq, psc_seq)
    torch.testing.assert_close(records["shifted"], torch.stack(v_seq, 0) + 1)


def test_getitem_indexes_each_step_then_stacks(sequence):
    # col[...] is a PER-STEP index into each step's [B, N] value; to index the TIME
    # axis use map_seq(lambda V: V[k]).
    v_seq, psc_seq = sequence
    stacked = torch.stack(v_seq, 0)  # [T, B, F]
    recorder = record_over(
        StatefulNet(),
        {
            "feat1": col("neuron.v")[:, 1],  # per-step feature index -> [T, B]
            "feat1_mean": col("neuron.v")[:, 1].mean(),  # index, then reduce over time
            "at_step2": map_seq(
                lambda V: V[2], col("neuron.v")
            ),  # multistep: value at t=2
        },
    )
    records = run_steps(recorder, v_seq, psc_seq)
    torch.testing.assert_close(records["feat1"], stacked[:, :, 1])
    torch.testing.assert_close(records["feat1_mean"], stacked[:, :, 1].mean(0))
    torch.testing.assert_close(records["at_step2"], stacked[2])


def test_diff_is_first_difference_with_warmup_masked(sequence):
    v_seq, psc_seq = sequence
    stacked = torch.stack(v_seq, 0)
    recorder = record_over(
        StatefulNet(),
        {
            "diff": col("neuron.v").diff(),
            "mean_of_diff": col("neuron.v").diff().mean(),
        },
    )
    records = run_steps(recorder, v_seq, psc_seq)
    expected_diff = torch.zeros_like(stacked)
    expected_diff[1:] = stacked[1:] - stacked[:-1]  # step 0 has no predecessor -> 0
    torch.testing.assert_close(records["diff"], expected_diff)
    # the warmup step is excluded from the mean's denominator: divide by (T-1)
    torch.testing.assert_close(
        records["mean_of_diff"], (stacked[1:] - stacked[:-1]).mean(0)
    )


def test_second_difference_excludes_both_warmup_steps(sequence):
    # edge case: a window fed by another window (diff().diff()). Both the first
    # two steps are warmup, so count is T-2, not T.
    v_seq, psc_seq = sequence
    stacked = torch.stack(v_seq, 0)
    recorder = record_over(
        StatefulNet(),
        {
            "d2": col("neuron.v").diff().diff(),
            "d2_count": col("neuron.v").diff().diff().count(),
            "d2_mean": col("neuron.v").diff().diff().mean(),
        },
    )
    records = run_steps(recorder, v_seq, psc_seq)
    first = stacked[1:] - stacked[:-1]
    second = first[1:] - first[:-1]  # true 2nd difference, length T-2
    expected = torch.zeros_like(stacked)
    expected[2:] = second
    torch.testing.assert_close(records["d2"], expected)
    torch.testing.assert_close(records["d2_count"], torch.tensor(float(T - 2)))
    torch.testing.assert_close(records["d2_mean"], second.mean(0))


def test_map_seq_runs_a_custom_fn_on_the_whole_sequence(sequence):
    v_seq, psc_seq = sequence
    # map_seq materialises [T, B, F] and applies fn once (needed for non-streaming
    # ops like median)
    median_over_time = map_seq(lambda V: V.median(0).values, col("neuron.v"))
    recorder = record_over(StatefulNet(), {"median": median_over_time})
    records = run_steps(recorder, v_seq, psc_seq)
    expected = torch.stack(v_seq, 0).median(0).values
    torch.testing.assert_close(records["median"], expected)


def test_pipe_is_a_polars_style_alias_of_map_seq(sequence):
    v_seq, psc_seq = sequence
    piped = col("neuron.v").pipe(lambda V: V.sum(0))
    records = run_steps(record_over(StatefulNet(), {"s": piped}), v_seq, psc_seq)
    torch.testing.assert_close(records["s"], torch.stack(v_seq, 0).sum(0))


def test_map_seq_result_can_be_post_processed(sequence):
    # edge case: a map_seq nested under further arithmetic (not the root).
    v_seq, psc_seq = sequence
    recorder = record_over(
        StatefulNet(), {"twice_sum": map_seq(lambda V: V.sum(0), col("neuron.v")) * 2}
    )
    records = run_steps(recorder, v_seq, psc_seq)
    torch.testing.assert_close(records["twice_sum"], torch.stack(v_seq, 0).sum(0) * 2)


def test_grad_monitor_registers_a_grad_spec():
    # grad(...) is recorded via backward hooks by the consumer; the engine just
    # exposes it as a grad spec.
    recorder = record_over(StatefulNet(), {"v_grad": grad("neuron.v")})
    assert recorder.has_grad_specs
    assert recorder.grad_specs[0][0] == "v_grad"


def test_custom_reducer_fold(sequence):
    # a custom streaming reduction via col(...).fold(Reducer())
    v_seq, psc_seq = sequence
    recorder = record_over(StatefulNet(), {"v_ema": col("neuron.v").fold(_EMA(0.3))})
    records = run_steps(recorder, v_seq, psc_seq)
    expected = torch.zeros(BATCH, FEATURES)
    for v in v_seq:
        expected = 0.3 * v + 0.7 * expected
    torch.testing.assert_close(records["v_ema"], expected)


def test_custom_reducer_sees_validity_flag(sequence):
    # fold() over a windowed input: the validity flag excludes warmup, so a
    # custom weighted mean matches the built-in diff().mean() (divide by T-1).
    v_seq, psc_seq = sequence
    stacked = torch.stack(v_seq, 0)
    recorder = record_over(
        StatefulNet(), {"m": col("neuron.v").diff().fold(_WeightedMean())}
    )
    records = run_steps(recorder, v_seq, psc_seq)
    torch.testing.assert_close(records["m"], (stacked[1:] - stacked[:-1]).mean(0))


def test_validate_reducer_accepts_a_pure_reducer():
    validate_reducer(_EMA(0.1), torch.randn(BATCH, FEATURES))
    validate_reducer(_WeightedMean(), torch.randn(BATCH, FEATURES))


def test_validate_reducer_rejects_in_place_carry_mutation():
    class _BadInPlace(Reducer):
        def init(self, example):
            return torch.zeros_like(example)

        def update(self, carry, value, valid):
            carry += value  # in-place mutation -> breaks checkpoint / cudagraph
            return carry

        def finalize(self, carry):
            return carry

    with pytest.raises(RuntimeError, match="in place"):
        validate_reducer(_BadInPlace(), torch.randn(BATCH, FEATURES))


def test_validate_reducer_rejects_self_mutation():
    class _BadSelf(Reducer):
        def __init__(self):
            self.n = 0  # a plain-int counter -> still caught (not just tensors)

        def init(self, example):
            return torch.zeros_like(example)

        def update(self, carry, value, valid):
            self.n += 1  # per-step state on self -> silent breakage
            return carry + value

        def finalize(self, carry):
            return carry

    with pytest.raises(RuntimeError, match="its own state"):
        validate_reducer(_BadSelf(), torch.randn(BATCH, FEATURES))


def test_validate_reducer_rejects_value_mutation():
    class _BadValue(Reducer):
        def init(self, example):
            return torch.zeros_like(example)

        def update(self, carry, value, valid):
            value.add_(1)  # mutates the network's live state buffer!
            return carry + value

        def finalize(self, carry):
            return carry

    with pytest.raises(RuntimeError, match="value"):
        validate_reducer(_BadValue(), torch.randn(BATCH, FEATURES))


class _StructureChanging(Reducer):
    # init -> 1 leaf, update -> 2 leaves: the carry pytree is not constant.
    def init(self, example):
        return torch.zeros_like(example)

    def update(self, carry, value, valid):
        base = carry[0] if isinstance(carry, tuple) else carry
        return base + value, value

    def finalize(self, carry):
        return carry[0] if isinstance(carry, tuple) else carry


def test_validate_reducer_rejects_changing_carry_structure():
    with pytest.raises(RuntimeError, match="structure"):
        validate_reducer(_StructureChanging(), torch.randn(BATCH, FEATURES))


def test_engine_rejects_changing_carry_structure(sequence):
    v_seq, psc_seq = sequence
    recorder = record_over(
        StatefulNet(), {"g": col("neuron.v").fold(_StructureChanging())}
    )
    with pytest.raises(ValueError, match="constant"):
        run_steps(recorder, v_seq, psc_seq)


# ===========================================================================
# Engine-internal guarantees (memory, sharing, chunking, compile)
# ===========================================================================
def test_streaming_reduction_allocates_no_time_axis_buffer():
    # the memory-honesty guarantee: a pure reduction folds into an O(1) carry and
    # never materialises a [T, ...] buffer.
    recorder = record_over(StatefulNet(), {"v_mean": col("neuron.v").mean()})
    assert recorder.stack_nids == ()
    assert recorder.raw_columns == set()


def test_shared_subexpression_is_computed_once():
    # CSE: col("neuron.v") used by two reductions interns to a single Source node.
    graph = build(
        [col("neuron.v").mean().alias("a"), col("neuron.v").sum().alias("b")],
        Resolver(StatefulNet()),
    )
    sources = [node for node in graph.nodes if node.op == "source"]
    assert len(sources) == 1


def test_temporal_levels_and_stack_points():
    # a bare column is a per-step (STEP) value materialised as a stack point; a
    # reduction is a time-invariant (AGG) value.
    graph = build(
        [col("neuron.v"), col("neuron.v").mean().alias("m")], Resolver(StatefulNet())
    )
    by_op = {}
    for node in graph.nodes:
        by_op.setdefault(node.op, []).append(node)
    assert by_op["source"][0].level == STEP
    assert by_op["source"][0].is_stack_point  # raw root -> materialised
    assert by_op["reduce"][0].level == AGG


def test_carry_threading_across_chunks_matches_single_pass(sequence):
    # the carry is threaded functionally, so processing the sequence in chunks
    # (as grad_checkpoint / cpu_offload do) matches a single pass -- including a
    # window straddling a chunk boundary.
    v_seq, psc_seq = sequence
    specs = {
        "raw": col("neuron.v"),
        "mean": col("neuron.v").mean(),
        "diff": col("neuron.v").diff(),
        "mean_of_diff": col("neuron.v").diff().mean(),
        "std": col("neuron.v").std(),
    }
    single_pass = run_steps(record_over(StatefulNet(), dict(specs)), v_seq, psc_seq)
    chunked = _run_in_chunks(
        record_over(StatefulNet(), dict(specs)), v_seq, psc_seq, boundaries=[2, 4]
    )
    for key in single_pass:
        torch.testing.assert_close(single_pass[key], chunked[key], msg=key)


def test_step_kernel_compiles_fullgraph_without_graph_breaks(sequence):
    # the per-step fold + torch.stack over a fixed unroll compiles to one graph.
    v_seq, psc_seq = sequence
    recorder = record_over(
        StatefulNet(),
        {
            "raw": col("neuron.v"),
            "mean": col("neuron.v").mean(),
            "mean_of_diff": col("neuron.v").diff().mean(),
            "std": col("neuron.v").std(),
        },
    )

    def run_fixed_unroll(v_list, carry):
        buffers = {nid: [] for nid in recorder.stack_nids}
        for t in range(T):
            carry, stack = recorder.step_kernel(carry, EagerFrame([v_list[t]]))
            for nid, value in stack.items():
                buffers[nid].append(value)
        return recorder.finalize(carry, buffers)

    eager = run_fixed_unroll(v_seq, recorder.init_carry(EagerFrame([v_seq[0]])))
    torch._dynamo.reset()
    from torch._dynamo.utils import counters

    counters.clear()
    compiled = torch.compile(run_fixed_unroll, fullgraph=True)
    out = compiled(v_seq, recorder.init_carry(EagerFrame([v_seq[0]])))
    breaks = sum(
        (v if isinstance(v, int) else len(v)) for v in counters["graph_break"].values()
    )
    assert breaks == 0, f"unexpected graph breaks: {dict(counters['graph_break'])}"
    for key in eager:
        torch.testing.assert_close(eager[key], out[key], msg=key)


def _run_in_chunks(recorder, v_seq, psc_seq, boundaries):
    """Drive the recorder in chunks, threading the carry across them."""
    net = recorder.resolver.root
    net.neuron.v, net.synapse.psc = v_seq[0], psc_seq[0]
    carry = recorder.init_carry(recorder.resolver.frame())
    buffers = recorder.new_buffers()
    edges = [0, *boundaries, len(v_seq)]
    for lo, hi in zip(edges, edges[1:]):
        for t in range(lo, hi):
            net.neuron.v, net.synapse.psc = v_seq[t], psc_seq[t]
            carry, stack = recorder.step_kernel(carry, recorder.resolver.frame())
            for nid, value in stack.items():
                buffers[nid].append(value)
    return recorder.finalize(carry, buffers)


# ===========================================================================
# Edge cases: rejected specs (each fails loudly at build time)
# ===========================================================================
def test_broadcast_back_is_rejected():
    # a per-step value minus a whole-sequence reduction needs two passes; v1 rejects
    with pytest.raises(NotImplementedError, match="broadcast-back"):
        record_over(StatefulNet(), {"x": col("neuron.v") - col("neuron.v").mean()})


def test_grad_leaf_cannot_be_composed():
    with pytest.raises(TypeError, match="grad"):
        grad("neuron.v").mean()


def test_processed_monitor_requires_an_explicit_name():
    # a reduction/derived monitor has no natural key; the user must .alias() it
    with pytest.raises(ValueError, match="explicit name"):
        record_over(StatefulNet(), [col("neuron.v").mean()])


def test_duplicate_output_key_is_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        record_over(
            StatefulNet(),
            ["neuron.v", col("synapse.psc").mean().alias("neuron.v")],
        )


def test_missing_target_fails_at_build_time():
    with pytest.raises(KeyError):
        record_over(StatefulNet(), ["neuron.does_not_exist"])


def test_map_seq_rejects_a_non_per_step_input():
    with pytest.raises(NotImplementedError, match="per-step"):
        record_over(StatefulNet(), {"x": map_seq(lambda a: a, lit(3.0))})
