"""Tests for the standalone recording engine (:mod:`btorch.monitor`), no RNN.

The engine records a *stepped computation* whose named state buffers are the
"columns" and whose timesteps are the "rows".  A :class:`Recorder` compiles a
Polars-style spec (``col("neuron.v").mean()`` etc.) and is driven one frame per
timestep; ``run`` returns the finalised ``{name: tensor}`` records.

Below, ``StatefulNet`` stands in for a real model: two submodules each holding one
state buffer (``neuron.v`` and ``synapse.psc``).  The tests read as usage examples
(top) followed by engine-internal guarantees and rejected specs (bottom).
"""

import platform

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
    map_step,
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
    recorder = record_over(
        StatefulNet(), ["neuron.v", col("synapse.psc").alias("membrane")]
    )
    records = run_steps(recorder, v_seq, psc_seq)
    # a bare dotted string stacks the buffer over time -> [T, B, F];
    # .alias() renames the output key without changing what is recorded
    torch.testing.assert_close(records["neuron.v"], torch.stack(v_seq, 0))
    torch.testing.assert_close(records["membrane"], torch.stack(psc_seq, 0))
    assert recorder.raw_columns == {"neuron.v", "membrane"}


def test_bare_tensor_reference_resolves_like_a_dotted_name(sequence):
    v_seq, psc_seq = sequence
    net = StatefulNet()
    # col(net.neuron.v) identifies the buffer by identity, then records like a name
    recorder = Recorder([col(net.neuron.v).alias("v")], resolver=Resolver(net))
    records = run_steps(recorder, v_seq, psc_seq)
    torch.testing.assert_close(records["v"], torch.stack(v_seq, 0))


def test_scalar_reductions_over_time(sequence):
    v_seq, psc_seq = sequence
    stacked = torch.stack(v_seq, 0)
    recorder = record_over(
        StatefulNet(),
        {
            "mean": col("neuron.v").mean(),
            "sum": col("neuron.v").sum(),
            "last": col("neuron.v").last(),
            "first": col("neuron.v").first(),
            "min": col("neuron.v").min(),
            "max": col("neuron.v").max(),
            "count": col("neuron.v").count(),
        },
    )
    records = run_steps(recorder, v_seq, psc_seq)
    torch.testing.assert_close(records["mean"], stacked.mean(0))
    torch.testing.assert_close(records["sum"], stacked.sum(0))
    torch.testing.assert_close(records["last"], v_seq[-1])
    torch.testing.assert_close(records["first"], v_seq[0])
    torch.testing.assert_close(records["min"], stacked.min(0).values)
    torch.testing.assert_close(records["max"], stacked.max(0).values)
    torch.testing.assert_close(records["count"], torch.tensor(float(T)))


def test_std_var_default_matches_torch_correction_convention(sequence):
    # .std()/.var() default to correction=1 like torch's .std(dim=0)/.var(dim=0);
    # correction=0 gives the population statistic.
    v_seq, psc_seq = sequence
    stacked = torch.stack(v_seq, 0)
    recorder = record_over(
        StatefulNet(),
        {
            "std": col("neuron.v").std(),
            "var": col("neuron.v").var(),
            "std_pop": col("neuron.v").std(correction=0),
            "var_pop": col("neuron.v").var(correction=0),
        },
    )
    records = run_steps(recorder, v_seq, psc_seq)
    torch.testing.assert_close(records["std"], stacked.std(0))
    torch.testing.assert_close(records["var"], stacked.var(0))
    torch.testing.assert_close(records["std_pop"], stacked.std(0, correction=0))
    torch.testing.assert_close(records["var_pop"], stacked.var(0, correction=0))


def test_per_step_expressions_over_columns_and_scalars(sequence):
    v_seq, psc_seq = sequence
    # elementwise over two state buffers and against scalars: evaluated per
    # timestep, stacked over time
    recorder = record_over(
        StatefulNet(),
        {
            "drive": col("neuron.v") - col("synapse.psc"),
            "shifted": col("neuron.v") + 1,
        },
    )
    records = run_steps(recorder, v_seq, psc_seq)
    torch.testing.assert_close(
        records["drive"], torch.stack(v_seq, 0) - torch.stack(psc_seq, 0)
    )
    torch.testing.assert_close(records["shifted"], torch.stack(v_seq, 0) + 1)


def test_elementwise_math_matches_torch(sequence):
    # the unary/binary math surface against the same ops on the stacked trace
    v_seq, psc_seq = sequence
    stacked = torch.stack(v_seq, 0)
    recorder = record_over(
        StatefulNet(),
        {
            "absed": abs(col("neuron.v")),
            "relued": col("neuron.v").relu(),
            "clamped": col("neuron.v").clamp(-0.5, 0.5),
            "squared": col("neuron.v") ** 2,
            "halved": col("neuron.v") / 2,
        },
    )
    records = run_steps(recorder, v_seq, psc_seq)
    torch.testing.assert_close(records["absed"], stacked.abs())
    torch.testing.assert_close(records["relued"], stacked.relu())
    torch.testing.assert_close(records["clamped"], stacked.clamp(-0.5, 0.5))
    torch.testing.assert_close(records["squared"], stacked.pow(2))
    torch.testing.assert_close(records["halved"], stacked / 2)


def test_tensor_literal_broadcasts_like_a_torch_operand(sequence):
    # a tensor literal joins the graph as a constant and broadcasts per step
    v_seq, psc_seq = sequence
    bias = torch.tensor([10.0, 20.0, 30.0])
    recorder = record_over(StatefulNet(), {"biased": col("neuron.v") + lit(bias)})
    records = run_steps(recorder, v_seq, psc_seq)
    torch.testing.assert_close(records["biased"], torch.stack(v_seq, 0) + bias)


def test_min_max_work_on_integer_spike_buffers():
    # spike counts are integer buffers; min/max accumulate +-inf so they
    # promote their accumulator to float instead of crashing (regression).
    net = StatefulNet()
    torch.manual_seed(0)
    spk_seq = [torch.randint(0, 4, (BATCH, FEATURES)) for _ in range(T)]
    recorder = Recorder(
        {"lo": col("synapse.psc").min(), "hi": col("synapse.psc").max()},
        resolver=Resolver(net),
    )
    frames = []
    for spk in spk_seq:
        net.synapse.psc = spk
        frames.append(recorder.resolver.frame())
    records = recorder.run(frames)
    stacked = torch.stack([s.float() for s in spk_seq], 0)
    assert records["lo"].dtype.is_floating_point
    torch.testing.assert_close(records["lo"], stacked.min(0).values)
    torch.testing.assert_close(records["hi"], stacked.max(0).values)


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


def test_shift_lags_with_zero_fill_not_wraparound(sequence):
    # shift(n) is the value from n steps earlier; warmup is zero-filled
    # (torch.roll would wrap around, torch.diff-style ops would shrink T).
    v_seq, psc_seq = sequence
    stacked = torch.stack(v_seq, 0)
    recorder = record_over(
        StatefulNet(),
        {
            "lag1": col("neuron.v").shift(),
            "lag3_mean": col("neuron.v").shift(3).mean(),  # denom = T-3, not T
        },
    )
    records = run_steps(recorder, v_seq, psc_seq)
    expected = torch.zeros_like(stacked)
    expected[1:] = stacked[:-1]
    torch.testing.assert_close(records["lag1"], expected)
    # the mean only divides by valid rows (T-3), i.e. the values x_0..x_{T-4}
    torch.testing.assert_close(records["lag3_mean"], stacked[: T - 3].mean(0))


def test_first_last_under_a_windowed_input_skip_warmup(sequence):
    # first/last are validity-gated: on a diff() input they take the first/last
    # VALID difference, not the zero-filled warmup rows.
    v_seq, psc_seq = sequence
    stacked = torch.stack(v_seq, 0)
    diffs = stacked[1:] - stacked[:-1]
    recorder = record_over(
        StatefulNet(),
        {
            "first_diff": col("neuron.v").diff().first(),
            "last_diff": col("neuron.v").diff().last(),
        },
    )
    records = run_steps(recorder, v_seq, psc_seq)
    torch.testing.assert_close(records["first_diff"], diffs[0])
    torch.testing.assert_close(records["last_diff"], diffs[-1])


def test_map_step_applies_fn_per_step_across_columns(sequence):
    v_seq, psc_seq = sequence
    # map_step is the streamable custom boundary: fn sees each step's [B, N]
    # slices (map_seq sees the whole [T, B, N] stack instead).
    recorder = record_over(
        StatefulNet(),
        {"gain": map_step(lambda a, b: a + b, col("neuron.v"), col("synapse.psc"))},
    )
    records = run_steps(recorder, v_seq, psc_seq)
    torch.testing.assert_close(
        records["gain"], torch.stack(v_seq, 0) + torch.stack(psc_seq, 0)
    )


def test_map_seq_runs_a_custom_fn_on_the_whole_sequence(sequence):
    v_seq, psc_seq = sequence
    # map_seq materialises [T, B, F] and applies fn once (needed for non-streaming
    # ops like median); it can be nested under further arithmetic (not just a root)
    recorder = record_over(
        StatefulNet(),
        {
            "median": map_seq(lambda V: V.median(0).values, col("neuron.v")),
            "twice_sum": map_seq(lambda V: V.sum(0), col("neuron.v")) * 2,
        },
    )
    records = run_steps(recorder, v_seq, psc_seq)
    stacked = torch.stack(v_seq, 0)
    torch.testing.assert_close(records["median"], stacked.median(0).values)
    torch.testing.assert_close(records["twice_sum"], stacked.sum(0) * 2)


def test_grad_specs_materialise_nothing_but_share_resolution(sequence):
    # grad(...) is recorded via backward hooks by the consumer; the engine only
    # exposes it as a grad spec.  A grad-only spec emits NO source/stack node:
    # no [T, ...] value trace is materialised unless col("v")/"v" is also
    # requested.  When both are requested they resolve to ONE TargetRef
    # (Resolver is idempotent per name), so values+grads cost a single
    # per-step read.
    net = StatefulNet()
    grad_only = record_over(StatefulNet(), {"v_grad": grad("neuron.v")})
    assert grad_only.has_grad_specs
    assert grad_only.grad_specs[0][0] == "v_grad"
    assert not grad_only.is_empty  # the grad spec alone still counts as work
    assert grad_only.raw_columns == set()
    assert list(grad_only.stack_nids) == []  # nothing to materialise

    both = record_over(net, [{"v": col("neuron.v"), "v_grad": grad("neuron.v")}])
    assert both.resolver.n_refs == 1
    assert both.raw_columns == {"v"}
    assert len(both.graph.grad_specs) == 1
    v_seq, psc_seq = sequence
    records = run_steps(both, v_seq, psc_seq)
    torch.testing.assert_close(records["v"], torch.stack(v_seq, 0))


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


@pytest.mark.skipif(platform.system() != "Linux", reason="torch.compile: Linux only")
def test_validate_reducer_accepts_a_pure_reducer():
    # validate_reducer traces the reducer with torch.compile(fullgraph=True)
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


class _BadInPlace(Reducer):
    """Folds value into the carry in place -- pure-looking eager, corrupt
    later."""

    def init(self, example):
        return torch.zeros_like(example)

    def update(self, carry, value, valid):
        carry += value
        return carry

    def finalize(self, carry):
        return carry


def test_fold_is_purity_checked_at_build_time():
    # building a recorder runs the cheap invariant checks automatically: an
    # in-place reducer is rejected where the mistake is made, not silently
    # under checkpoint/cudagraph later.
    with pytest.raises(RuntimeError, match="in place"):
        record_over(StatefulNet(), {"m": col("neuron.v").fold(_BadInPlace())})


def test_auto_validate_false_opts_out_of_build_time_checks(sequence):
    class _Unchecked(_BadInPlace):
        auto_validate = False  # exotic reducers may legitimately opt out

    v_seq, psc_seq = sequence
    recorder = record_over(StatefulNet(), {"m": col("neuron.v").fold(_Unchecked())})
    records = run_steps(recorder, v_seq, psc_seq)
    # eager results look right (the corruption is mode-specific) -- opting out
    # is on your head; validate_reducer remains as the explicit deep check.
    torch.testing.assert_close(records["m"], torch.stack(v_seq, 0).sum(0))


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


@pytest.mark.skipif(platform.system() != "Linux", reason="torch.compile: Linux only")
def test_validate_reducer_ignores_opaque_object_state():
    # An attribute of an unknown type cannot be compared by value soundly
    # (repr may embed volatile state, e.g. ids or counters); the snapshot skips
    # it instead of risking a false "mutated" error.  Regression guard: this
    # used to raise because repr() changed between calls.

    class _Opaque:
        """An object whose repr is unstable across calls."""

        def __init__(self):
            self.ticks = 0

        def __repr__(self):  # deliberately volatile
            self.ticks += 1
            return f"<Opaque call#{self.ticks}>"

    class _Volatile(Reducer):
        def __init__(self):
            self.opaque = _Opaque()  # repr changes even when untouched

        def init(self, example):
            return torch.zeros_like(example)

        def update(self, carry, value, valid):
            return carry + value

        def finalize(self, carry):
            return carry

    validate_reducer(_Volatile(), torch.randn(BATCH, FEATURES))


def test_validate_reducer_rejects_container_state_mutation():
    # Tracked containers (deep-copied at snapshot) are still checked: appending
    # per-step data to a list on self must be rejected like a scalar counter.

    class _BadList(Reducer):
        def __init__(self):
            self.seen = []  # mutable container -> still caught

        def init(self, example):
            return torch.zeros_like(example)

        def update(self, carry, value, valid):
            self.seen.append(value)
            return carry + value

        def finalize(self, carry):
            return carry

    with pytest.raises(RuntimeError, match="its own state"):
        validate_reducer(_BadList(), torch.randn(BATCH, FEATURES))


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


def test_reducer_with_changing_carry_structure_is_rejected_twice(sequence):
    # layer 1: the build-time purity check rejects it statically...
    with pytest.raises(RuntimeError, match="structure"):
        record_over(StatefulNet(), {"g": col("neuron.v").fold(_StructureChanging())})

    # ...layer 2 (defence in depth for opt-out reducers): the runtime fold
    # re-checks the carry pytree per step.
    class _Unchecked(_StructureChanging):
        auto_validate = False

    recorder = record_over(StatefulNet(), {"g": col("neuron.v").fold(_Unchecked())})
    with pytest.raises(ValueError, match="constant"):
        run_steps(recorder, *sequence)


# ===========================================================================
# Engine-internal guarantees (memory, sharing, chunking, compile)
# ===========================================================================
def test_graph_shape_streams_shares_and_levels():
    # one test telling the whole IR story: a reduction folds into an O(1)
    # carry (no [T, ...] buffer), shared subexpressions intern to a single
    # node (CSE), bare columns are STEP-level stack points while reductions
    # are AGG.
    recorder = record_over(StatefulNet(), {"v_mean": col("neuron.v").mean()})
    assert recorder.stack_nids == ()
    assert recorder.raw_columns == set()

    graph = build(
        [col("neuron.v").mean().alias("a"), col("neuron.v").sum().alias("b")],
        Resolver(StatefulNet()),
    )
    assert len([n for n in graph.nodes if n.op == "source"]) == 1

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


def test_init_carry_reinfers_when_state_shapes_change():
    # The carry layout is cached from the first example; a consumer that resets
    # its state with a new batch size must get a fresh-shaped carry, not a
    # stale one (a stale shape can silently broadcast in reductions).
    recorder = record_over(StatefulNet(), {"m": col("neuron.v").mean()})
    net = recorder.resolver.root

    def sum_shape(carry):
        # the mean's accumulator slot -- its only [B, F]-shaped carry entry
        return next(v.shape for v in carry.values() if v.dim() == 2)

    net.neuron.v = torch.randn(2, FEATURES)
    carry_b2 = recorder.init_carry(recorder.resolver.frame())
    assert sum_shape(carry_b2) == (2, FEATURES)

    net.neuron.v = torch.randn(5, FEATURES)
    carry_b5 = recorder.init_carry(recorder.resolver.frame())
    assert sum_shape(carry_b5) == (5, FEATURES)

    # same shapes again -> hot path reuses the cached layout
    net.neuron.v = torch.randn(2, FEATURES)
    assert sum_shape(recorder.init_carry(recorder.resolver.frame())) == (2, FEATURES)


@pytest.mark.skipif(platform.system() != "Linux", reason="torch.compile: Linux only")
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


def test_grad_leaf_cannot_be_a_binary_operand():
    # the guard must check BOTH sides: a grad on the right used to slip past
    # the DSL and only fail later at IR build, far from the offending line
    with pytest.raises(TypeError, match="grad"):
        col("neuron.v") * grad("neuron.v")
    with pytest.raises(TypeError, match="grad"):
        2.0 / col("neuron.v") + grad("neuron.v")


def test_grad_leaf_cannot_be_a_map_target():
    with pytest.raises(TypeError, match="grad"):
        map_step(lambda a, b: a + b, col("neuron.v"), grad("neuron.v"))


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
