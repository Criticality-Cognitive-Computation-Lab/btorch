import pytest
import torch

from btorch.models.base import ParamBufferMixin


class DummyParamModule(ParamBufferMixin):
    """Minimal module used to validate ParamBufferMixin behavior.

    The module intentionally keeps only one configurable field (`weight`) so
    tests can focus on shape policy and checkpoint transitions.
    """

    def __init__(
        self,
        n_neuron: int | tuple[int, ...],
        weight,
        *,
        trainable_param: bool | set[str] = False,
        trainable_shape: str = "auto",
        sizes: tuple[int, ...] | None = None,
    ):
        super().__init__()
        if isinstance(n_neuron, int):
            self.n_neuron = (n_neuron,)
        else:
            self.n_neuron = tuple(n_neuron)
        if isinstance(trainable_param, set):
            self.trainable_param = set(trainable_param)
        else:
            self.trainable_param = set()

        self.def_param(
            "weight",
            weight,
            trainable_param=trainable_param,
            trainable_shape=trainable_shape,
            sizes=sizes,
        )


def test_trailing_dims_not_compacted_when_uniform():
    """Uniform trailing dimensions must be preserved across save/load.

    Trailing dimensions often encode additional channels or modes (for
    example, adaptation components). Even if all values are uniform,
    these dimensions are semantically meaningful and must not be
    collapsed to scalar.
    """

    module = DummyParamModule(
        n_neuron=(2, 3),
        weight=torch.ones(2, 3, 4),
        sizes=(2, 3, 4),
    )
    assert tuple(module.weight.shape) == (2, 3, 4)

    state = module.state_dict()
    assert tuple(state["weight"].shape) == (2, 3, 4)

    # Loading a compact checkpoint value should broadcast back to the full
    # trailing shape, preserving the intended structural dimensions.
    compact_state = {"weight": torch.tensor(5.0)}
    module.load_state_dict(compact_state, strict=False)
    assert tuple(module.weight.shape) == (2, 3, 4)
    assert torch.all(module.weight == 5.0)


def test_uniform_and_nonuniform_checkpoint_transition_for_auto_shape():
    """Auto shape should switch between scalar and full based on checkpoint.

    Expected behavior:
    - loading uniform full tensors should compact to scalar
    - loading non-uniform full tensors should promote storage to full tensor
    """

    target = DummyParamModule(n_neuron=4, weight=20.0)
    assert tuple(target.weight.shape) == ()

    src_uniform = DummyParamModule(n_neuron=4, weight=torch.full((4,), 20.0))
    target.load_state_dict(src_uniform.state_dict())
    assert tuple(target.weight.shape) == ()
    assert torch.allclose(target.weight, torch.tensor(20.0, dtype=target.weight.dtype))

    src_nonuniform = DummyParamModule(
        n_neuron=4, weight=torch.tensor([1.0, 2.0, 3.0, 4.0])
    )
    target.load_state_dict(src_nonuniform.state_dict())
    assert tuple(target.weight.shape) == (4,)
    assert torch.allclose(target.weight, torch.tensor([1.0, 2.0, 3.0, 4.0]))


def test_trainable_shape_scalar_and_full():
    """Trainable shape policy should be explicit and predictable.

    This test demonstrates the intended API:
    - `trainable_shape="scalar"` keeps a learnable scalar
    - `trainable_shape="full"` keeps a learnable per-neuron tensor
    """

    scalar_param = DummyParamModule(
        n_neuron=4,
        weight=2.0,
        trainable_param={"weight"},
        trainable_shape="scalar",
    )
    assert isinstance(scalar_param.weight, torch.nn.Parameter)
    assert tuple(scalar_param.weight.shape) == ()

    full_param = DummyParamModule(
        n_neuron=4,
        weight=2.0,
        trainable_param={"weight"},
        trainable_shape="full",
    )
    assert isinstance(full_param.weight, torch.nn.Parameter)
    assert tuple(full_param.weight.shape) == (4,)


def test_scalar_mode_is_consistent_during_init_and_load():
    """Scalar mode should keep trainable layout consistent on load.

    Consistency contract:
    - initialization with scalar mode requires uniform values
    - trainable scalar rejects non-scalar checkpoint tensors
    - non-trainable scalar can promote to non-scalar on checkpoint load
    """

    with pytest.raises(ValueError):
        DummyParamModule(
            n_neuron=4,
            weight=torch.tensor([1.0, 2.0, 3.0, 4.0]),
            trainable_param={"weight"},
            trainable_shape="scalar",
        )

    scalar_param = DummyParamModule(
        n_neuron=4,
        weight=2.0,
        trainable_param={"weight"},
        trainable_shape="scalar",
    )

    scalar_ckpt = {"weight": torch.tensor(3.0)}
    scalar_param.load_state_dict(scalar_ckpt, strict=False)
    assert tuple(scalar_param.weight.shape) == ()
    assert torch.allclose(scalar_param.weight, torch.tensor(3.0))

    non_scalar_ckpt = {"weight": torch.full((4,), 3.0)}
    with pytest.raises(RuntimeError, match="non-scalar checkpoint tensor"):
        scalar_param.load_state_dict(non_scalar_ckpt, strict=False)

    # Non-trainable scalar mode is allowed to promote shape when checkpoint is
    # non-scalar, which preserves checkpoint information instead of dropping it.
    scalar_buffer = DummyParamModule(
        n_neuron=4,
        weight=2.0,
        trainable_param=False,
        trainable_shape="scalar",
    )
    scalar_buffer.load_state_dict({"weight": torch.tensor([1.0, 2.0, 3.0, 4.0])})
    assert tuple(scalar_buffer.weight.shape) == (4,)
    assert torch.allclose(scalar_buffer.weight, torch.tensor([1.0, 2.0, 3.0, 4.0]))
