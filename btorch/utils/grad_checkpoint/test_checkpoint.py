# adapted from https://github.com/pytorch/pytorch/issues/96136

# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.
"""Test checkpoint with variable buffers."""

from typing import Optional

import torch
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper as torch_checkpoint_wrapper,  # noqa: F401
)
from torch.nn import Linear, Sequential
from torch.optim import SGD

from ...models import environ
from ...models.base import MemoryModule
from ...models.functional import init_net_state
from .checkpoint import checkpoint_wrapper


class TestModule(MemoryModule):
    def __init__(self, sizes):
        super().__init__()
        self.register_memory("v", 0, sizes)

    # @environ.context(**environ.all())
    def forward(self, x):
        # breakpoint()
        self.v = self.v + x * environ.get("dt")
        return self.v


def get_model(checkpointed, sizes):
    assert checkpointed in [True, False], checkpointed
    model = Sequential(TestModule(sizes), Linear(3, 2))
    init_net_state(model)

    if checkpointed:
        model = checkpoint_wrapper(model)
        # model = torch_checkpoint_wrapper(model)

    model = torch.compile(model)

    return model


# copied from https://github.com/facebookresearch/fairscale/blob/146f160241651e1211c4247979f159a4ef43b54a/fairscale/fair_dev/testing/testing.py#L480
def objects_are_equal(
    a,
    b,
    raise_exception: bool = False,
    dict_key: Optional[str] = None,
    rtol: Optional[float] = None,
    atol: Optional[float] = None,
) -> bool:
    """Test that two objects are equal.

    Tensors are compared to ensure matching size, dtype, device and
    values.
    """
    if type(a) is not type(b):
        if raise_exception:
            raise ValueError(f"type mismatch {type(a)} vs. {type(b)}")
        return False
    if isinstance(a, dict):
        if set(a.keys()) != set(b.keys()):
            if raise_exception:
                raise ValueError(f"keys mismatch {a.keys()} vs. {b.keys()}")
            return False
        for k in a.keys():
            if not objects_are_equal(a[k], b[k], raise_exception, k):
                return False
        return True
    elif isinstance(a, (list, tuple, set)):
        if len(a) != len(b):
            if raise_exception:
                raise ValueError(f"length mismatch {len(a)} vs. {len(b)}")
            return False
        return all(objects_are_equal(x, y, raise_exception) for x, y in zip(a, b))
    elif torch.is_tensor(a):
        try:
            # assert_close doesn't strictly test shape, dtype and device
            shape_dtype_device_match = (
                a.size() == b.size() and a.dtype == b.dtype and a.device == b.device
            )
            if not shape_dtype_device_match:
                if raise_exception:
                    msg = f"sizes: {a.size()} vs. {b.size()}, "
                    msg += f"types: {a.dtype} vs. {b.dtype}, "
                    msg += f"device: {a.device} vs. {b.device}"
                    raise AssertionError(msg)
                else:
                    return False
            torch.testing.assert_close(a, b, rtol=rtol, atol=atol)
            return True
        except (AssertionError, RuntimeError) as e:
            if raise_exception:
                if dict_key and isinstance(e, AssertionError):
                    # Add dict key to the assertion error.
                    msg = e.args[0]
                    new_msg = f"For dict key '{dict_key}': {msg}"
                    raise AssertionError(new_msg) from None
                else:
                    raise e
            else:
                return False
    else:
        return a == b


def test_checkpointed_variable_buffer(device):
    # Get input, ref, checkpoint models and make them equal.
    sizes = (2, 2, 3, 3)
    in_data = torch.rand(*sizes).to(device)
    # # these match
    # m_ref = get_model(True, sizes).to(device)
    # m_cpt = get_model(True, sizes).to(device)
    # # these match
    # m_ref = get_model(True, sizes).to(device)
    # m_cpt = get_model(True, sizes).to(device)

    m_ref = get_model(False, sizes).to(device)
    m_cpt = get_model(True, sizes).to(device)
    m_cpt.load_state_dict(m_ref.state_dict())

    assert objects_are_equal(m_ref.state_dict(), m_cpt.state_dict())

    # Needed due to checkpointing.
    in_data.requires_grad = True
    for model in (m_ref, m_cpt):
        optim = SGD(model.parameters(), lr=0.1)
        with environ.context(dt=2.0):
            out = model(in_data)
        out.sum().backward()
        optim.step()

    assert objects_are_equal(m_ref.state_dict(), m_cpt.state_dict())


test_checkpointed_variable_buffer("cuda")
