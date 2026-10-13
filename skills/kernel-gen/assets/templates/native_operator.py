"""Show independent inference and training operators with Torch registration.

Replace this affine smoke-test computation with the confirmed model's
kernel. The toy is intentionally not a neuroscience model or numerical
oracle.
"""

import torch
from torch import Tensor

from btorch.models.kernels.native import KernelBinding


@torch.library.custom_op("btorch_kernel_example::inference", mutates_args=())
def inference(x: Tensor, gain: Tensor) -> Tensor:
    return (x * gain).clone()


@inference.register_fake
def inference_fake(x: Tensor, gain: Tensor) -> Tensor:
    return (x * gain).clone()


@torch.library.custom_op("btorch_kernel_example::training", mutates_args=())
def training(x: Tensor, gain: Tensor) -> Tensor:
    return (x * gain).clone()


@training.register_fake
def training_fake(x: Tensor, gain: Tensor) -> Tensor:
    return (x * gain).clone()


def setup_context(ctx, inputs, output) -> None:
    x, gain = inputs
    ctx.save_for_backward(x, gain)


def backward(ctx, grad_output: Tensor) -> tuple[Tensor | None, Tensor | None]:
    x, gain = ctx.saved_tensors
    grad_x = (
        (grad_output * gain).sum_to_size(x.shape) if ctx.needs_input_grad[0] else None
    )
    grad_gain = (
        (grad_output * x).sum_to_size(gain.shape) if ctx.needs_input_grad[1] else None
    )
    return grad_x, grad_gain


training.register_autograd(backward, setup_context=setup_context)


def run(x: Tensor, gain: Tensor) -> Tensor:
    """Select the whole invocation's path without using Module training
    mode."""
    if torch.is_grad_enabled() and (x.requires_grad or gain.requires_grad):
        return training(x, gain)
    return inference(x, gain)


# Durable OOT package entrypoint; Torch retains operator/autograd ownership.

KERNEL_BINDINGS = {"affine": KernelBinding(inference=inference, training=training)}
