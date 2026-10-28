"""Construct original PyTorch neurons without sharing generated equations."""

import torch

from ... import surrogate


def neuron_module(cls, spec, neurons, **kwargs):
    functions = {
        "sigmoid": surrogate.Sigmoid,
        "atan": surrogate.ATan,
        "atan_approx": surrogate.ATanApprox,
        "erf": surrogate.Erf,
        "superspike": surrogate.SuperSpike,
        "triangle": surrogate.Triangle,
        "poisson": surrogate.PoissonRandomSpike,
    }
    options = (
        {
            "spiking": spec.poisson.spiking,
            "tau": spec.poisson.tau,
            "rho": spec.poisson.rho,
            "leak": spec.poisson.leak,
            "k": spec.poisson.k,
            "damping_factor": spec.damping,
        }
        if spec.poisson is not None
        else {"alpha": spec.alpha, "damping_factor": spec.damping}
    )
    function = functions[spec.surrogate](**options)
    if spec.surrogate_width is not None:
        function = surrogate.Truncated(function, spec.surrogate_width)
    if spec.refractory:
        kwargs["tau_ref"] = 1.0
    return cls(
        neurons,
        backend="torch",
        step_mode="s",
        surrogate_function=function,
        hard_reset=spec.hard_reset,
        detach_reset=spec.detach_reset,
        device="cpu",
        dtype=torch.float32,
        **kwargs,
    )
