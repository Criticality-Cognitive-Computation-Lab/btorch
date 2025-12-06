"""Custom parameter constraint system that stores constrained values as
buffers.

Unlike torch.parametrize, this requires explicit constrain() calls and
caches the constrained value as a buffer to avoid recomputation.
"""

# TODO: support ParametrizationList
# TODO: support inject forward

import abc
from collections.abc import Sequence

import torch
from torch import nn
from torch.__future__ import get_swap_module_params_on_conversion
from torch.nn import Parameter
from torch.utils._python_dispatch import is_traceable_wrapper_subclass


def _maybe_set(dest: torch.Tensor, src: torch.Tensor) -> None:
    should_swap = (
        get_swap_module_params_on_conversion() or is_traceable_wrapper_subclass(dest)
    )
    if should_swap:
        if isinstance(dest, Parameter) and not isinstance(src, Parameter):
            src = Parameter(src, requires_grad=dest.requires_grad)
        torch.utils.swap_tensors(dest, src)
    else:
        dest.set_(src)  # type: ignore[call-overload]


def _register_parameter_or_buffer(module, name, X):
    if isinstance(X, Parameter):
        module.register_parameter(name, X)
    else:
        module.register_buffer(name, X)


class Constraint(nn.Module, abc.ABC):
    """Base class for parameter constraints."""

    forward_does_nothing: bool = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Transform unconstrained parameter to constrained space."""
        return x

    @abc.abstractmethod
    def constrain(self, x: torch.Tensor) -> torch.Tensor: ...

    def right_inverse(self, x: torch.Tensor) -> torch.Tensor:
        return self.constrain(x)


class PositiveConstraint(Constraint):
    """Constraint to keep parameters positive."""

    def __init__(self, eps: float = 1e-6):
        self.eps = eps

    def constrain(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.softplus(x) + self.eps


class UnitNormConstraint(Constraint):
    """Constraint to normalize parameters to unit norm."""

    def __init__(self, dim: int = -1, eps: float = 1e-8):
        self.dim = dim
        self.eps = eps

    def constrain(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.normalize(x, p=2, dim=self.dim, eps=self.eps)


class BoundConstraint(Constraint):
    """Constraint to bound parameters within a range."""

    def __init__(self, lower: float = 0.0, upper: float = 1.0):
        self.lower = lower
        self.upper = upper

    def constrain(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(x) * (self.upper - self.lower) + self.lower


def register_constraint(
    module: nn.Module, param_name: str, constraint: Constraint, unsafe: bool = False
) -> None:
    """Register a constraint on a parameter of a module.

    This modifies the module in-place:
    - Renames the original parameter to {param_name}_unconstrained
    - Registers a buffer {param_name} to store the constrained value
    - Stores constraint metadata in module._constraint_info

    Args:
        module: The module containing the parameter
        param_name: Name of the parameter to constrain
        constraint: Constraint object to apply
        unsafe: If True, allow replacing existing constraints (default: False)
    """
    if not hasattr(module, param_name):
        raise ValueError(f"Module has no parameter '{param_name}'")

    if not isinstance(getattr(module, param_name), nn.Parameter):
        raise ValueError(f"'{param_name}' is not a Parameter")

    # Initialize constraint info storage if needed
    if not hasattr(module, "_constraint_info"):
        module._constraint_info = {}

    # Check if already constrained
    if param_name in module._constraint_info and not unsafe:
        raise ValueError(
            f"Parameter '{param_name}' already has a constraint. "
            f"Use unsafe=True to replace it."
        )

    # Get the original parameter
    original_param = getattr(module, param_name)

    # Store constraint info
    unconstrained_name = f"{param_name}_unconstrained"
    module._constraint_info[param_name] = {
        "constraint": constraint,
        "unconstrained_name": unconstrained_name,
    }

    if not constraint.forward_does_nothing:
        # Remove the original parameter from _parameters
        delattr(module, param_name)

        # Register the unconstrained parameter
        module.register_parameter(unconstrained_name, original_param)

        # Apply constraint and register as buffer
        constrained_value = constraint.constrain(original_param)

        module.register_buffer(param_name, constrained_value, persistent=False)
    else:
        module._constraint_info[param_name]["unconstrained_name"] = param_name


def constrain(
    module: nn.Module, param_names: str | Sequence[str] | None = None
) -> None:
    """Apply constraints to update buffer values from unconstrained parameters.

    This should be called after optimizer.step() to update the constrained buffers.

    Args:
        module: The module with constrained parameters
        param_names: Specific parameter name(s) to constrain, or None for all
    """
    if not hasattr(module, "_constraint_info"):
        return

    # Determine which parameters to constrain
    if param_names is None:
        params_to_constrain = list(module._constraint_info.keys())
    elif isinstance(param_names, str):
        params_to_constrain = [param_names]
    else:
        params_to_constrain = list(param_names)

    # Apply constraints
    with torch.no_grad():
        for param_name in params_to_constrain:
            if param_name not in module._constraint_info:
                raise ValueError(f"No constraint registered for '{param_name}'")

            info = module._constraint_info[param_name]
            unconstrained_param = getattr(module, info["unconstrained_name"])
            constrained_value = info["constraint"].constrain(unconstrained_param.data)

            # Update the buffer in-place
            buffer = getattr(module, param_name)
            _maybe_set(buffer, constrained_value)


def constrain_net(module: nn.Module) -> None:
    """Apply all constraints in a module (and optionally its submodules).

    Args:
        module: The root module
    """
    for m in module.modules():
        constrain(m)


def remove_constraint(module: nn.Module, param_name: str) -> None:
    """Remove a constraint from a parameter.

    This restores the parameter to its unconstrained state and removes the buffer.

    Args:
        module: The module containing the constrained parameter
        param_name: Name of the constrained parameter
    """
    if (
        not hasattr(module, "_constraint_info")
        or param_name not in module._constraint_info
    ):
        raise ValueError(f"No constraint registered for '{param_name}'")

    info = module._constraint_info[param_name]
    unconstrained_name = info["unconstrained_name"]

    # Get the unconstrained parameter
    unconstrained_param = getattr(module, unconstrained_name)

    if not info["constraint"].forward_does_nothing:
        # Remove the buffer
        delattr(module, param_name)

        # Remove the unconstrained parameter
        delattr(module, unconstrained_name)

        # Re-register as a normal parameter with the original name
        module.register_parameter(param_name, unconstrained_param)

    # Clean up constraint info
    del module._constraint_info[param_name]


# Example usage
if __name__ == "__main__":
    # Create a simple model
    model = nn.Linear(10, 5)

    # Register constraints
    register_constraint(model, "weight", UnitNormConstraint(dim=1))
    register_constraint(model, "bias", PositiveConstraint())

    # Setup optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Training loop
    for step in range(3):
        # Forward pass (uses constrained buffers)
        x = torch.randn(2, 10)
        output = model(x)
        loss = output.sum()

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        print(f"\nStep {step}")
        print(f"Weight norm before constrain: {model.weight.norm(dim=1)}")
        print(f"Bias min before constrain: {model.bias.min().item():.4f}")

        # Update unconstrained parameters
        optimizer.step()

        # EXPLICITLY apply constraints to update buffers
        constrain_net(model)

        print(f"Weight norm after constrain: {model.weight.norm(dim=1)}")
        print(f"Bias min after constrain: {model.bias.min().item():.4f}")
