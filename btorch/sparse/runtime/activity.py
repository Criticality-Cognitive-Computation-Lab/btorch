"""Invocation-owned sparse gradient activity shared by producers and
consumers."""

from dataclasses import dataclass

from torch import Tensor


BLOCK_SIZE = 128


@dataclass(frozen=True)
class GradientList:
    """View a trajectory-owned compact-support tape for one timestep.

    Keep the support mask for dense consumers and two block-local lists:
    near-threshold rows, and their union with fired rows needed for weight
    gradients. Counts are device-resident and every producer owns its block.
    """

    tape: Tensor
    n_lanes: int

    @staticmethod
    def size(n_lanes: int) -> int:
        blocks = (n_lanes + BLOCK_SIZE - 1) // BLOCK_SIZE
        return 3 * blocks * BLOCK_SIZE + 2 * blocks

    @property
    def capacity(self) -> int:
        return ((self.n_lanes + BLOCK_SIZE - 1) // BLOCK_SIZE) * BLOCK_SIZE

    @property
    def mask(self) -> Tensor:
        return self.tape[: self.n_lanes]

    def work(self, weight_gradient: bool) -> tuple[Tensor, Tensor]:
        capacity = self.capacity
        blocks = capacity // BLOCK_SIZE
        offset = int(weight_gradient)
        return (
            self.tape[(1 + offset) * capacity : (2 + offset) * capacity],
            self.tape[
                3 * capacity + offset * blocks : 3 * capacity + (offset + 1) * blocks
            ],
        )
