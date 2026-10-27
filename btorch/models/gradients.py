"""Configure recurrent gradient algorithms independently of kernel backends."""

from dataclasses import dataclass


@dataclass(frozen=True)
class BPTT:
    """Differentiate the discrete recurrent trajectory through time.

    Args:
        sparse: Exploit exact zeros in the selected surrogate's support when
            the connection and recorded state dependencies permit it. This
            does not truncate a surrogate or suppress physical-state adjoints.
            Ineligible products retain the full BPTT calculation.
    """

    sparse: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.sparse, bool):
            raise TypeError("BPTT.sparse must be a bool")
