from __future__ import annotations

from abc import abstractmethod

from .base import SparseTensorBase


class ProceduralSparseMatrix(SparseTensorBase):
    """Base for deterministic, non-materialized sparse matrices.

    Procedural implementations own compact parameters (shape, seed,
    etc.) and must regenerate the same logical matrix on demand. They
    cannot be used directly with backends until materialized via
    to_csr().
    """

    seed: int

    def mm(self, x):
        return self.to_csr().mm(x)

    def to_coo(self):
        return self.to_csr().to_coo()

    def to_csc(self):
        return self.to_csr().to_csc()

    def to_ell(self):
        return self.to_csr().to_ell()

    def logical_edges(self):
        return self.to_csr().logical_edges()

    def nnz(self) -> int:
        return self.to_csr().nnz()

    def effective_values(self):
        return self.to_csr().effective_values()

    @abstractmethod
    def to_csr(self):
        """Materialize the procedural matrix as stored CSR."""
        raise NotImplementedError
