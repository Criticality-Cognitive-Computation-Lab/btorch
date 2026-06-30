class SparseError(RuntimeError):
    """Base error for sparse matrix execution."""


class BackendUnavailableError(SparseError):
    """Raised when an explicitly selected backend is not installed."""


class UnsupportedCapabilityError(SparseError):
    """Raised when no backend supports the requested operation."""
