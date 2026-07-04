"""Backend entry points with optional heavy backend imports."""


def coo_spmm(*args, **kwargs):
    """Dispatch COO SpMM through the configured sparse backend."""

    from .sparse import coo_spmm as _coo_spmm

    return _coo_spmm(*args, **kwargs)


def coo_spmv(*args, **kwargs):
    """Dispatch COO SpMV through the configured sparse backend."""

    from .sparse import coo_spmv as _coo_spmv

    return _coo_spmv(*args, **kwargs)


__all__ = ["coo_spmm", "coo_spmv"]
