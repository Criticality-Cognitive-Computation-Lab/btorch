import os

# Import backends
from .triton import sparse as triton_backend
from .warp import sparse as warp_backend


# Simple backend selection
# Can be controlled via env var BTORCH_BACKEND="triton" or "warp"
BACKEND = os.environ.get("BTORCH_BACKEND", "triton").lower()


def coo_spmv(indices, values, vec, is_bool_float=False, size_m=None):
    if BACKEND == "warp":
        return warp_backend.coo_spmv_warp(indices, values, vec, is_bool_float, size_m)
    else:
        return triton_backend.coo_spmv(indices, values, vec, is_bool_float, size_m)


def coo_spmm(indices, values, mat, is_bool_float=False, size_m=None):
    if BACKEND == "warp":
        return warp_backend.coo_spmm_warp(indices, values, mat, is_bool_float, size_m)
    else:
        return triton_backend.coo_spmm(indices, values, mat, is_bool_float, size_m)
