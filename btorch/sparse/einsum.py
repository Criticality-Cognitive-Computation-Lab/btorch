"""Experimental Einstein summation with one N-D sparse operand.

:func:`einsum` contracts one sparse array with dense tensors and returns a
dense tensor. It is a gather / contract / scatter lowering over the COO entry
list, not a sparse tensor compiler: see ``docs/en/docs/design/
sparse_einsum_notes.md`` for what the general problem needs.
"""

from __future__ import annotations

import functools
import math
import string

import torch
from torch import Tensor

from .base import Sparse
from .coo import _ravel


def _parse(subscripts: str, n_operands: int) -> tuple[list[str], str]:
    """Split ``subscripts`` into one term per operand and the output term."""
    if not isinstance(subscripts, str):
        raise TypeError(
            "sparse.einsum needs the subscripts as a string, e.g. "
            f"einsum('ij,j->i', A, x); got {type(subscripts).__name__}."
        )
    spec = subscripts.replace(" ", "")
    if "." in spec:
        raise NotImplementedError(
            f"sparse.einsum does not support an ellipsis ('...') in {subscripts!r}; "
            "name every dimension explicitly."
        )
    lhs, arrow, out = spec.partition("->")
    if "->" in out:
        raise ValueError(f"{subscripts!r} contains more than one '->'.")
    terms = lhs.split(",")
    for term in [*terms, out]:
        if not all(c in string.ascii_letters for c in term):
            raise ValueError(
                f"Subscripts must be ASCII letters, got {term!r} in {subscripts!r}."
            )
    if len(terms) != n_operands:
        raise ValueError(
            f"{subscripts!r} has {len(terms)} input term(s) but {n_operands} "
            "operand(s) were given."
        )
    seen = "".join(terms)
    if not arrow:
        # Implicit mode (NumPy / PyTorch): every letter that appears exactly
        # once, in alphabetical order with upper case before lower case.
        return terms, "".join(sorted(c for c in set(seen) if seen.count(c) == 1))
    if len(set(out)) != len(out):
        raise ValueError(f"An output subscript appears more than once in {out!r}.")
    missing = sorted(set(out) - set(seen))
    if missing:
        raise ValueError(
            f"Output subscript(s) {missing} of {subscripts!r} do not appear in "
            "any input term."
        )
    return terms, out


def einsum(subscripts: str, A: Sparse, *operands: Tensor) -> Tensor:
    """Contract one sparse array with dense tensors (experimental).

    The result equals ``torch.einsum(subscripts, A.to_dense(), *operands)``
    without ever building ``A.to_dense()``:

    .. math::
        Y_{\\text{out}} = \\sum_{\\text{letters not in out}}
            A_{\\text{sub}_0} \\prod_k X^{(k)}_{\\text{sub}_k}

    **Scope.** Exactly one sparse operand, which comes first; every other
    operand is a dense tensor (there may be none, e.g. ``"ii->i"``); the
    output is always dense. Sparse times sparse and sparse outputs are not
    implemented. This function is experimental and may change.

    Every subscript letter of ``A`` names one logical axis of
    ``A.shape == (*batch, *sparse, *dense)``. A letter is *indexed* if its
    axis is stored as coordinates in ``A.tocoo().indices()`` (the sparse
    dimensions and the batch dimensions of a different-pattern batch), and
    *non-indexed* if it is an axis of ``values`` (the batch dimensions of a
    shared-pattern batch and the dense dimensions).

    **Algorithm** (``E`` is the number of stored entries):

    1. *Diagonal filter*: if an indexed letter is repeated in ``A``'s
       subscript (``"ii->i"``), keep only the entries whose coordinates agree.
    2. *Gather*: index every dense operand with the coordinates of the indexed
       letters it shares with ``A``; those axes collapse into one entry axis,
       giving ``[E, *other axes]``. Operands sharing no indexed letter are
       left alone.
    3. *Contract*: one ``torch.einsum`` over the values, the gathered operands
       and the entry axis, producing per-entry contributions
       ``[E, *non-indexed output letters]``. Indexed letters missing from the
       output are summed here together with the entry axis.
    4. *Scatter*: ``index_add`` the contributions at the linearised
       coordinates of the indexed letters that survive in the output, then
       reshape and permute to the requested order. Duplicate entries add up.
       If no indexed letter survives, step 3 already produces the result.

    **Cost.** Time and memory are proportional to ``E`` and never to
    ``prod(sparse_shape)``, except for the dense output itself. The peak
    temporaries are the gathered operands and the per-entry contributions,
    each ``O(E * prod(sizes of the non-indexed letters it carries))``;
    ``torch.einsum`` may hold pairwise intermediates of the same form, bounded
    by ``E`` times the sizes of all non-indexed letters of the expression.

    Args:
        subscripts: Einsum expression with single-letter subscripts, with or
            without ``->`` (implicit output: the letters that appear once,
            sorted). An ellipsis is not supported.
        A: Sparse array in any format; its term must have ``A.ndim`` letters.
        *operands: Dense tensors, one per remaining term.

    Returns:
        Dense tensor with one axis per output letter. Its dtype is the
        promotion of all operand dtypes (``torch.einsum`` would raise on
        mixed dtypes), so integer and bool operands may be passed directly.

    Raises:
        NotImplementedError: For an ellipsis, a second sparse operand, or a
            letter repeated in ``A`` on both an indexed and a non-indexed axis.
        ValueError: For malformed subscripts, a term whose length differs
            from the operand's number of dimensions, or a letter used with
            two different sizes. Size-1 axes do **not** broadcast.
        TypeError: If ``A`` is not a sparse array.

    Examples:
        >>> import torch
        >>> from btorch import sparse
        >>> from btorch.sparse.einsum import einsum
        >>> A = sparse.coo(torch.tensor([[0, 1], [2, 0]]),
        ...                torch.tensor([3.0, 5.0]), shape=(2, 3))
        >>> einsum("mn,bn->bm", A, torch.tensor([[1.0, 0.0, 2.0]]))
        tensor([[6., 5.]])
        >>> einsum("mn->", A)
        tensor(8.)

        A batch of third-order sparse arrays with a vector per entry,
        contracted with one vector per batch member:

        >>> Y = einsum("bijkd,bk->bijd", A5, x)        # doctest: +SKIP
    """
    if not isinstance(A, Sparse):
        raise TypeError(
            "The first operand of sparse.einsum must be a sparse array, got "
            f"{type(A).__name__}."
        )
    for op in operands:
        if isinstance(op, Sparse):
            raise NotImplementedError(
                "sparse.einsum supports exactly one sparse operand (the first); "
                "sparse x sparse contraction is not implemented. Densify the "
                "other operands with to_dense() if they are small."
            )
        if not isinstance(op, Tensor):
            raise TypeError(f"Operands must be tensors, got {type(op).__name__}.")
    terms, out = _parse(subscripts, 1 + len(operands))
    sub_a, sub_ops = terms[0], terms[1:]

    coo = A.tocoo()
    indices, values = coo._indices, coo._values
    n_vb, n_index = coo._value_batch_dim, indices.shape[0]

    # ---- sizes: every occurrence of a letter must have the same size.
    if len(sub_a) != A.ndim:
        raise ValueError(
            f"The sparse term {sub_a!r} has {len(sub_a)} letter(s) but the "
            f"array has {A.ndim} dimensions, shape {A.shape} = (*batch "
            f"{A.batch_shape}, *sparse {A.sparse_shape}, *dense {A.dense_shape})."
        )
    size: dict[str, int] = {}

    def register(term: str, shape, what: str) -> None:
        for letter, n in zip(term, shape):
            if size.setdefault(letter, int(n)) != int(n):
                raise ValueError(
                    f"Subscript {letter!r} has size {int(n)} in {what} but "
                    f"size {size[letter]} elsewhere in {subscripts!r} "
                    "(size-1 axes do not broadcast)."
                )

    register(sub_a, A.shape, f"the sparse operand (shape {A.shape})")
    for k, (term, op) in enumerate(zip(sub_ops, operands), start=1):
        if len(term) != op.ndim:
            raise ValueError(
                f"Term {term!r} has {len(term)} letter(s) but operand {k} has "
                f"shape {tuple(op.shape)}."
            )
        register(term, op.shape, f"operand {k} (shape {tuple(op.shape)})")

    # ---- classify the letters of A by where their axis is stored.
    idx_letters = sub_a[n_vb : n_vb + n_index]
    val_letters = sub_a[:n_vb] + sub_a[n_vb + n_index :]
    mixed = sorted(set(idx_letters) & set(val_letters))
    if mixed:
        raise NotImplementedError(
            f"Subscript(s) {mixed} of {sub_a!r} repeat across a coordinate "
            "(sparse) axis and a batch/dense value axis of the sparse operand; "
            "this diagonal is not implemented."
        )

    # ---- 1. diagonal filter over repeated indexed letters.
    rows: dict[str, Tensor] = {}
    mask = None
    for r, letter in enumerate(idx_letters):
        if letter in rows:
            agree = indices[r] == rows[letter]
            mask = agree if mask is None else mask & agree
        else:
            rows[letter] = indices[r]
    if mask is not None:
        keep = mask.nonzero().squeeze(1)
        rows = {letter: row[keep] for letter, row in rows.items()}
        values = values.index_select(n_vb, keep)

    # ---- dtype: promote like a product of all operands would.
    dtype = functools.reduce(
        torch.promote_types, [op.dtype for op in operands], values.dtype
    )
    values = values.to(dtype)

    # ---- 2. gather the dense operands along the indexed letters.
    used = set("".join(terms)) | set(out)
    free = [c for c in string.ascii_letters if c not in used]
    if not free:
        raise NotImplementedError("sparse.einsum needs one unused subscript letter.")
    entry = free[0]
    inner_terms = [sub_a[:n_vb] + entry + sub_a[n_vb + n_index :]]
    inner_ops = [values]
    for term, op in zip(sub_ops, operands):
        op = op.to(dtype)
        hit = [d for d, letter in enumerate(term) if letter in rows]
        if hit:
            rest = [d for d in range(op.ndim) if d not in hit]
            # Indexed axes first, so the advanced index yields [E, *rest].
            op = op.permute(*hit, *rest)[tuple(rows[term[d]] for d in hit)]
            term = entry + "".join(term[d] for d in rest)
        inner_terms.append(term)
        inner_ops.append(op)

    # ---- 3. contract; 4. scatter into the surviving indexed letters.
    kept = [c for c in out if c in rows]
    if not kept:
        return torch.einsum(",".join(inner_terms) + "->" + out, *inner_ops)
    rest_out = [c for c in out if c not in rows]
    contrib = torch.einsum(
        ",".join(inner_terms) + "->" + entry + "".join(rest_out), *inner_ops
    )
    kept_sizes = tuple(size[c] for c in kept)
    key = _ravel(torch.stack([rows[c] for c in kept]), kept_sizes)
    flat = contrib.new_zeros(math.prod(kept_sizes), *contrib.shape[1:])
    result = flat.index_add(0, key, contrib).reshape(*kept_sizes, *contrib.shape[1:])
    current = kept + rest_out
    return result.permute(*[current.index(c) for c in out])
