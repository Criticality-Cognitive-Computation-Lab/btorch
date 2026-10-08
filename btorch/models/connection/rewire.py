"""Structural plasticity: fixed-slot hard rewiring (Deep R).

Rewiring is a *training policy* applied to a
:class:`~btorch.models.connection.SparseConnection`, not a sparse format: the
connection keeps its ``K`` edge slots and a controller changes which
``(post, pre)`` pair each slot stands for, at the optimizer-step boundary.

Only the hard variant is implemented (:class:`HardDeepR`).

**Soft Deep R is not implemented.** In exact soft Deep R a dormant connection
keeps its latent parameter and keeps receiving the noise/prior update, so it
can reactivate at its old position with its old history; the number of active
connections fluctuates. That needs one latent parameter for *every* candidate
connection, i.e. dense ``n_post * n_pre`` memory (plus an active mask), which
is exactly what a sparse connection avoids. A scalable approximation that
keeps only a bounded pool of dormant candidates is a different algorithm (the
pool, not the prior, decides what can reactivate) and would have to carry a
distinct name such as ``SampledSoftRewire`` and be documented as *not* exact
soft Deep R.

References:
    [1] Bellec, Kappel, Maass, Legenstein, "Deep Rewiring: Training very
        sparse deep networks", ICLR 2018.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.utils.hooks import RemovableHandle

from .sparse import SparseConnection
from .weight import EdgeWeight


__all__ = ["HardDeepR", "HardDeepROptions"]


@dataclass(frozen=True)
class HardDeepROptions:
    """Options of :class:`HardDeepR`.

    Args:
        optimizer_state: What happens to the optimizer state of a rewired
            slot. ``"reset"`` zeroes the slot's entry in every per-parameter
            state tensor that has the parameter's shape (Adam ``exp_avg``,
            ``exp_avg_sq``, ``max_exp_avg_sq``, SGD ``momentum_buffer``,
            RMSprop ``square_avg``, ...), so the new connection does not
            inherit the moments of the removed one. ``"keep"`` leaves the
            state untouched (the slot's history is transferred to the new
            connection). Scalar state such as Adam's ``step`` is never
            changed, so bias correction of a reset slot follows the global
            step count. State that is not a tensor of the parameter's shape
            (e.g. L-BFGS history) is not handled.
        every: Run the structural update on every ``every``-th optimizer
            step. ``l1`` and ``noise`` are applied on every step.
        init: Initial :math:`\\theta` of a newly activated connection. The
            small positive default keeps a new connection alive until a
            gradient (or ``l1`` / ``noise``) moves it; with ``0.0`` a new
            connection that receives an exactly zero update is dormant again
            at the next check and is redrawn.
        l1: Strength :math:`\\alpha` of the L1 prior, applied in the
            optimizer hook as a direct shrink
            :math:`\\theta \\leftarrow \\theta - \\eta\\alpha` (not through
            the loss, so adaptive optimizers do not rescale it). :math:`\\eta`
            is the learning rate of the parameter group.
        noise: Temperature :math:`T` of the Deep R random walk, applied in
            the optimizer hook as
            :math:`\\theta \\leftarrow \\theta + \\sqrt{2\\eta T}\\,\\nu`,
            :math:`\\nu \\sim \\mathcal{N}(0, 1)`.
        sign: Sign of a new connection. ``"pre"``: the sign of its source
            neuron, derived once at construction from the signs of the
            source's existing edges (sources without edges, or with balanced
            signs, get a random sign). ``"random"``: independent
            :math:`\\pm 1` per new connection. A ``[n_pre]`` tensor: explicit
            sign per source neuron. ``None`` selects ``"pre"`` with Dale's
            law and ``"random"`` without.
        allow_autapses: Allow new connections with ``post == pre`` (only
            meaningful for recurrent connections, ``n_post == n_pre``).
        candidate: Optional ``candidate(post, pre) -> bool mask`` restricting
            where new connections may appear, e.g.
            ``lambda post, pre: is_excitatory[pre]``. Both arguments are
            ``long`` tensors of the same (arbitrary) shape on the
            connection's device. Sampling is by rejection, so a very
            restrictive mask is slow.
        max_tries: Rejection-sampling rounds before giving up.
    """

    optimizer_state: Literal["reset", "keep"] = "reset"
    every: int = 1
    init: float = 1e-6
    l1: float = 0.0
    noise: float = 0.0
    sign: Literal["pre", "random"] | Tensor | None = None
    allow_autapses: bool = True
    candidate: Callable[[Tensor, Tensor], Tensor] | None = None
    max_tries: int = 100

    def __post_init__(self) -> None:
        if self.optimizer_state not in ("reset", "keep"):
            raise ValueError(
                "optimizer_state must be 'reset' or 'keep', got "
                f"{self.optimizer_state!r}."
            )
        if self.every < 1:
            raise ValueError(f"every must be >= 1, got {self.every}.")
        if self.init < 0 or self.l1 < 0 or self.noise < 0:
            raise ValueError("init, l1 and noise must be non-negative.")
        if not isinstance(self.sign, Tensor) and self.sign not in (
            None,
            "pre",
            "random",
        ):
            raise ValueError(
                "sign must be 'pre', 'random', a [n_pre] tensor or None, got "
                f"{self.sign!r}."
            )


# Candidates drawn per vacant slot in one rejection round.
_OVERSAMPLE = 4


class HardDeepR:
    """Hard Deep Rewiring with a fixed budget of edge slots.

    Every edge slot ``k`` of the connection carries a parameter
    :math:`\\theta_k` and a fixed sign :math:`s_k`; its weight is
    :math:`w_k = s_k \\theta_k`. After an optimizer step, slots with
    :math:`\\theta_k \\le 0` are *dormant*: their connection is removed and
    the slot is re-used for a new connection at a random currently
    unconnected position, with :math:`\\theta_k` = ``init``. The number of
    connections therefore stays exactly ``K = conn.nnz``; no tensor changes
    shape and the weight parameter object is never replaced, so optimizers,
    compiled graphs and CUDA graphs stay valid.

    .. math::
        \\theta_k \\leftarrow \\theta_k - \\eta \\frac{\\partial E}{\\partial
        \\theta_k} - \\eta\\alpha + \\sqrt{2 \\eta T}\\,\\nu_k

    The gradient term is the ordinary optimizer step; the optional prior
    (``l1``) and noise terms are applied by this controller.

    **Parameterisation.** The connection stores signed weights (an
    :class:`~btorch.models.connection.EdgeWeight`), so
    :math:`\\theta_k = s_k \\cdot \\mathrm{value}_k`.

    - With Dale's law (``Synapse(dale=True)``) :math:`s_k` is the weight's
      own ``sign`` buffer, which is part of the connection's ``state_dict``.
    - Without Dale's law the controller tracks :math:`s_k` itself: the sign
      the weight had when the slot was last (re)activated (at construction:
      the sign of the initial weight). A weight is dormant once it reached or
      crossed zero relative to that sign. This buffer is not checkpointed;
      with ``init > 0`` (the default) it is recoverable as ``sign(value)``
      because every active slot then has :math:`\\theta > 0`. Call
      :meth:`sync` after loading a checkpoint into the connection. With
      ``init=0`` slots activated by the last update have a zero weight and
      are simply re-drawn after such a reload.

    The dormancy test is :math:`\\theta \\le 0` (the paper uses
    :math:`\\theta < 0` with new connections at exactly zero) so that weights
    clamped to zero by a Dale projection (``constrain_net``) that ran before
    the update are still recognised. Slots whose weight is exactly zero at
    construction are dormant at the first update. After a structural update
    every slot satisfies Dale's law; with ``every > 1`` a weight may have the
    wrong sign for up to ``every - 1`` steps unless the training loop also
    applies the Dale projection (``constrain_net``), which is compatible.

    **New positions** are drawn uniformly from the pairs that are not
    connected when the update starts (the positions being vacated in this
    update become eligible at the next one, so a rewired slot always moves),
    without duplicates. With receptors / delays a slot keeps its receptor and
    delay, and "connected" refers to the full ``(post, pre, receptor,
    delay)`` tuple. Sampling is by rejection against the sorted keys of the
    existing edges; no dense ``n_post x n_pre`` structure is built.

    Construct the controller *before* ``torch.compile(conn)``: it calls
    :meth:`SparseConnection.enable_rewiring`, which makes the traced forward
    independent of the edge order.

    Args:
        conn: Connection to rewire. Its weight must be a trainable, unbatched
            :class:`~btorch.models.connection.EdgeWeight`.
        options: Rewiring options.
        generator: Random generator for reproducible rewiring. Samples are
            drawn on the generator's device and moved to the connection's
            device. ``None`` uses the global generator of the connection's
            device.

    Raises:
        TypeError: The weight is not a trainable ``EdgeWeight``.
        NotImplementedError: Batched weights or a batch of different
            sparsity patterns.

    Examples:
        >>> conn = SparseConnection.from_adjacency(W, Synapse(dale=True))
        >>> rewire = HardDeepR(conn, HardDeepROptions(l1=1e-4))
        >>> optimizer = torch.optim.Adam(conn.parameters(), lr=1e-3)
        >>> rewire.attach(optimizer)                        # doctest: +SKIP
        >>> loss.backward(); optimizer.step()               # rewires as needed

    Notes:
        ``optimizer.zero_grad()`` is not affected; a stale ``.grad`` entry of
        a rewired slot is overwritten by the next backward pass. Gradient
        scalers that skip ``optimizer.step`` also skip the update. The
        derived execution layouts are rebuilt (two sorts over the edges) on
        every update that rewires at least one slot; use ``every`` to bound
        that cost.
    """

    def __init__(
        self,
        conn: SparseConnection,
        options: HardDeepROptions | None = None,
        *,
        generator: torch.Generator | None = None,
    ):
        weight = conn.weight
        if not isinstance(weight, EdgeWeight):
            raise TypeError(
                "HardDeepR needs one independent weight per edge (EdgeWeight); "
                f"{type(weight).__name__} does not support structural rewiring."
            )
        if not isinstance(weight.value, nn.Parameter):
            raise TypeError(
                "HardDeepR needs trainable weights; this EdgeWeight is fixed "
                "(trainable=False), so no connection could ever become dormant."
            )
        if conn._member_shape:
            raise NotImplementedError(
                "HardDeepR does not support a batch of networks with different "
                "sparsity patterns."
            )
        if weight.value.ndim != 1:
            raise NotImplementedError(
                "HardDeepR does not support batched weights "
                f"(shape {tuple(weight.value.shape)}): the networks of a batch "
                "share one topology but would go dormant at different slots."
            )
        self.conn = conn
        self.options = options or HardDeepROptions()
        self.generator = generator
        self.n_steps = 0
        self.n_rewired = 0
        self._optimizer: Optimizer | None = None
        self._handle: RemovableHandle | None = None

        n_free = conn._lowered_shape[0] * conn._lowered_shape[1] - conn.nnz
        if n_free <= 0:
            raise ValueError("The connection is dense: there is nowhere to rewire to.")

        self._slot_sign: Tensor | None = None
        self.sync()
        self._pre_sign = self._make_pre_sign()
        # Slot order no longer equals execution order once edges move; make
        # the forward independent of it now so compiled graphs stay valid.
        conn.enable_rewiring()

    # ------------------------------------------------------------- signs
    @property
    def _dale(self) -> bool:
        return bool(self.conn.weight.dale)

    @torch.no_grad()
    def sync(self) -> None:
        """Re-derive the reference signs from the current weights.

        Only relevant without Dale's law, where the controller tracks the
        sign of every slot: call it after loading a checkpoint into the
        connection or after editing its weights or edges by hand.
        """
        if not self._dale:
            self._slot_sign = torch.sign(self.conn.weight.value.detach())

    def _signs(self) -> Tensor:
        """``[K]`` reference sign of every slot (a live buffer)."""
        if self._dale:
            return self.conn.weight.sign
        value = self.conn.weight.value
        self._slot_sign = self._slot_sign.to(device=value.device, dtype=value.dtype)
        return self._slot_sign

    def _make_pre_sign(self) -> Tensor | None:
        """``[n_pre]`` sign of every source neuron, or ``None`` (random)."""
        mode = self.options.sign
        if mode is None:
            mode = "pre" if self._dale else "random"
        n_pre = self.conn.n_pre
        if isinstance(mode, Tensor):
            if mode.shape != (n_pre,) or not bool((mode.abs() == 1).all()):
                raise ValueError(
                    f"sign must hold +1/-1 for each of the {n_pre} source "
                    f"neurons, got shape {tuple(mode.shape)}."
                )
            return torch.sign(mode.detach().to(torch.float32)).cpu()
        if mode == "random":
            return None
        slot_sign = self._signs().detach().float().cpu()
        total = torch.zeros(n_pre).index_add_(0, self.conn.indices[1].cpu(), slot_sign)
        fallback = self._rand_sign(n_pre, torch.device("cpu"))
        return torch.where(total == 0, fallback, torch.sign(total))

    # ------------------------------------------------------------ sampling
    def _randint(self, high: int, shape: tuple[int, ...], device) -> Tensor:
        g = self.generator
        if g is None:
            return torch.randint(high, shape, device=device)
        return torch.randint(high, shape, generator=g, device=g.device).to(device)

    def _randn(self, n: int, like: Tensor) -> Tensor:
        g = self.generator
        if g is None:
            return torch.randn(n, device=like.device, dtype=like.dtype)
        return torch.randn(n, generator=g, device=g.device, dtype=like.dtype).to(
            like.device
        )

    def _rand_sign(self, n: int, device) -> Tensor:
        return self._randint(2, (n,), device).float() * 2 - 1

    def _sample(self, slots: Tensor) -> tuple[Tensor, Tensor]:
        """Draw new ``(post, pre)`` for ``slots``.

        Every vacant slot draws i.i.d. uniform pairs and takes the first
        one that is allowed and not occupied; ties between slots that
        picked the same position are resolved for one of them and the
        others redraw. Rejecting occupied / disallowed / already taken
        positions from uniform draws is uniform sampling without
        replacement from the allowed unconnected positions.
        """
        conn, opt = self.conn, self.options
        device = conn.indices.device
        n_in = conn._lowered_shape[1]
        n_r, n_d = conn.n_receptor, conn.n_delay
        row, col = conn._lowered_edges()
        # Every current edge is forbidden, including the ones being removed.
        taken = torch.sort(row * n_in + col).values

        def offset(attr: Tensor | None, n: int) -> Tensor | int:
            # A slot keeps its receptor / delay; an attribute that is absent
            # although the axis exists means channel 0 (see _lowered_edges).
            return attr[slots] if attr is not None and n > 1 else 0

        rec, dly = offset(conn.receptor, n_r), offset(conn.delay, n_d)
        new_post = torch.empty_like(slots)
        new_pre = torch.empty_like(slots)
        pending = torch.arange(slots.shape[0], device=device)
        for _ in range(opt.max_tries):
            m = pending.shape[0]
            if m == 0:
                return new_post, new_pre
            post = self._randint(conn.n_post, (m, _OVERSAMPLE), device)
            pre = self._randint(conn.n_pre, (m, _OVERSAMPLE), device)
            p_rec = rec[pending, None] if isinstance(rec, Tensor) else rec
            p_dly = dly[pending, None] if isinstance(dly, Tensor) else dly
            key = (post * n_r + p_rec) * n_in + (pre * n_d + p_dly)
            pos = torch.searchsorted(taken, key).clamp(max=taken.shape[0] - 1)
            ok = taken[pos] != key
            if not opt.allow_autapses:
                ok &= post != pre
            if opt.candidate is not None:
                ok &= opt.candidate(post, pre).to(device=device, dtype=torch.bool)
            # First acceptable draw of every slot.
            first = ok.to(torch.uint8).argmax(dim=1, keepdim=True)
            found = ok.any(dim=1)
            key = key.gather(1, first)[:, 0]
            post = post.gather(1, first)[:, 0]
            pre = pre.gather(1, first)[:, 0]
            # Two slots may have picked the same position: keep one of them.
            found_idx = found.nonzero()[:, 0]
            sorted_key, order = torch.sort(key[found_idx], stable=True)
            unique = torch.ones_like(sorted_key, dtype=torch.bool)
            unique[1:] = sorted_key[1:] != sorted_key[:-1]
            win = found_idx[order[unique]]
            new_post[pending[win]] = post[win]
            new_pre[pending[win]] = pre[win]
            taken = torch.sort(torch.cat([taken, key[win]])).values
            keep = torch.ones(m, dtype=torch.bool, device=device)
            keep[win] = False
            pending = pending[keep]
        if pending.shape[0]:
            raise RuntimeError(
                f"HardDeepR could not find unconnected positions for "
                f"{pending.shape[0]} of {slots.shape[0]} dormant slots in "
                f"{opt.max_tries} rounds; the connection is (nearly) full or "
                "the candidate restriction leaves too few free positions."
            )
        return new_post, new_pre

    # -------------------------------------------------------------- update
    @torch.no_grad()
    def dormant_slots(self) -> Tensor:
        """Edge slots whose connection is dormant (``theta <= 0``)."""
        theta = self.conn.weight.value * self._signs()
        return (theta <= 0).nonzero()[:, 0]

    @torch.no_grad()
    def step(self, optimizer: Optimizer | None = None) -> int:
        """Run one structural update.

        Called automatically after ``optimizer.step()`` once attached; call
        it manually for custom schedules. ``l1`` and ``noise`` are *not*
        applied here, only by the optimizer hook.

        Args:
            optimizer: Optimizer whose state follows ``optimizer_state``.
                Defaults to the attached optimizer; without one no optimizer
                state is touched.

        Returns:
            Number of rewired slots.
        """
        conn, opt = self.conn, self.options
        weight = conn.weight
        slots = self.dormant_slots()
        n = int(slots.shape[0])
        if n == 0:
            return 0
        post, pre = self._sample(slots)
        if self._pre_sign is None:
            sign = self._rand_sign(n, slots.device)
        else:
            self._pre_sign = self._pre_sign.to(slots.device)
            sign = self._pre_sign[pre]
        sign = sign.to(weight.value.dtype)

        conn.set_edges_(slots, post, pre)
        weight.reset_slots(slots, sign * opt.init, sign)
        if not self._dale:
            self._signs()[slots] = sign

        if optimizer is None:
            optimizer = self._optimizer
        if optimizer is not None and opt.optimizer_state == "reset":
            param = weight.value
            for state in optimizer.state.get(param, {}).values():
                if isinstance(state, Tensor) and state.shape == param.shape:
                    state[slots] = 0
        self.n_rewired += n
        return n

    @torch.no_grad()
    def _regularize(self, optimizer: Optimizer) -> None:
        """L1 shrink and random walk of theta (the non-gradient Deep R
        terms)."""
        opt = self.options
        if opt.l1 == 0 and opt.noise == 0:
            return
        value = self.conn.weight.value
        lr = float(self._group(optimizer, value)["lr"])
        delta = torch.full_like(value, -lr * opt.l1)
        if opt.noise > 0:
            delta += math.sqrt(2 * lr * opt.noise) * self._randn(value.shape[0], value)
        value.add_(self._signs() * delta)

    @staticmethod
    def _group(optimizer: Optimizer, param: Tensor) -> dict:
        for group in optimizer.param_groups:
            if any(p is param for p in group["params"]):
                return group
        raise ValueError(
            "The optimizer does not optimise the weights of this connection."
        )

    # ---------------------------------------------------------- optimizer
    def attach(self, optimizer: Optimizer) -> RemovableHandle:
        """Run the update after every ``optimizer.step()``.

        Args:
            optimizer: Optimizer that holds the connection's weight
                parameter.

        Returns:
            Handle of the registered step hook (also removed by
            :meth:`detach`).

        Raises:
            ValueError: The optimizer does not hold the weight parameter.
            RuntimeError: Already attached.
        """
        if self._handle is not None:
            raise RuntimeError("HardDeepR is already attached; call detach() first.")
        self._group(optimizer, self.conn.weight.value)
        self._optimizer = optimizer
        self._handle = optimizer.register_step_post_hook(self._hook)
        return self._handle

    def detach(self) -> None:
        """Remove the optimizer hook (no-op when not attached)."""
        if self._handle is not None:
            self._handle.remove()
        self._handle = None
        self._optimizer = None

    def _hook(self, optimizer: Optimizer, args, kwargs) -> None:
        self.n_steps += 1
        self._regularize(optimizer)
        if self.n_steps % self.options.every == 0:
            self.step(optimizer)

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(n_slot={self.conn.nnz}, "
            f"n_rewired={self.n_rewired}, attached={self._handle is not None}, "
            f"{self.options})"
        )
