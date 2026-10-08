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
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

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
        optimizer_state: What happens to the per-slot optimizer state of a
            rewired slot, i.e. to its entry in every state tensor of the
            attached optimizer that has the parameter's shape. Scalar state
            (Adam's global ``step``) is never changed, which is what makes
            the choice matter:

            - ``"neutral"`` (default): first-moment-like state (``exp_avg``,
              ``momentum_buffer``, ``grad_avg``) is zeroed, so the new
              connection inherits no direction from the removed one;
              second-moment-like state (``exp_avg_sq``, ``max_exp_avg_sq``,
              ``square_avg``, Adagrad ``sum``, ``exp_inf``, ``acc_delta``) is
              set to the mean over the slots that are *not* dormant in this
              update (over all slots if every slot is), so the per-gradient
              step size of the new connection equals that of the population.
              With a gradient of typical magnitude the first step is
              :math:`\\approx(1-\\beta_1)\\eta` for Adam (the usual momentum
              ramp) and :math:`\\approx\\eta` for RMSprop. State tensors with
              another name are zeroed.
            - ``"reset"``: every per-slot state entry is zeroed. Because
              ``step`` is global, Adam's bias correction no longer
              compensates the empty second moment: at step :math:`t` the
              first update of a reset slot is :math:`\\eta\\,
              \\frac{1-\\beta_1}{1-\\beta_1^t}\\sqrt{\\frac{1-\\beta_2^t}
              {1-\\beta_2}}`, which grows to :math:`\\eta(1-\\beta_1) /
              \\sqrt{1-\\beta_2} \\approx 3.2\\eta` with the default betas
              (:math:`2.5\\eta` at :math:`t=1000`) against
              :math:`\\approx\\eta` for an established slot. Optimizers
              without bias correction are hit harder: RMSprop takes a first
              step of :math:`\\eta/\\sqrt{1-\\alpha} = 10\\eta`
              (``alpha=0.99``), and Adagrad a step of the full initial
              :math:`\\eta`. Exact for plain momentum SGD.
            - ``"keep"``: the state is left untouched; the new connection
              inherits the moments (including the momentum direction) of the
              removed one.

            ``"neutral"`` is the default because it is the only policy under
            which a new connection neither inherits a stale direction nor
            takes a larger step than the established connections; a new
            connection starts at ``init`` close to zero, so an inflated first
            step in the wrong direction makes it dormant again immediately.
            State that is not a tensor of the parameter's shape (e.g. L-BFGS
            history) is not handled.
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
        candidate: Optional ``candidate(pre, post) -> bool mask`` restricting
            where new connections may appear, e.g.
            ``lambda pre, post: is_excitatory[pre]``. Both arguments are
            ``long`` tensors of the same (arbitrary) shape on the
            connection's device. It is evaluated at every update, so it may
            depend on training progress.
        max_tries: Rejection-sampling rounds per update. Only relevant when
            the candidate universe is too large to enumerate (see
            :class:`HardDeepR`): dormant slots that found no position within
            ``max_tries`` rounds stay dormant and are retried at the next
            update; nothing is raised.
    """

    optimizer_state: Literal["neutral", "reset", "keep"] = "neutral"
    every: int = 1
    init: float = 1e-6
    l1: float = 0.0
    noise: float = 0.0
    sign: Literal["pre", "random"] | Tensor | None = None
    allow_autapses: bool = True
    candidate: Callable[[Tensor, Tensor], Tensor] | None = None
    max_tries: int = 100

    def __post_init__(self) -> None:
        if self.optimizer_state not in ("neutral", "reset", "keep"):
            raise ValueError(
                "optimizer_state must be 'neutral', 'reset' or 'keep', got "
                f"{self.optimizer_state!r}."
            )
        if self.every < 1:
            raise ValueError(f"every must be >= 1, got {self.every}.")
        if self.max_tries < 0:
            raise ValueError(f"max_tries must be >= 0, got {self.max_tries}.")
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
# Largest candidate universe (n_post * n_receptor * n_pre * n_delay) whose
# free positions are enumerated exactly: one bool and one long per pair.
_ENUMERATE_MAX = 1 << 22
# Enumerate right away (skip rejection sampling) when fewer than
# ``_SCARCE * n_dormant`` free positions exist: collisions between the slots
# and with existing edges then dominate the rejection rounds.
_SCARCE = 16
# Per-slot optimizer state by role (see ``HardDeepROptions.optimizer_state``).
_SECOND_MOMENT = frozenset(
    {"exp_avg_sq", "max_exp_avg_sq", "square_avg", "sum", "exp_inf", "acc_delta"}
)


class HardDeepR:
    """Hard Deep Rewiring with a fixed budget of edge slots.

    Every edge slot ``k`` of the connection carries a parameter
    :math:`\\theta_k` and a fixed sign :math:`s_k`; its weight is
    :math:`w_k = s_k \\theta_k`. After an optimizer step, slots with
    :math:`\\theta_k \\le 0` are *dormant*: their connection is removed and
    the slot is re-used for a new connection at a random currently
    unconnected position, with :math:`\\theta_k` = ``init``. The number of
    edge slots therefore stays exactly ``K = conn.nnz``; no tensor changes
    shape and the weight parameter object is never replaced, so optimizers
    and ``torch.compile``d modules stay valid. A CUDA graph captured around
    the connection records the wiring it was captured with and has to be
    captured again after an update that moved edges
    (``RecurrentNN(cudagraph=True)`` does this by itself; see
    ``SparseConnection.capture_version``).

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
      crossed zero relative to that sign. This buffer is part of the
      controller's :meth:`state_dict`. :meth:`sync` re-derives it from the
      current weights after editing weights or edges by hand.

    The dormancy test is :math:`\\theta \\le 0` (the paper uses
    :math:`\\theta < 0` with new connections at exactly zero) so that weights
    clamped to zero by a Dale projection (``constrain_net``) that ran before
    the update are still recognised. Slots whose weight is exactly zero at
    construction are dormant at the first update. After a structural update
    every slot satisfies Dale's law; with ``every > 1`` a weight may have the
    wrong sign for up to ``every - 1`` steps unless the training loop also
    applies the Dale projection (``constrain_net``), which is compatible.

    **New positions** are drawn uniformly, without duplicates, from the
    admissible pairs (``allow_autapses``, ``candidate``) that are not
    connected when the update starts. Positions vacated in the same update
    are deliberately *not* eligible: they become free at the next update.
    A rewired slot therefore always moves (a connection that was just
    pruned cannot be re-created in place by the same update), and a slot
    that could not be placed can simply stay where it is without ever
    colliding with a new connection. With receptors / delays a slot keeps
    its receptor and delay, and "connected" refers to the full ``(post,
    pre, receptor, delay)`` tuple.

    Positions are found by rejection sampling against the sorted keys of the
    existing edges, without building a dense ``n_post x n_pre`` structure.
    When unconnected positions are scarce (fewer than 16 per dormant slot, or
    less than 1/8 of all pairs) or rejection sampling did not finish within
    ``max_tries`` rounds, the free admissible positions are enumerated
    exactly instead, provided the universe ``n_post * n_receptor * n_pre *
    n_delay`` has at most :math:`2^{22}` pairs. Both paths sample exactly
    uniformly. The shortcut looks at unconnected positions only: a
    restrictive ``candidate`` on a sparse layer still costs ``max_tries``
    rejection rounds per update before the enumeration takes over.

    **Unplaced slots.** If there are more dormant slots than free admissible
    positions (small or nearly dense layers, a restrictive ``candidate``), or
    the universe is too large to enumerate and rejection sampling ran out of
    rounds, the update places as many slots as it can and never raises. The
    remaining slots keep their position, are held at a weight of exactly zero
    (also against gradient, ``l1`` and ``noise`` updates, on every hooked
    optimizer step) so they contribute nothing, and are retried at every
    following update. Their number in the last update is :attr:`n_unplaced`;
    one :class:`RuntimeWarning` per controller reports the first occurrence.

    **Optimizers.** One controller follows one optimizer. All per-parameter
    state tensors of the parameter's shape are handled, which covers the
    single-tensor, ``foreach`` and ``fused`` implementations (they share the
    state layout). If the weight parameter is held by a second optimizer,
    that optimizer's state is not touched by the hook (its moments of a
    rewired slot are kept); a second controller for the same weights cannot
    be attached to the same optimizer. A step in which the weight has no
    gradient (``.grad is None``) is skipped by the optimizer and, like weight
    decay, by ``l1`` and ``noise``; the structural update still runs on
    schedule.

    **Checkpointing.** Save :meth:`state_dict` together with the state of the
    connection and of the optimizer. To resume, rebuild the connection, the
    optimizer and the controller (same options; a generator if one was used),
    load the connection and the optimizer, then call
    :meth:`load_state_dict`. With a generator the resumed run reproduces the
    uninterrupted one exactly.

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
            device (whose state is then not part of :meth:`state_dict`).

    Attributes:
        n_steps: Number of hooked optimizer steps so far.
        n_rewired: Total number of slots moved to a new position.
        n_unplaced: Dormant slots the last update could not place.

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
        # A Projection is a thin front end: rewire the connection it built.
        conn = (
            getattr(conn, "connection", conn) if not hasattr(conn, "indices") else conn
        )
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
        self.n_unplaced = 0
        self._warned = False
        self._optimizer: Optimizer | None = None
        self._handle: RemovableHandle | None = None

        n_free = conn._lowered_shape[0] * conn._lowered_shape[1] - conn.nnz
        if n_free <= 0:
            raise ValueError("The connection is dense: there is nowhere to rewire to.")

        # [K] slots that are dormant but could not be placed yet.
        self._unplaced = torch.zeros(conn.nnz, dtype=torch.bool)
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
        sign of every slot: call it after editing the connection's weights
        or edges by hand. Not needed after :meth:`load_state_dict`, which
        restores the tracked signs.
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

    def _unplaced_mask(self) -> Tensor:
        """``[K]`` bool mask of the unplaced slots (a live buffer)."""
        self._unplaced = self._unplaced.to(self.conn.weight.value.device)
        return self._unplaced

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

    def _randperm(self, n: int, device) -> Tensor:
        g = self.generator
        if g is None:
            return torch.randperm(n, device=device)
        return torch.randperm(n, generator=g, device=g.device).to(device)

    def _randn(self, n: int, like: Tensor) -> Tensor:
        g = self.generator
        if g is None:
            return torch.randn(n, device=like.device, dtype=like.dtype)
        return torch.randn(n, generator=g, device=g.device, dtype=like.dtype).to(
            like.device
        )

    def _rand_sign(self, n: int, device) -> Tensor:
        return self._randint(2, (n,), device).float() * 2 - 1

    def _sample(self, slots: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Draw new ``(post, pre)`` for the dormant ``slots``.

        Rejection sampling: every vacant slot draws i.i.d. uniform pairs and
        takes the first one that is allowed and not occupied; ties between
        slots that picked the same position are resolved for one of them and
        the others redraw. Rejecting occupied / disallowed / already taken
        positions from uniform draws is uniform sampling without replacement
        from the allowed unconnected positions. Slots still vacant afterwards
        are served from an exact enumeration of the remaining free positions
        when the universe is small enough, which preserves uniformity.

        Returns:
            ``(post, pre, placed)``: new coordinates and a bool mask of the
            slots that found a position (coordinates of the others are
            meaningless).
        """
        conn, opt = self.conn, self.options
        device = conn.indices.device
        n = slots.shape[0]
        n_in = conn._lowered_shape[1]
        n_r, n_d = conn.n_receptor, conn.n_delay
        row, col = conn._lowered_edges()
        # Every current edge is forbidden, including the ones being removed.
        taken = torch.sort(row.long() * n_in + col.long()).values

        # A slot keeps its receptor / delay; an attribute that is absent
        # although the axis exists means channel 0 (see _lowered_edges).
        rec = dly = torch.zeros(n, dtype=torch.long, device=device)
        if conn.receptor is not None and n_r > 1:
            rec = conn.receptor[slots].long()
        if conn.delay is not None and n_d > 1:
            dly = conn.delay[slots].long()

        new_post = torch.zeros(n, dtype=torch.long, device=device)
        new_pre = torch.zeros(n, dtype=torch.long, device=device)
        placed = torch.zeros(n, dtype=torch.bool, device=device)
        pending = torch.arange(n, device=device)

        universe = conn.n_post * conn.n_pre
        enumerable = universe * n_r * n_d <= _ENUMERATE_MAX
        # Free positions per (receptor, delay) channel, before restrictions.
        n_free = (universe * n_r * n_d - taken.shape[0]) // (n_r * n_d)
        scarce = n_free < _SCARCE * n or n_free * 2 * _OVERSAMPLE < universe
        rounds = 0 if enumerable and scarce else opt.max_tries
        for _ in range(rounds):
            m = pending.shape[0]
            if m == 0:
                break
            post = self._randint(conn.n_post, (m, _OVERSAMPLE), device)
            pre = self._randint(conn.n_pre, (m, _OVERSAMPLE), device)
            key = (post * n_r + rec[pending, None]) * n_in + (
                pre * n_d + dly[pending, None]
            )
            pos = torch.searchsorted(taken, key).clamp(max=taken.shape[0] - 1)
            ok = taken[pos] != key
            if not opt.allow_autapses:
                ok &= post != pre
            if opt.candidate is not None:
                ok &= opt.candidate(pre, post).to(device=device, dtype=torch.bool)
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
            placed[pending[win]] = True
            taken = torch.sort(torch.cat([taken, key[win]])).values
            keep = torch.ones(m, dtype=torch.bool, device=device)
            keep[win] = False
            pending = pending[keep]

        if pending.shape[0] and enumerable:
            # Exact fallback: list the free admissible positions of every
            # (receptor, delay) channel that still has vacant slots.
            flat = torch.arange(universe, device=device)
            post, pre = flat // conn.n_pre, flat % conn.n_pre
            allowed = torch.ones(universe, dtype=torch.bool, device=device)
            if not opt.allow_autapses:
                allowed &= post != pre
            if opt.candidate is not None:
                allowed &= opt.candidate(pre, post).to(device=device, dtype=torch.bool)
            t_row, t_col = taken // n_in, taken % n_in
            t_channel = (t_row % n_r) * n_d + t_col % n_d
            t_flat = (t_row // n_r) * conn.n_pre + t_col // n_d
            channel = rec[pending] * n_d + dly[pending]
            for c in channel.unique().tolist():
                members = pending[channel == c]
                free = allowed.clone()
                free[t_flat[t_channel == c]] = False
                free = free.nonzero()[:, 0]
                m = min(members.shape[0], free.shape[0])
                if m == 0:
                    continue
                pick = free[self._randperm(free.shape[0], device)[:m]]
                new_post[members[:m]] = post[pick]
                new_pre[members[:m]] = pre[pick]
                placed[members[:m]] = True
        return new_post, new_pre, placed

    # -------------------------------------------------------------- update
    @torch.no_grad()
    def dormant_slots(self) -> Tensor:
        """Edge slots whose connection is dormant.

        These are the slots with ``theta <= 0`` plus the slots a previous
        update could not place (they stay dormant whatever the optimizer did
        to their weight in the meantime).
        """
        theta = self.conn.weight.value * self._signs()
        return ((theta <= 0) | self._unplaced_mask()).nonzero()[:, 0]

    @torch.no_grad()
    def step(self, optimizer: Optimizer | None = None) -> int:
        """Run one structural update.

        Called automatically after ``optimizer.step()`` once attached; call
        it manually for custom schedules. This is only the structural part of
        Deep R. ``l1`` and ``noise`` are *not* applied here: they are terms
        of the per-optimizer-step parameter update, scaled by the learning
        rate of that step, so tying them to a manual structural schedule
        would change their strength with the schedule. A manual loop that
        wants them calls :meth:`regularize` after each ``optimizer.step()``;
        the optimizer hook is exactly ``regularize(optimizer)`` followed, on
        every ``every``-th step, by ``step(optimizer)``.

        Dormant slots that cannot be placed are set to zero and retried at
        the next update (see the class documentation); nothing is raised.
        Only the optimizer hook re-zeroes them after every optimizer step: in
        a manual loop without :meth:`attach`, call ``step()`` after every
        optimizer step while :attr:`n_unplaced` is non-zero, otherwise the
        gradient moves their weight until the next call.

        Args:
            optimizer: Optimizer whose state follows ``optimizer_state``.
                Defaults to the attached optimizer; without one no optimizer
                state is touched.

        Returns:
            Number of rewired (placed) slots.
        """
        conn, opt = self.conn, self.options
        weight = conn.weight
        unplaced = self._unplaced_mask()
        slots = self.dormant_slots()
        if slots.shape[0] == 0:
            self.n_unplaced = 0
            return 0
        post, pre, placed = self._sample(slots)
        moved, waiting = slots[placed], slots[~placed]
        post, pre = post[placed], pre[placed]
        n = int(moved.shape[0])

        if self._pre_sign is None:
            sign = self._rand_sign(n, slots.device)
        else:
            self._pre_sign = self._pre_sign.to(slots.device)
            sign = self._pre_sign[pre]
        sign = sign.to(weight.value.dtype)

        # Both calls are no-ops for an empty slot list.
        conn.set_edges_(moved, pre=pre, post=post)
        weight.reset_slots(moved, sign * opt.init, sign)
        if not self._dale:
            self._signs()[moved] = sign
        # Slots without a new position stay where they are, with a weight of
        # exactly zero and their old reference sign.
        weight.reset_slots(waiting, 0.0)
        unplaced.fill_(False)
        unplaced[waiting] = True

        if optimizer is None:
            optimizer = self._optimizer
        if optimizer is not None and n:
            self._reset_optimizer_state(optimizer, moved, slots)

        self.n_rewired += n
        self.n_unplaced = int(waiting.shape[0])
        if self.n_unplaced and not self._warned:
            self._warned = True
            warnings.warn(
                f"HardDeepR could not place {self.n_unplaced} of "
                f"{slots.shape[0]} dormant slots: the connection is (nearly) "
                "full or the candidate restriction leaves too few free "
                "positions. They stay dormant (weight zero) and are retried at "
                "every update; see `n_unplaced`. This warning is shown once "
                "per controller.",
                RuntimeWarning,
                stacklevel=2,
            )
        return n

    def _reset_optimizer_state(
        self, optimizer: Optimizer, moved: Tensor, dormant: Tensor
    ) -> None:
        """Apply ``optimizer_state`` to the state of the ``moved`` slots.

        ``dormant`` (a superset of ``moved``) is excluded from the population
        statistics of the ``"neutral"`` policy.
        """
        policy = self.options.optimizer_state
        if policy == "keep":
            return
        param = self.conn.weight.value
        alive = torch.ones(param.shape[0], dtype=torch.bool, device=param.device)
        alive[dormant] = False
        for name, state in optimizer.state.get(param, {}).items():
            if not isinstance(state, Tensor) or state.shape != param.shape:
                continue
            if policy == "neutral" and name in _SECOND_MOMENT:
                # No established slot left: fall back to all slots.
                state[moved] = state[alive].mean() if alive.any() else state.mean()
            else:
                state[moved] = 0

    @torch.no_grad()
    def regularize(self, optimizer: Optimizer | None = None) -> None:
        """Apply the L1 shrink and the random walk of theta once.

        These are the non-gradient terms of the Deep R update of one
        optimizer step. The optimizer hook calls this after every
        ``optimizer.step()`` in which the weight had a gradient; call it
        yourself only in a manual loop without :meth:`attach`. Unplaced slots
        are not changed.

        Args:
            optimizer: Optimizer that provides the learning rate of the
                weight's parameter group. Defaults to the attached optimizer.

        Raises:
            ValueError: No optimizer, or it does not hold the weights.
        """
        opt = self.options
        if opt.l1 == 0 and opt.noise == 0:
            return
        if optimizer is None:
            optimizer = self._optimizer
        if optimizer is None:
            raise ValueError("regularize() needs an optimizer for the learning rate.")
        value = self.conn.weight.value
        lr = float(self._group(optimizer, value)["lr"])
        delta = torch.full_like(value, -lr * opt.l1)
        if opt.noise > 0:
            delta += math.sqrt(2 * lr * opt.noise) * self._randn(value.shape[0], value)
        if self.n_unplaced:
            delta[self._unplaced_mask()] = 0
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
    @property
    def attached(self) -> bool:
        """Whether the optimizer hook is currently registered."""
        if self._handle is None or self._optimizer is None:
            return False
        # The hook may have been removed through the handle attach() returned.
        hooks = getattr(self._optimizer, "_optimizer_step_post_hooks", {})
        return self._handle.id in hooks

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
            RuntimeError: Already attached, or another controller for the
                same weights is attached to this optimizer.
        """
        if self.attached:
            raise RuntimeError("HardDeepR is already attached; call detach() first.")
        param = self.conn.weight.value
        self._group(optimizer, param)
        for hook in getattr(optimizer, "_optimizer_step_post_hooks", {}).values():
            other = getattr(hook, "__self__", None)
            if isinstance(other, HardDeepR) and other.conn.weight.value is param:
                raise RuntimeError(
                    "Another HardDeepR controller for the same weights is "
                    "already attached to this optimizer."
                )
        self._optimizer = optimizer
        self._handle = optimizer.register_step_post_hook(self._hook)
        return self._handle

    def detach(self) -> None:
        """Remove the optimizer hook (no-op when not attached)."""
        if self._handle is not None:
            self._handle.remove()
        self._handle = None
        self._optimizer = None

    @torch.no_grad()
    def _hook(self, optimizer: Optimizer, args, kwargs) -> None:
        self.n_steps += 1
        value = self.conn.weight.value
        if value.grad is not None:
            # Without a gradient the optimizer skipped the parameter (as it
            # skips weight decay); the prior and the noise are skipped too.
            self.regularize(optimizer)
            if self.n_unplaced:
                # Undo the gradient step on slots that wait for a position.
                value[self._unplaced_mask()] = 0
        if self.n_steps % self.options.every == 0:
            self.step(optimizer)

    # ------------------------------------------------------- checkpointing
    def state_dict(self) -> dict[str, Any]:
        """State needed to resume rewiring exactly (tensors and ints).

        Holds the counters, the mask of unplaced slots, the per-source sign
        table (it has a random fallback for sources without edges), the
        per-slot reference signs (without Dale's law) and the state of the
        generator the controller was given. Tensors are CPU copies. The
        options, the connection and the optimizer are not included; the
        global random state used with ``generator=None`` is not either.
        """
        state: dict[str, Any] = {
            "n_steps": self.n_steps,
            "n_rewired": self.n_rewired,
            "n_unplaced": self.n_unplaced,
            "warned": int(self._warned),
            "unplaced": self._unplaced.detach().cpu().clone(),
        }
        if self._pre_sign is not None:
            state["pre_sign"] = self._pre_sign.detach().cpu().clone()
        if self._slot_sign is not None:
            state["slot_sign"] = self._slot_sign.detach().cpu().clone()
        if self.generator is not None:
            state["generator"] = self.generator.get_state().clone()
        return state

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore a state saved by :meth:`state_dict`.

        Call it on a controller built with the same options for a connection
        of the same size, after loading the connection's own ``state_dict``.

        Args:
            state_dict: Output of :meth:`state_dict`.

        Raises:
            ValueError: The entries do not match this controller (different
                sign mode, Dale setting, generator or sizes). Nothing is
                changed in that case.
        """
        expected = set(self.state_dict())
        if set(state_dict) != expected:
            raise ValueError(
                "The rewiring checkpoint does not match this controller: it "
                f"holds {sorted(state_dict)}, expected {sorted(expected)} "
                "(different `sign` mode, Dale setting or generator?)."
            )
        shapes = {
            "unplaced": (self.conn.nnz,),
            "pre_sign": (self.conn.n_pre,),
            "slot_sign": (self.conn.nnz,),
        }
        for name, shape in shapes.items():
            if name in state_dict and tuple(state_dict[name].shape) != shape:
                raise ValueError(
                    f"'{name}' in the rewiring checkpoint has shape "
                    f"{tuple(state_dict[name].shape)}, expected {shape}."
                )
        if self.generator is not None:
            self.generator.set_state(state_dict["generator"])
        self.n_steps = int(state_dict["n_steps"])
        self.n_rewired = int(state_dict["n_rewired"])
        self.n_unplaced = int(state_dict["n_unplaced"])
        self._warned = bool(state_dict["warned"])
        self._unplaced = state_dict["unplaced"].detach().clone().to(torch.bool)
        if self._pre_sign is not None:
            self._pre_sign = state_dict["pre_sign"].detach().clone()
        if self._slot_sign is not None:
            self._slot_sign = state_dict["slot_sign"].detach().clone()

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(n_slot={self.conn.nnz}, "
            f"n_rewired={self.n_rewired}, n_unplaced={self.n_unplaced}, "
            f"attached={self.attached}, {self.options})"
        )
