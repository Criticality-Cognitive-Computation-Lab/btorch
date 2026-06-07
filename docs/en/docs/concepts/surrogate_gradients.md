# Surrogate Gradients

Spiking neurons use a discontinuous activation (a spike is emitted when the
membrane voltage crosses a threshold). This discontinuity makes standard
backpropagation through time (BPTT) impossible because the gradient of the
spike function is zero almost everywhere.

Surrogate gradients solve this by replacing the true gradient with a smooth
approximation during the backward pass.

## Available Surrogates

btorch provides several surrogate gradient functions in `btorch.models.surrogate`:

| Class | Surrogate gradient `g(v)` | Default alpha |
|-------|---------------------------|---------------|
| `ATan` | `1 / (1 + (α v)²)` | 2.0 |
| `ATanApprox` | rational approx of `ATan` | 2.0 |
| `Sigmoid` | `4 σ(k·α·v)(1 − σ(k·α·v))`,  k = 2 ln(√2+1) | 2.0 |
| `Erf` | `2^{−(α v)²}` | 4.0 |
| `Triangle` | `(1 − \|α v\| / 2)₊` | 2.0 |
| `SuperSpike` | `(1 + (√2−1)·α·\|v\|)⁻²` | 2.0 |

All default values correspond to a half-width of 0.5 (see
[Alpha convention](#alpha-convention-hwhm--1alpha) below), except `Erf` whose
default alpha=4 (HWHM=0.25) is calibrated to match the V1 model of
Chen et al. (2022).

## Convention 1 — Peak normalisation: `g(0) = 1`

All btorch surrogates satisfy:

> **`g(v=0, damping_factor=1) == 1.0` for any value of `alpha`.**

This ensures the effective gradient magnitude at the spike threshold is always 1
when no intentional scaling is applied. Switching surrogates or changing `alpha`
does not accidentally rescale the learning signal.

`damping_factor` is the *sole* control for intentional gradient scaling.

This convention follows Zenke & Neftci (2021), who show empirically that the
key property of a well-behaved surrogate is a unit response at the threshold,
not a unit integral over voltage. Previously, btorch
scaled each derivative to integrate to 1 — motivated by an analogy to
probability densities — but this is the wrong invariant.

> Zenke, F., & Neftci, E. O. (2021). *The remarkable robustness of surrogate
> gradient learning for instilling complex function in spiking neural networks.*
> Neural Computation, 33(4), 899–925.
> https://doi.org/10.1162/neco_a_01097

The normalisation factor for each surrogate:

| Surrogate   | Unnormalised peak | Factor baked in | Normalised `g(v)` |
|-------------|-------------------|-----------------|-------------------|
| Triangle    | `alpha/2`         | `2/alpha`       | `(1 − \|αv\|/2)₊` |
| Sigmoid     | `alpha/4`         | `4/alpha`       | `4σ(k·αv)(1−σ)`, k=2ln(√2+1) |
| Erf         | `alpha/√π`        | `√π/alpha`      | `2^{−(αv)²}` |
| ATan        | `alpha/2`         | `2/alpha`       | `1/(1+(αv)²)` |
| ATanApprox  | `alpha/2`         | `2/alpha`       | rational approx |
| SuperSpike  | 1                 | —               | `(1+(√2−1)α\|v\|)⁻²` |

## Convention 2 — Alpha convention: HWHM = 1/alpha

btorch also standardises the meaning of `alpha` across all surrogates:

> **`alpha` is the inverse half-width at half-maximum (HWHM). For any
> surrogate, `g(1/alpha) = 0.5` when `damping_factor = 1`.**

This means equal `alpha` gives equal gradient width regardless of which
surrogate is used. SpikingJelly and most other libraries do not share this
convention — their `alpha` scales differ across surrogates by up to 4×.

Each surrogate achieves this by absorbing an irrational constant into its
internal argument:

| Surrogate   | Internal argument | HWHM analytic |
|-------------|-------------------|---------------|
| Triangle    | `alpha·v / 2`     | `1/alpha` (exact) |
| Sigmoid     | `2ln(√2+1)·alpha·v` ≈ `1.763·alpha·v` | `1/alpha` (exact) |
| Erf         | `alpha·v`          | `1/alpha` (exact, `g = 2^{−(αv)²}`) |
| ATan        | `alpha·v`          | `1/alpha` (exact) |
| ATanApprox  | `alpha·v`          | `≈ 0.92/alpha` (approx, rational approx error) |
| SuperSpike  | `(√2−1)·alpha·v` ≈ `0.414·alpha·v` | `1/alpha` (exact) |

## Usage

Most neuron constructors accept a `surrogate_function` argument:

```python
from btorch.models.neurons import LIF
from btorch.models.surrogate import ATan, Erf

# Default ATan, HWHM = 1/2 = 0.5
neuron = LIF(n_neuron=100, surrogate_function=ATan(alpha=2.0))

# Erf matching the V1 model of Chen et al. (2022)
neuron = LIF(n_neuron=100, surrogate_function=Erf(alpha=4.0, damping_factor=0.5))
```

## Choosing a Surrogate

- **ATan** — Cauchy/Lorentz kernel; smooth with polynomial tails. Good general default.
- **ATanApprox** — Rational approximation of ATan; avoids the `atan` call.
- **Sigmoid** — Exponential tails; stronger gradient signal far from threshold.
- **Triangle** — Compact support (zero outside `|v| > 2/alpha`); computationally cheap.
- **Erf** — Gaussian tails; sub-exponential decay, very local gradient.
  Default alpha=4 matches the V1 model (Chen et al. 2022).
- **SuperSpike** — Power-law (heavy) tails; useful for irregular or sparse activity
  (Zenke & Ganguli 2018).

## Adding a New Surrogate

Subclass `SurrogateFunctionBase` and implement `primitive` and `derivative`.
Both conventions must be satisfied before submitting:

```python
import torch

x = torch.tensor(0.0, requires_grad=True)
MySurrogate(alpha=1.0, damping_factor=1.0)(x).backward()
assert abs(x.grad.item() - 1.0) < 1e-5, "peak normalisation failed"

x = torch.tensor(1.0, requires_grad=True)  # v = 1/alpha at alpha=1
MySurrogate(alpha=1.0, damping_factor=1.0)(x).backward()
assert abs(x.grad.item() - 0.5) < 0.02, "HWHM convention failed"
```

The tests `test_unit_gradient_at_threshold` and `test_consistent_hwhm` in
`tests/models/test_surrogate.py` enforce both conventions for all built-in
surrogates automatically.

## Migration guide

### From SpikingJelly

SpikingJelly's `alpha` does not have a consistent meaning across surrogates —
the gradient width and peak at threshold both scale with `alpha` in
surrogate-specific ways. btorch fixes both (peak always 1, HWHM always 1/alpha).

To preserve the **same gradient width** when porting, convert the SpikingJelly
`alpha_sj` to btorch `alpha_bt` using:

| SJ surrogate | SJ HWHM | btorch equivalent | Conversion |
|---|---|---|---|
| `Sigmoid(alpha_sj)` | `1.763/alpha_sj` | `Sigmoid` | `alpha_bt = 1.763 * alpha_sj` |
| `ATan(alpha_sj)` | `2/(π·alpha_sj)` | `ATan` | `alpha_bt = 2/π · alpha_sj ≈ 0.637 * alpha_sj` |
| `Triangle(alpha_sj)` | `1/alpha_sj` | `Triangle` | `alpha_bt = alpha_sj` (same) |

To preserve the **same peak magnitude** at the threshold, set
`damping_factor = old_peak / 1.0`:

| SJ surrogate | SJ peak at v=0 | btorch `damping_factor` |
|---|---|---|
| `Sigmoid(alpha_sj)` | `alpha_sj / 4` | `alpha_sj / 4` |
| `ATan(alpha_sj)` | `alpha_sj / 2` | `alpha_sj / 2` |
| `Triangle(alpha_sj)` | `alpha_sj` | `alpha_sj` |

**Example** — porting `ATan(alpha=2)` from SpikingJelly:

```python
# SpikingJelly: HWHM = 2/(π·2) ≈ 0.318, peak = 2/2 = 1.0
# btorch equivalent preserving both width and magnitude:
from btorch.models.surrogate import ATan
import math
alpha_sj = 2.0
surrogate = ATan(alpha=2/math.pi * alpha_sj, damping_factor=alpha_sj/2)
# ATan(alpha≈0.637, damping_factor=1.0) — peak stays 1, HWHM stays 0.318
```

### From braintools / brainstate

braintools uses JAX and a different internal scaling. The surrogates map as
follows (use HWHM = 1/alpha_bt to find the matching btorch alpha):

| braintools surrogate | bt HWHM | btorch equivalent | Conversion |
|---|---|---|---|
| `Sigmoid(alpha_bt_lib)` | `1.763/alpha` | `Sigmoid` | `alpha_bt = 1.763 * alpha` |
| `ATan(alpha_bt_lib)` | `2/(π·alpha)` | `ATan` | `alpha_bt = 2/π · alpha ≈ 0.637 * alpha` |
| `SuperSpike(alpha_bt_lib)` | `(√2−1)/alpha` | `SuperSpike` | `alpha_bt = (√2−1) * alpha ≈ 0.414 * alpha` |
| `PiecewiseQuadratic(alpha_bt_lib)` | `1/alpha` | `Triangle` | `alpha_bt = alpha` (same shape, different name) |
| `PiecewiseExp(alpha_bt_lib)` | `ln2/alpha` | — | No exact btorch equivalent |
| `Erf(alpha_bt_lib)` | `√ln2/alpha` | `Erf` | `alpha_bt = √ln2 * alpha ≈ 0.833 * alpha` |

For the peak magnitude, set `damping_factor` to the braintools peak value:

| braintools surrogate | bt peak at v=0 | btorch `damping_factor` |
|---|---|---|
| `Sigmoid(alpha)` | `alpha/4` | `alpha/4` |
| `ATan(alpha)` | `alpha/2` | `alpha/2` |
| `SuperSpike(alpha)` | `alpha/2` | `alpha/2` |
| `PiecewiseQuadratic(alpha)` | `alpha` | `alpha` |
| `PiecewiseExp(alpha)` | `alpha/2` | `alpha/2` |
| `Erf(alpha)` | `alpha/√π` | `alpha/√π` |

**Example** — porting `Erf(alpha=2)` from braintools:

```python
# braintools: HWHM = sqrt(ln2)/2 ≈ 0.416, peak = 2/sqrt(pi) ≈ 1.128
import math
from btorch.models.surrogate import Erf
alpha_bt_lib = 2.0
surrogate = Erf(
    alpha=math.sqrt(math.log(2)) * alpha_bt_lib,   # ≈ 1.665 → HWHM = 0.416
    damping_factor=alpha_bt_lib / math.sqrt(math.pi),  # ≈ 1.128
)
```

## References

- Zenke, F., & Neftci, E. O. (2021). *The remarkable robustness of surrogate
  gradient learning for instilling complex function in spiking neural networks.*
  Neural Computation, 33(4), 899–925.
- Zenke, F., & Ganguli, S. (2018). *SuperSpike: Supervised learning in
  multi-layer spiking neural networks.* Neural Computation, 30(6), 1514–1541.
- Chen, G., Scherr, F., & Maass, W. (2022). *A data-based large-scale model
  for primary visual cortex enables brain-like robust and versatile visual
  processing.* Science Advances, 8(44), eabq7592.
