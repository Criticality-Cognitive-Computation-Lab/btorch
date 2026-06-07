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

## References

- Zenke, F., & Neftci, E. O. (2021). *The remarkable robustness of surrogate
  gradient learning for instilling complex function in spiking neural networks.*
  Neural Computation, 33(4), 899–925.
- Zenke, F., & Ganguli, S. (2018). *SuperSpike: Supervised learning in
  multi-layer spiking neural networks.* Neural Computation, 30(6), 1514–1541.
- Chen, G., Scherr, F., & Maass, W. (2022). *A data-based large-scale model
  for primary visual cortex enables brain-like robust and versatile visual
  processing.* Science Advances, 8(44), eabq7592.
