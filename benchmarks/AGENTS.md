## GPU kernel DSLs (Warp, Triton, Gluon, CuPy)

- **Always consult the online docs/examples before asserting what a DSL can or
  cannot do — never claim "X can't do Y" from memory.** This especially applies
  to NVIDIA **Warp** (warp-lang): search its docs
  (`nvidia.github.io/warp`), the tile-programming blog, and the
  `warp/examples/tile/*` examples. Repeated real cases where the "Warp can't"
  assumption was wrong: Warp *does* have cooperative tile reductions
  (`wp.tile_sum` / `wp.tile_reduce`), per-thread↔tile conversion (`wp.tile` /
  `wp.untile`), warp shuffles under the hood, and `wp.Tape` autodiff (differentiate
  the kernel, don't hand-write a backward — even indexed gathers/atomic scatters
  get correct adjoints).
- Prefer the DSL's intended idiom over hand-rolled workarounds: cooperative
  reductions over atomics, `wp.Tape`/autograd over manual backward kernels,
  CSR-vector (warp-per-row) over thread-per-row for irregular SpMV.
- When unsure of an API signature, read the installed package source only as a
  last resort after the docs; verify by launching a tiny real kernel (Warp does
  not compile kernels defined in `exec`'d strings — use a file).

