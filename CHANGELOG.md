# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and
this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
While on `0.x` the public API may change between minor versions; pin to a tag or
commit for reproducible builds.

## [0.1.0]

Initial release — a vectorized PyTorch implementation of the Time2Vec temporal
embedding layer (Kazemi et al., 2019, https://arxiv.org/abs/1907.05321).

### Added
- `Time2Vec` layer: a single `nn.Linear(1, k)` projection followed by
  `sin` on the periodic channels, device- and shape-agnostic over leading
  dimensions, with `sin` applied to the contiguous projection for GPU efficiency.
- Learned frequencies/phases: periodic frequencies initialize small (`N(0, 0.1)`)
  with random phases (`U(0, 2π)`); the linear term is pinned to identity
  (`ω₀=1, φ₀=0`).
- PyTorch-idiomatic API: `reset_parameters()`, `extra_repr()`, an `out_features`
  attribute, `device`/`dtype` factory keyword arguments, and `forward`
  input-shape validation.
- Compatibility: eager, `autocast`/bf16/fp16, CUDA/MPS, `torch.compile`, and
  TorchScript (`torch.jit.script`/`save`/`load`).
- Inline type hints with a `py.typed` marker (strict `mypy` clean).
- Test suite (15 tests): output shape, gradient flow, paper-formula equivalence,
  `out_features=1` and bad-input edge cases, argument validation, init invariant,
  training convergence, precision/device coverage, `repr`, and scriptability.
- Packaging via the `uv_build` backend (wheel + sdist), Apache-2.0 licensed.
- GitHub Actions CI: Python 3.10–3.13 matrix plus a lowest-dependency job that
  validates the declared `torch>=2.0` / `pytest>=8` floor.
