# fast-time2vec-torch
A high-performance, fully vectorized PyTorch implementation of the Time2Vec temporal embedding layer.

[![CI](https://github.com/fbia/fast-time2vec-torch/actions/workflows/ci.yml/badge.svg)](https://github.com/fbia/fast-time2vec-torch/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org)

An efficient PyTorch implementation of the **Time2Vec** temporal embedding layer (*[Time2Vec: Learning a Vector Representation of Time](https://arxiv.org/abs/1907.05321)*, Kazemi et al., 2019), built as a single fused projection.

## 📦 Installation

```bash
pip install git+https://github.com/fbia/fast-time2vec-torch.git
```

While on `0.x` the API may change between minor versions — pin to a tag or commit for reproducible builds, e.g. `...fast-time2vec-torch.git@<tag-or-sha>`.

## 🚀 Quick Start

```python
import torch
from time2vec import EfficientTime2Vec

t2v = EfficientTime2Vec(out_features=16)

timestamps = torch.rand(32, 100, 1)  # [Batch, Seq, 1]
embeddings = t2v(timestamps)         # -> [32, 100, 16]
```

`out_features` is the total embedding size: 1 linear term + `out_features - 1` periodic terms. The input's last dimension must be 1 (a scalar time feature).

> **Tip:** scale/normalize your time input. The linear channel (`ω₀·τ + φ₀`) is
> unbounded, so very large `τ` can dominate the embedding and destabilize training.
> Normalized timestamps — together with the small frequency init — train more
> stably ([discussion](https://github.com/ojus1/Time2Vec-PyTorch)).

## ✨ Features

* **Mathematical alignment**: one linear term plus sinusoidal terms, matching the paper's scalar-frequency mapping.
* **Single fused projection**: linear and periodic components share one `nn.Linear`, so the whole layer is a single matrix multiplication — no `repeat_interleave` or redundant tensor copies.
* **Device & dtype agnostic**: runs on CPU, GPU, and MPS without hardcoded device strings — moves with `.to(device)`, or construct directly via `EfficientTime2Vec(k, device=..., dtype=...)`.
* **Deployment ready**: standard `reset_parameters()`/`extra_repr()`, runs under `autocast`/bf16/fp16, and is both `torch.compile`- and `torch.jit.script`-compatible.
* **Sensible init**: periodic frequencies init small (`N(0, 0.1)`) with random phases (`U(0, 2π)`); the linear term is pinned to identity (`ω₀=1, φ₀=0`) — the common Time2Vec recipe for learned frequencies, avoiding premature high-frequency oscillation.

## ⚡ Performance

The layer is one `nn.Linear` projection followed by `sin` applied to the
**contiguous** projection — a `sin` over a *strided* slice is dramatically slower
on GPU backends. On an Apple M-series GPU (PyTorch MPS), median forward latency
over 100 runs for a `[256, 512, 1]` batch:

| out_features | this layer | [ojus1/Time2Vec-PyTorch](https://github.com/ojus1/Time2Vec-PyTorch) | strided-`sin` variant |
|---:|---:|---:|---:|
| 64  | **2.1 ms** | 4.4 ms | 15.7 ms |
| 128 | **3.1 ms** | 8.0 ms | 37.9 ms |

CPU latency is comparable across implementations; the win is on GPU. Numbers are
hardware-dependent — reproduce with:

```python
import time, torch
from time2vec import EfficientTime2Vec

layer = EfficientTime2Vec(64).to("mps")
x = torch.randn(256, 512, 1, device="mps")
for _ in range(30):
    layer(x)  # warmup
torch.mps.synchronize()
t0 = time.perf_counter()
for _ in range(100):
    layer(x)
torch.mps.synchronize()
print(f"{(time.perf_counter() - t0) / 100 * 1e3:.3f} ms/forward")
```

## 🔧 Development

This project uses [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/fbia/fast-time2vec-torch.git
cd fast-time2vec-torch
uv sync          # create the environment and install dev dependencies
uv run pytest    # run the test suite
uv run mypy      # static type check (strict)
uv build         # build wheel + sdist
```

## ⚖️ License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 👤 Author

Francesco Brundu — [LinkedIn](https://www.linkedin.com/in/fbrundu)
