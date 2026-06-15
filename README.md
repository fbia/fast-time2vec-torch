# fast-time2vec-torch
A high-performance, fully vectorized PyTorch implementation of the Time2Vec temporal embedding layer.

[![License: MIT](https://shields.io)](https://opensource.org)
[![PyTorch](https://shields.io)](https://pytorch.org)

A production-grade, highly efficient implementation of the **Time2Vec** temporal embedding layer in PyTorch. 

This repository fixes the performance bottlenecks, over-parameterization, and device-locking bugs found in other community implementations. It perfectly mirrors the mathematical formulation proposed in the original paper: *[Time2Vec: Learning a Vector Representation of Time](https://arxiv.org)*.

## 🚀 Key Features

* **Strict Mathematical Alignment**: Uses the exact scalar-frequency mapping from the paper.
* **Hardware Accelerated**: Consolidated into a single matrix multiplication.
* **Device Agnostic**: Natively supports CPU, GPU, and MPS without hardcoded strings.
* **Memory Efficient**: Zero redundant tensor replications or `repeat_interleave` calls.

## 📦 Installation & Usage

Copy the module directly into your project:

```python
import torch
import torch.nn as nn

class EfficientTime2Vec(nn.Module):
    def __init__(self, out_features: int):
        super().__init__()
        # Combines linear and periodic weights into one step
        self.project = nn.Linear(1, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Expected input shape: [Batch, Seq, 1]
        projection = self.project(x)
        
        linear_part = projection[..., 0:1]
        periodic_part = torch.sin(projection[..., 1:])
        
        return torch.cat([linear_part, periodic_part], dim=-1)

# Example initialization
t2v = EfficientTime2Vec(out_features=16)
dummy_timestamps = torch.rand(32, 100, 1) # [Batch, Seq, Feature]
embeddings = t2v(dummy_timestamps)        # Returns [32, 100, 16]
```

## ⚖️ License

This project is licensed under the MIT License - see the LICENSE file for details.
