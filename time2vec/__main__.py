import torch

from time2vec import EfficientTime2Vec

# Quick sanity check
layer = EfficientTime2Vec(out_features=16)

# Simulate: [Batch Size = 32, Sequence Length = 100, Time Feature = 1]
dummy_input = torch.randn(32, 100, 1)
output = layer(dummy_input)

print("Execution successful!")
print(f"Input shape:  {dummy_input.shape}")
print(f"Output shape: {output.shape}")
