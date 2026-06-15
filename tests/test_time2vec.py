import torch
import pytest
from time2vec import EfficientTime2Vec

def test_output_shape():
    """Verifies that the layer outputs the requested feature dimension."""
    batch, seq, out_features = 16, 50, 8
    layer = EfficientTime2Vec(out_features=out_features)
    x = torch.randn(batch, seq, 1)
    
    out = layer(x)
    assert out.shape == (batch, seq, out_features)

def test_gradient_flow():
    """Ensures weights and biases receive gradients during backward pass."""
    layer = EfficientTime2Vec(out_features=4)
    x = torch.randn(4, 10, 1)
    
    out = layer(x)
    loss = out.sum()
    loss.backward()
    
    assert layer.project.weight.grad is not None
    assert layer.project.bias.grad is not None

def test_mathematical_split():
    """Confirms the periodic outputs strictly respect sine wave boundaries."""
    layer = EfficientTime2Vec(out_features=4)
    x = torch.tensor([[[100.0]]])  # Large input value
    
    out = layer(x)
    periodic_part = out[..., 1:]
    
    assert torch.all(periodic_part >= -1.0)
    assert torch.all(periodic_part <= 1.0)

def test_arbitrary_dimensions():
    """Validates that the layer natively handles high-dimensional tensors."""
    layer = EfficientTime2Vec(out_features=6)
    x = torch.randn(2, 3, 4, 1)  # 4D tensor input
    
    out = layer(x)
    assert out.shape == (2, 3, 4, 6)
