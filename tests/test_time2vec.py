import torch
import torch.nn as nn
import pytest
from time2vec import EfficientTime2Vec

def test_output_shape() -> None:
    """Verifies that the layer outputs the requested feature dimension."""
    batch, seq, out_features = 16, 50, 8
    layer = EfficientTime2Vec(out_features=out_features)
    x = torch.randn(batch, seq, 1)
    
    out = layer(x)
    assert out.shape == (batch, seq, out_features)

def test_gradient_flow() -> None:
    """Ensures weights and biases receive gradients during backward pass."""
    layer = EfficientTime2Vec(out_features=4)
    x = torch.randn(4, 10, 1)
    
    out = layer(x)
    loss = out.sum()
    loss.backward()
    
    assert layer.project.weight.grad is not None
    assert layer.project.bias.grad is not None

def test_matches_paper_formula() -> None:
    """Verifies the output is the linear term plus sin of the remaining projection."""
    torch.manual_seed(0)
    layer = EfficientTime2Vec(out_features=5)
    x = torch.randn(3, 7, 1)

    out = layer(x)
    proj = layer.project(x)

    assert torch.allclose(out[..., 0:1], proj[..., 0:1])
    assert torch.allclose(out[..., 1:], torch.sin(proj[..., 1:]))

def test_out_features_one_is_linear_only() -> None:
    """With out_features=1 there are no periodic terms, so the output is the raw projection."""
    layer = EfficientTime2Vec(out_features=1)
    x = torch.randn(2, 3, 1)

    out = layer(x)

    assert out.shape == (2, 3, 1)
    assert torch.allclose(out, layer.project(x))

def test_rejects_invalid_out_features() -> None:
    """out_features below 1 is rejected with a ValueError."""
    with pytest.raises(ValueError):
        EfficientTime2Vec(out_features=0)

def test_linear_term_is_pinned_at_init() -> None:
    """The linear channel initializes to identity (omega0=1, phi0=0)."""
    layer = EfficientTime2Vec(out_features=8)

    assert layer.project.weight[0, 0].item() == 1.0
    assert layer.project.bias[0].item() == 0.0

def test_layer_learns() -> None:
    """The layer plus a linear readout fits a periodic+linear signal via gradient descent."""
    torch.manual_seed(0)
    out_features = 32
    model = nn.Sequential(EfficientTime2Vec(out_features), nn.Linear(out_features, 1))

    t = torch.linspace(0.0, 1.0, 256).unsqueeze(-1)
    target = torch.sin(2.0 * torch.pi * t) + 0.5 * t

    opt = torch.optim.Adam(model.parameters(), lr=0.05)
    loss_fn = nn.MSELoss()

    initial = loss_fn(model(t), target).item()
    for _ in range(500):
        opt.zero_grad()
        loss = loss_fn(model(t), target)
        loss.backward()
        opt.step()
    final = loss_fn(model(t), target).item()

    assert final < initial
    assert final < 0.05

def test_rejects_wrong_input_shape() -> None:
    """A last dimension other than 1 raises a clear ValueError."""
    layer = EfficientTime2Vec(out_features=8)
    with pytest.raises(ValueError):
        layer(torch.randn(4, 3))

def test_constructs_with_dtype() -> None:
    """The dtype factory kwarg places parameters in the requested dtype."""
    layer = EfficientTime2Vec(out_features=8, dtype=torch.bfloat16)

    assert layer.project.weight.dtype == torch.bfloat16

def test_repr_includes_out_features() -> None:
    """The module repr surfaces its configuration."""
    layer = EfficientTime2Vec(out_features=16)

    assert "out_features=16" in repr(layer)

def test_is_torchscriptable() -> None:
    """The layer compiles with TorchScript and matches eager output (deployment path)."""
    layer = EfficientTime2Vec(out_features=8)
    scripted = torch.jit.script(layer)
    x = torch.randn(2, 5, 1)

    assert torch.allclose(scripted(x), layer(x))

def test_runs_in_bfloat16() -> None:
    """The layer preserves dtype and produces finite outputs in bfloat16."""
    layer = EfficientTime2Vec(out_features=16).to(torch.bfloat16)
    x = torch.randn(4, 8, 1, dtype=torch.bfloat16)

    out = layer(x)

    assert out.dtype == torch.bfloat16
    assert torch.isfinite(out).all()

def test_runs_under_autocast() -> None:
    """Forward works under CPU autocast (mixed-precision training path)."""
    layer = EfficientTime2Vec(out_features=16)
    x = torch.randn(4, 8, 1)

    with torch.autocast("cpu", dtype=torch.bfloat16):
        out = layer(x)

    assert torch.isfinite(out).all()

def _accelerator() -> str | None:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return None

@pytest.mark.skipif(_accelerator() is None, reason="no CUDA/MPS accelerator available")
def test_runs_on_accelerator() -> None:
    """The layer runs on the available accelerator with outputs on the same device."""
    device = _accelerator()
    assert device is not None
    layer = EfficientTime2Vec(out_features=16).to(device)
    x = torch.randn(4, 8, 1, device=device)

    out = layer(x)

    assert out.device.type == device
    assert out.shape == (4, 8, 16)
    assert torch.isfinite(out).all()

def test_arbitrary_dimensions() -> None:
    """Validates that the layer natively handles high-dimensional tensors."""
    layer = EfficientTime2Vec(out_features=6)
    x = torch.randn(2, 3, 4, 1)  # 4D tensor input

    out = layer(x)
    assert out.shape == (2, 3, 4, 6)
