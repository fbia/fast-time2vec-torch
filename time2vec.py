import torch
import torch.nn as nn

class EfficientTime2Vec(nn.Module):
    def __init__(self, out_features: int):
        super().__init__()
        self.project = nn.Linear(1, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        projection = self.project(x)
        linear_part = projection[..., 0:1]
        periodic_part = torch.sin(projection[..., 1:])
        return torch.cat([linear_part, periodic_part], dim=-1)

if __name__ == "__main__":
    # Quick sanity check
    layer = EfficientTime2Vec(out_features=16)
    
    # Simulate: [Batch Size = 32, Sequence Length = 100, Time Feature = 1]
    dummy_input = torch.randn(32, 100, 1)
    output = layer(dummy_input)
    
    print("Execution successful!")
    print(f"Input shape:  {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
