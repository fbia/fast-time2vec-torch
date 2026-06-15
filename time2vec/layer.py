import torch
import torch.nn as nn


class EfficientTime2Vec(nn.Module):
    """Vectorized Time2Vec temporal embedding layer.

    Implements the representation of Kazemi et al. (2019), "Time2Vec: Learning a
    Vector Representation of Time" (https://arxiv.org/abs/1907.05321): for a
    scalar time input the output is one linear term followed by
    ``out_features - 1`` sinusoidal terms, computed as a single affine
    projection followed by ``sin`` on the periodic channels.

    Expected input shape is ``[..., 1]`` (a scalar time feature in the last
    dimension); the output shape is ``[..., out_features]``, where
    ``out_features`` is the total embedding size: 1 linear term plus
    ``out_features - 1`` periodic terms.

    Periodic frequencies are initialized small (``N(0, 0.1)``) with random phases
    (``U(0, 2*pi)``), and the linear term is pinned to identity (``omega0 = 1``,
    ``phi0 = 0``). Small initial frequencies are the common Time2Vec practice for
    the learned ``omega``/``phi`` parameters and avoid premature high-frequency
    oscillation. Scale/normalize the time input: the linear channel is unbounded,
    so very large inputs can dominate the embedding.
    """

    def __init__(
        self,
        out_features: int,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if out_features < 1:
            raise ValueError(f"out_features must be >= 1, got {out_features}")
        self.out_features = out_features
        self.project = nn.Linear(1, out_features, device=device, dtype=dtype)
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self) -> None:
        """Init periodic frequencies small with random phases; pin the linear term to identity."""
        nn.init.normal_(self.project.weight, std=0.1)
        nn.init.uniform_(self.project.bias, 0.0, 2.0 * torch.pi)
        self.project.weight[0, 0] = 1.0
        self.project.bias[0] = 0.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != 1:
            raise ValueError(
                "EfficientTime2Vec expects the last input dimension to be 1 "
                "(a scalar time feature), got " + str(int(x.shape[-1]))
            )
        projection = self.project(x)
        sined = torch.sin(projection)
        linear_part = projection.narrow(-1, 0, 1)
        periodic_part = sined.narrow(-1, 1, self.out_features - 1)
        return torch.cat([linear_part, periodic_part], dim=-1)

    def extra_repr(self) -> str:
        return f"out_features={self.out_features}"
