import torch
import torch.nn as nn

class AdaLN(nn.Module):
    def __init__(self, num_channels, embed_dim):
        super().__init__()
        self.norm = nn.LayerNorm(num_channels)
        self.linear = nn.Linear(embed_dim, num_channels * 2)
        self.embed_dim = embed_dim

    def forward(self, x, cond):
        assert cond is not None, "AdaLN: received cond=None!"
        orig_shape = x.shape
        if x.dim() == 4:
            B, C, F, T = x.shape
            x_ = x.permute(0, 2, 3, 1).reshape(-1, C)  # [B*F*T, C]
            normed = self.norm(x_)
            normed = normed.view(B, F, T, C).permute(0, 3, 1, 2)  # [B, C, F, T]
        elif x.dim() == 3:
            B, C, T = x.shape
            x_ = x.permute(0, 2, 1).reshape(-1, C)  # [B*T, C]
            normed = self.norm(x_)
            normed = normed.view(B, T, C).permute(0, 2, 1)  # [B, C, T]
        else:
            raise RuntimeError("AdaLN: x must be 3D or 4D")

        # cond: [B, embed_dim]
        gamma_beta = self.linear(cond)  # [B, 2C]
        gamma, beta = gamma_beta.chunk(2, dim=-1)  # ([B, C], [B, C])

        # Expand for time/freq dimensions
        if x.dim() == 4:
            gamma = gamma[:, :, None, None]
            beta = beta[:, :, None, None]
        else:
            gamma = gamma[:, :, None]
            beta = beta[:, :, None]

        out = gamma * normed + beta
        return out