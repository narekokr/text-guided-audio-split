import torch
import torch.nn as nn

class AdaLN(nn.Module):
    def __init__(self, num_channels, embed_dim):
        super().__init__()
        self.norm = nn.LayerNorm(num_channels)
        self.linear = nn.Linear(embed_dim, num_channels * 2)
        self.embed_dim = embed_dim

    def forward(self, x, cond):
        # x: (B, C, T) or (B, C, F, T)
        orig_shape = x.shape
        if x.dim() == 4:  # reshape to (B*F, T, C)
            B, C, F, T = x.shape
            x = x.permute(0, 2, 3, 1).reshape(B * F * T, C)
        elif x.dim() == 3:
            x = x.permute(0, 2, 1)  # (B, T, C)

        gamma_beta = self.linear(cond)  # (B, 2C)
        gamma, beta = gamma_beta.chunk(2, dim=-1)  # (B, C), (B, C)

        normed = self.norm(x)  # Apply LayerNorm
        out = gamma.unsqueeze(1) * normed + beta.unsqueeze(1)

        if x.dim() == 3:
            out = out.permute(0, 2, 1)  # back to (B, C, T)
        elif x.dim() == 2:
            out = out.view(orig_shape)
        return out