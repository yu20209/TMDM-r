import torch
import torch.nn as nn


class ResidualPriorNet(nn.Module):
    """
    Predict residual prior center R_prior from encoder features and base forecast.

    Input:
        enc_feat: [B, S, D] or [B, L, D]
        y_base:   [B, L, C]

    Output:
        r_prior:  [B, L, C]
    """
    def __init__(self, d_model: int, c_out: int, hidden_dim: int = 256):
        super().__init__()
        self.c_out = c_out

        self.enc_proj = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.y_proj = nn.Sequential(
            nn.Linear(c_out, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.out = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, c_out)
        )

    def forward(self, enc_feat: torch.Tensor, y_base: torch.Tensor) -> torch.Tensor:
        # enc_feat: [B, S, D]
        # y_base:   [B, L, C]
        B, L, C = y_base.shape

        pooled = enc_feat.mean(dim=1)               # [B, D]
        pooled = self.enc_proj(pooled)              # [B, H]
        pooled = pooled.unsqueeze(1).repeat(1, L, 1)  # [B, L, H]

        y_emb = self.y_proj(y_base)                 # [B, L, H]

        h = torch.cat([pooled, y_emb], dim=-1)      # [B, L, 2H]
        r_prior = self.out(h)                       # [B, L, C]
        return r_prior
