import torch
import torch.nn as nn

class ResidualPatchDenoiser(nn.Module):
    def __init__(self, args, num_timesteps):
        super().__init__()
        self.patch_len = getattr(args, 'patch_len', 16)
        self.stride = getattr(args, 'stride', 8)
        self.d_model = getattr(args, 'd_model', 256)
        self.pred_len = args.pred_len
        self.seq_len = args.seq_len
        self.c_out = args.c_out

        self.time_embed = nn.Embedding(num_timesteps + 1, self.d_model)

        self.patch_embed = nn.Linear(self.patch_len, self.d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=8,
            dim_feedforward=self.d_model * 4,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

        self.head = nn.Linear(self.d_model, self.patch_len)

    def patchify(self, x):
        # x: [B, L]
        patches = x.unfold(-1, self.patch_len, self.stride)
        return patches

    def unpatchify(self, patches, L):
        B, N, P = patches.shape
        out = torch.zeros(B, L, device=patches.device)
        count = torch.zeros(B, L, device=patches.device)
        idx = 0
        for i in range(N):
            start = i * self.stride
            out[:, start:start+self.patch_len] += patches[:, i]
            count[:, start:start+self.patch_len] += 1
        return out / (count + 1e-6)

    def forward(self, x, y_base, r_t, t):
        B, L, C = r_t.shape

        r = r_t.permute(0,2,1).reshape(B*C, L)
        base = y_base.permute(0,2,1).reshape(B*C, L)
        hist = x.permute(0,2,1).reshape(B*C, -1)

        r_p = self.patchify(r)
        base_p = self.patchify(base)

        r_tok = self.patch_embed(r_p)
        base_tok = self.patch_embed(base_p)

        t_emb = self.time_embed(t).repeat_interleave(C, dim=0).unsqueeze(1)

        tok = r_tok + base_tok + t_emb

        h = self.encoder(tok)

        eps_patch = self.head(h)
        eps = self.unpatchify(eps_patch, L)

        eps = eps.reshape(B, C, L).permute(0,2,1)
        return eps
