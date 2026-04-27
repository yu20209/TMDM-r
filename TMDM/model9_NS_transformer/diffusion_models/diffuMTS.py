import torch
import torch.nn as nn
from layers.Embed import DataEmbedding
import yaml
import argparse
from model9_NS_transformer.diffusion_models.diffusion_utils import *
from model9_NS_transformer.diffusion_models.residual_patch_denoiser import ResidualPatchDenoiser


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


class Model(nn.Module):
    def __init__(self, configs, device):
        super(Model, self).__init__()

        with open(configs.diffusion_config_dir, "r") as f:
            config = yaml.unsafe_load(f)
            diffusion_config = dict2namespace(config)

        diffusion_config.diffusion.timesteps = configs.timesteps
        
        self.args = configs
        self.device = device
        self.diffusion_config = diffusion_config

        self.num_timesteps = diffusion_config.diffusion.timesteps

        betas = make_beta_schedule(schedule=diffusion_config.diffusion.beta_schedule, num_timesteps=self.num_timesteps,
                                   start=diffusion_config.diffusion.beta_start, end=diffusion_config.diffusion.beta_end)
        betas = self.betas = betas.float().to(self.device)
        alphas = 1.0 - betas
        self.alphas = alphas
        alphas_cumprod = alphas.to('cpu').cumprod(dim=0).to(self.device)
        self.alphas_bar_sqrt = torch.sqrt(alphas_cumprod)
        self.one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_cumprod)

        # NEW: replace MLP with patch transformer denoiser
        self.denoiser = ResidualPatchDenoiser(configs, self.num_timesteps)

    def forward(self, x, x_mark, y_base, r_t, r_prior, t):
        # ignore r_prior for now (set =0)
        eps = self.denoiser(x, y_base, r_t, t)
        return eps
