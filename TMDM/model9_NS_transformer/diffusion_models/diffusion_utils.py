import math
import torch
import numpy as np


def make_beta_schedule(schedule="linear", num_timesteps=1000, start=1e-5, end=1e-2):
    if schedule == "linear":
        betas = torch.linspace(start, end, num_timesteps)
    elif schedule == "const":
        betas = end * torch.ones(num_timesteps)
    elif schedule == "quad":
        betas = torch.linspace(start ** 0.5, end ** 0.5, num_timesteps) ** 2
    elif schedule == "jsd":
        betas = 1.0 / torch.linspace(num_timesteps, 1, num_timesteps)
    elif schedule == "sigmoid":
        betas = torch.linspace(-6, 6, num_timesteps)
        betas = torch.sigmoid(betas) * (end - start) + start
    elif schedule == "cosine" or schedule == "cosine_reverse":
        max_beta = 0.999
        cosine_s = 0.008
        betas = torch.tensor(
            [min(
                1 - (math.cos(((i + 1) / num_timesteps + cosine_s) / (1 + cosine_s) * math.pi / 2) ** 2) /
                (math.cos((i / num_timesteps + cosine_s) / (1 + cosine_s) * math.pi / 2) ** 2),
                max_beta
            ) for i in range(num_timesteps)]
        )
        if schedule == "cosine_reverse":
            betas = betas.flip(0)
    elif schedule == "cosine_anneal":
        betas = torch.tensor(
            [start + 0.5 * (end - start) * (1 - math.cos(t / (num_timesteps - 1) * math.pi))
             for t in range(num_timesteps)]
        )
    return betas


def extract(input_tensor, t, x):
    shape = x.shape
    out = torch.gather(input_tensor, 0, t.to(input_tensor.device))
    reshape = [t.shape[0]] + [1] * (len(shape) - 1)
    return out.reshape(*reshape)


def q_sample_residual(r0, r_prior, alphas_bar_sqrt, one_minus_alphas_bar_sqrt, t, noise=None):
    """
    q(r_t | r_0, r_prior)
    r_t = sqrt(alpha_bar_t) * r0 + (1 - sqrt(alpha_bar_t)) * r_prior + sqrt(1-alpha_bar_t) * noise
    """
    if noise is None:
        noise = torch.randn_like(r0).to(r0.device)

    sqrt_alpha_bar_t = extract(alphas_bar_sqrt, t, r0)
    sqrt_one_minus_alpha_bar_t = extract(one_minus_alphas_bar_sqrt, t, r0)

    r_t = sqrt_alpha_bar_t * r0 + (1 - sqrt_alpha_bar_t) * r_prior + sqrt_one_minus_alpha_bar_t * noise
    return r_t


def p_sample_residual(model, x, x_mark, r_t, r_prior, t, alphas, one_minus_alphas_bar_sqrt):
    """
    One reverse step for residual diffusion.
    model predicts eps_theta.
    """
    device = next(model.parameters()).device
    z = torch.randn_like(r_t)
    t_tensor = torch.tensor([t]).to(device)

    alpha_t = extract(alphas, t_tensor, r_t)
    sqrt_one_minus_alpha_bar_t = extract(one_minus_alphas_bar_sqrt, t_tensor, r_t)
    sqrt_one_minus_alpha_bar_t_m_1 = extract(one_minus_alphas_bar_sqrt, t_tensor - 1, r_t)
    sqrt_alpha_bar_t = (1 - sqrt_one_minus_alpha_bar_t.square()).sqrt()
    sqrt_alpha_bar_t_m_1 = (1 - sqrt_one_minus_alpha_bar_t_m_1.square()).sqrt()

    gamma_0 = (1 - alpha_t) * sqrt_alpha_bar_t_m_1 / (sqrt_one_minus_alpha_bar_t.square())
    gamma_1 = (sqrt_one_minus_alpha_bar_t_m_1.square()) * (alpha_t.sqrt()) / (sqrt_one_minus_alpha_bar_t.square())
    gamma_2 = 1 + (sqrt_alpha_bar_t - 1) * (alpha_t.sqrt() + sqrt_alpha_bar_t_m_1) / (
        sqrt_one_minus_alpha_bar_t.square()
    )

    eps_theta = model(x, x_mark, 0, r_t, r_prior, t_tensor).to(device).detach()

    r0_reparam = 1 / sqrt_alpha_bar_t * (
        r_t - (1 - sqrt_alpha_bar_t) * r_prior - eps_theta * sqrt_one_minus_alpha_bar_t
    )

    r_t_m_1_hat = gamma_0 * r0_reparam + gamma_1 * r_t + gamma_2 * r_prior
    beta_t_hat = (sqrt_one_minus_alpha_bar_t_m_1.square()) / (sqrt_one_minus_alpha_bar_t.square()) * (1 - alpha_t)
    r_t_m_1 = r_t_m_1_hat.to(device) + beta_t_hat.sqrt().to(device) * z.to(device)
    return r_t_m_1


def p_sample_residual_t_1to0(model, x, x_mark, r_t, r_prior, one_minus_alphas_bar_sqrt):
    device = next(model.parameters()).device
    t_tensor = torch.tensor([0]).to(device)

    sqrt_one_minus_alpha_bar_t = extract(one_minus_alphas_bar_sqrt, t_tensor, r_t)
    sqrt_alpha_bar_t = (1 - sqrt_one_minus_alpha_bar_t.square()).sqrt()

    eps_theta = model(x, x_mark, 0, r_t, r_prior, t_tensor).to(device).detach()

    r0_reparam = 1 / sqrt_alpha_bar_t * (
        r_t - (1 - sqrt_alpha_bar_t) * r_prior - eps_theta * sqrt_one_minus_alpha_bar_t
    )
    return r0_reparam.to(device)


def p_sample_loop_residual(model, x, x_mark, r_prior, n_steps, alphas, one_minus_alphas_bar_sqrt):
    """
    Sample from p(r_0 | x) by initializing r_T ~ N(r_prior, I)
    """
    device = next(model.parameters()).device
    z = torch.randn_like(r_prior).to(device)
    cur_r = z + r_prior
    r_seq = [cur_r]

    for t in reversed(range(1, n_steps)):
        r_t = cur_r
        cur_r = p_sample_residual(
            model, x, x_mark, r_t, r_prior, t, alphas, one_minus_alphas_bar_sqrt
        )
        r_seq.append(cur_r)

    assert len(r_seq) == n_steps
    r0 = p_sample_residual_t_1to0(
        model, x, x_mark, r_seq[-1], r_prior, one_minus_alphas_bar_sqrt
    )
    r_seq.append(r0)
    return r_seq


def kld(y1, y2, grid=(-20, 20), num_grid=400):
    y1, y2 = y1.numpy().flatten(), y2.numpy().flatten()
    p_y1, _ = np.histogram(y1, bins=num_grid, range=[grid[0], grid[1]], density=True)
    p_y1 += 1e-7
    p_y2, _ = np.histogram(y2, bins=num_grid, range=[grid[0], grid[1]], density=True)
    p_y2 += 1e-7
    return (p_y1 * np.log(p_y1 / p_y2)).sum()
