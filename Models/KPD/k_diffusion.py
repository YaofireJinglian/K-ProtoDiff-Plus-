import math
import torch
import torch.nn.functional as F
import torch.nn.utils.weight_norm as wn
from torch import nn
from einops import reduce
from tqdm.auto import tqdm
from functools import partial
from Models.KPD.transformer import Transformer
from Models.KPD.model_utils import default, identity, extract
from Layer.KBasisNet import KBasisNet
from Layer.cross_attention import channel_AutoCorrelationLayer
from typing import List
import numpy as np
import pandas as pd
from pandas.tseries import offsets
from pandas.tseries.frequencies import to_offset
import torch.fft

class FDM(torch.nn.Module):
    def __init__(self, fl_ratio=0.1, fh_ratio=0.3, implementation='real'):
        """
        :param fl_ratio: low frequency threshold ratio (e.g., 0.1 means fl = 10% of fmax)
        :param fh_ratio: high frequency threshold ratio (e.g., 0.3 means fh = 30% of fmax)
        """
        super(FDM, self).__init__()
        self.fl_ratio = fl_ratio
        self.fh_ratio = fh_ratio
        self.implementation = implementation

    def forward(self, x):
        """
        :param x: Tensor of shape (B, T, D)
        :return: low_freq, mid_freq, high_freq components, each of shape (B, T, D)
        """
        _, T, _ = x.shape
        if self.implementation == 'legacy':
            x_fft = torch.fft.fft(x, dim=1)
            fl = int(self.fl_ratio * T)
            fh = int(self.fh_ratio * T)

            def legacy_band(start, end):
                mask = torch.zeros_like(x_fft, dtype=torch.bool)
                mask[:, start:end, :] = True
                band = torch.where(mask, x_fft, torch.zeros_like(x_fft))
                return torch.fft.ifft(band, dim=1).real

            return legacy_band(0, fl), legacy_band(fl, fh), legacy_band(fh, T)

        if self.implementation != 'real':
            raise ValueError(f'unknown FDM implementation: {self.implementation}')

        # rFFT/irFFT keeps conjugate symmetry for real-valued time series.  The
        # previous full-FFT masking silently discarded half of boundary bands
        # when the complex result was converted back to real values.
        x_fft = torch.fft.rfft(x, dim=1)

        fmax = x_fft.shape[1]
        fl = max(1, min(int(self.fl_ratio * fmax), fmax - 1))
        fh = max(fl + 1, min(int(self.fh_ratio * fmax), fmax))

        # Zero out irrelevant frequencies for each band
        def band_pass(x_fft, start, end):
            mask = torch.zeros_like(x_fft, dtype=torch.bool)
            mask[:, start:end, :] = True
            x_band = torch.where(mask, x_fft, torch.zeros_like(x_fft))
            return torch.fft.irfft(x_band, n=T, dim=1)

        # Get each frequency band
        low = band_pass(x_fft, 0, fl)
        mid = band_pass(x_fft, fl, fh)
        high = band_pass(x_fft, fh, fmax)

        return low, mid, high

class TaylorKAN(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, Q=4, P=3):
        super(TaylorKAN, self).__init__()
        self.Q = Q
        self.P = P
        self.phi = nn.ModuleList([
            nn.Sequential(
                nn.Linear(P, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            ) for _ in range(Q)
        ])
        self.psi = nn.ModuleList([
            nn.ModuleList([
                nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, 1)
                ) for _ in range(P)
            ]) for _ in range(Q)
        ])

    def forward(self, x):
        # x: (B, T, D)
        B, T, D = x.shape
        outputs = []
        for q in range(self.Q):
            psi_outs = []
            for p in range(self.P):
                # 对每个p进行ψ_q,p
                psi_out = self.psi[q][p](x)  # (B, T, 1)
                psi_outs.append(psi_out)
            psi_concat = torch.cat(psi_outs, dim=-1)  # (B, T, P)
            phi_out = self.phi[q](psi_concat)  # (B, T, 1)
            outputs.append(phi_out)
        return torch.sum(torch.cat(outputs, dim=-1), dim=-1, keepdim=True)  # (B, T, 1）

class MultiOrderKANBlock(nn.Module):
    def __init__(self, input_dim, taylor_hidden=64, Q=4, P=3):

        super(MultiOrderKANBlock, self).__init__()
        self.kan_l = TaylorKAN(input_dim, hidden_dim=taylor_hidden, Q=Q, P=P)
        self.kan_m = TaylorKAN(input_dim, hidden_dim=taylor_hidden, Q=Q, P=P)
        self.kan_h = TaylorKAN(input_dim, hidden_dim=taylor_hidden, Q=Q, P=P)
        self.fusion = nn.Linear(3, input_dim)  # (B,T,3) -> (B,T,input_dim)

    def forward(self, Fo_l, Fo_m, Fo_h):
        l_out = self.kan_l(Fo_l)  # (B, T, 1)
        m_out = self.kan_m(Fo_m)  # (B, T, 1)
        h_out = self.kan_h(Fo_h)  # (B, T, 1)

        concat = torch.cat([l_out, m_out, h_out], dim=-1)  # (B, T, 3)
        out = self.fusion(concat)  # (B, T, input_dim)
        return out

class PrototypeAssignment(nn.Module):
    def __init__(self, d_model, num_prototypes=8):
        super(PrototypeAssignment, self).__init__()
        self.d_model = d_model
        self.num_prototypes = num_prototypes

        self.prototypes = nn.Parameter(torch.randn(num_prototypes, d_model))


        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)

    def forward(self, H):
        B, T, D = H.shape
        Q = self.W_Q(H)               # (B, T, D)
        K = self.W_K(self.prototypes)  # (K, D)
        attn_scores = torch.matmul(Q, K.T) / (D ** 0.5)
        A = F.softmax(attn_scores, dim=-1)  
        Z_p = torch.matmul(A, self.prototypes)  # (B, T, D)
        return Z_p, A


class MultiScaleTemporalPrototypeAssignment(nn.Module):
    """Learn and assign variable-length temporal prototypes.

    Each scale owns a bank of raw time-series snippets rather than a bank of
    point-wise feature vectors.  KAN features are used as queries, while the
    reconstructed prototype context stays in data space so that it can also be
    reused by the adaptive reflection sampler.
    """

    def __init__(self, d_model, scales, num_prototypes=8, stride_ratio=0.5,
                 temperature=0.1):
        super().__init__()
        self.d_model = d_model
        self.scales = tuple(int(scale) for scale in scales)
        self.num_prototypes = int(num_prototypes)
        self.stride_ratio = float(stride_ratio)
        self.temperature = float(temperature)

        self.prototypes = nn.ParameterList([
            nn.Parameter(torch.randn(self.num_prototypes, scale, d_model) * 0.02)
            for scale in self.scales
        ])
        self.query_projections = nn.ModuleList([
            nn.Linear(d_model, d_model) for _ in self.scales
        ])
        self.key_projections = nn.ModuleList([
            nn.Linear(d_model, d_model) for _ in self.scales
        ])
        self.scale_logits = nn.Parameter(torch.zeros(len(self.scales)))
        self.register_buffer('prototypes_initialized', torch.tensor(False))

        # Start as an identity path with a small prototype residual.  This is
        # substantially more stable than replacing the noisy input outright.
        self.fusion = nn.Linear(d_model * 2, d_model)
        with torch.no_grad():
            self.fusion.weight.zero_()
            self.fusion.bias.zero_()
            eye = torch.eye(d_model)
            self.fusion.weight[:, :d_model].copy_(eye)
            self.fusion.weight[:, d_model:].copy_(0.1 * eye)

    @torch.no_grad()
    def initialize_from_batch(self, raw_series):
        """Seed every prototype bank with real windows from the first batch."""
        if bool(self.prototypes_initialized):
            return
        _, length, _ = raw_series.shape
        for scale, prototypes in zip(self.scales, self.prototypes):
            stride = max(1, int(round(scale * self.stride_ratio)))
            starts = self._window_starts(length, scale, stride)
            candidates = torch.cat(
                [raw_series[:, start:start + scale] for start in starts], dim=0
            )
            if len(candidates) >= self.num_prototypes:
                indices = torch.randperm(len(candidates), device=raw_series.device)[
                    :self.num_prototypes
                ]
            else:
                indices = torch.arange(
                    self.num_prototypes, device=raw_series.device
                ) % len(candidates)
            prototypes.copy_(candidates[indices])
        self.prototypes_initialized.fill_(True)

    @staticmethod
    def _window_starts(length, scale, stride):
        starts = list(range(0, length - scale + 1, stride))
        final_start = length - scale
        if starts[-1] != final_start:
            starts.append(final_start)
        return starts

    @staticmethod
    def _overlap_add(patches, starts, length):
        batch, _, scale, channels = patches.shape
        device, dtype = patches.device, patches.dtype
        positions = torch.stack([
            torch.arange(start, start + scale, device=device) for start in starts
        ]).reshape(-1)
        values = patches.reshape(batch, -1, channels)
        output = torch.zeros(batch, length, channels, device=device, dtype=dtype)
        output = output.index_add(1, positions, values)
        counts = torch.zeros(length, device=device, dtype=dtype)
        counts = counts.index_add(0, positions, torch.ones_like(positions, dtype=dtype))
        return output / counts.clamp_min(1.0).view(1, length, 1)

    def forward(self, query_features, raw_series, compute_aux=False):
        batch, length, channels = raw_series.shape
        contexts, assignments = [], []
        reconstruction_losses, balance_losses, diversity_losses = [], [], []

        for scale, prototypes, query_projection, key_projection in zip(
                self.scales, self.prototypes, self.query_projections, self.key_projections):
            stride = max(1, int(round(scale * self.stride_ratio)))
            starts = self._window_starts(length, scale, stride)
            query_windows = torch.stack(
                [query_features[:, start:start + scale] for start in starts], dim=1
            )
            query_windows = query_projection(query_windows).flatten(2)
            prototype_keys = key_projection(prototypes).flatten(1)

            normalized_queries = F.normalize(query_windows, dim=-1, eps=1e-6)
            normalized_keys = F.normalize(prototype_keys, dim=-1, eps=1e-6)
            scores = torch.einsum('bnf,kf->bnk', normalized_queries, normalized_keys)
            assignment = F.softmax(scores / max(self.temperature, 1e-4), dim=-1)
            reconstructed = torch.einsum('bnk,ksd->bnsd', assignment, prototypes)
            context = self._overlap_add(reconstructed, starts, length)

            contexts.append(context)
            assignments.append(assignment)

            if compute_aux:
                reconstruction_losses.append(F.mse_loss(context, raw_series))
                usage = assignment.mean(dim=(0, 1))
                uniform = torch.full_like(usage, 1.0 / self.num_prototypes)
                balance_losses.append(F.mse_loss(usage, uniform) * self.num_prototypes)

                flat_prototypes = F.normalize(prototypes.flatten(1), dim=-1, eps=1e-6)
                gram = flat_prototypes @ flat_prototypes.transpose(0, 1)
                off_diagonal = gram - torch.eye(
                    self.num_prototypes, device=gram.device, dtype=gram.dtype
                )
                diversity_losses.append(off_diagonal.square().mean())

        scale_weights = F.softmax(self.scale_logits, dim=0)
        prototype_context = sum(
            weight * context for weight, context in zip(scale_weights, contexts)
        )
        conditioned_series = self.fusion(torch.cat([raw_series, prototype_context], dim=-1))

        aux = None
        if compute_aux:
            aux = {
                'reconstruction': torch.stack(reconstruction_losses).mean(),
                'balance': torch.stack(balance_losses).mean(),
                'diversity': torch.stack(diversity_losses).mean(),
            }
        return conditioned_series, prototype_context, assignments, aux
class Model(nn.Module):
    def __init__(
            self,
            seq_length,
            feature_size,
            n_layer_enc=3,
            n_layer_dec=6,
            d_model=None,
            timesteps=1000,
            sampling_timesteps=None,
            loss_type='l1',
            beta_schedule='cosine',
            n_heads=4,
            mlp_hidden_times=4,
            eta=0.,
            attn_pd=0.,
            resid_pd=0.,
            kernel_size=None,
            padding_size=None,
            use_ff=True,
            reg_weight=None,
            num_prototypes=8,
            fl_ratio=0.1,
            fh_ratio=0.3,
            fdm_mode='legacy',
            kan_taylor_hidden=64,
            kan_Q=4,
            kan_P=3,
            prototype_mode='point',
            prototype_scales=(4, 8, 16),
            prototype_stride_ratio=0.5,
            prototype_temperature=0.1,
            prototype_loss_weight=0.05,
            prototype_balance_weight=0.1,
            prototype_diversity_weight=0.1,
            prototype_consistency_weight=0.1,
            T_max=1,
            lambda_step_ratio=0.01,
            inv_guidance_scale=0.0,
            sampling_mode='legacy',
            adaptive_sampling_timesteps=50,
            reflection_strength=0.1,
            reflection_threshold=0.15,
            reflection_temperature=0.05,
            reflection_roughness_threshold=1.25,
            reflection_roughness_temperature=0.15,
            reflection_start_ratio=0.5,
            reflection_topk_ratio=0.25,
            reflection_mask_temperature=0.1,
            adaptive_stride_gain=1.0,
            **kwargs
    ):
        super(Model, self).__init__()
        self.d_model = d_model
        self.epsilon = 1E-5
        self.eta, self.use_ff = eta, use_ff
        self.seq_length = seq_length
        self.feature_size = feature_size
        self.ff_weight = default(reg_weight, math.sqrt(self.seq_length) / 5)
        self.model = Transformer(n_feat=feature_size, n_channel=seq_length, n_layer_enc=n_layer_enc, n_layer_dec=n_layer_dec,
                                 n_heads=n_heads, attn_pdrop=attn_pd, resid_pdrop=resid_pd, mlp_hidden_times=mlp_hidden_times,
                                 max_len=seq_length, n_embd=d_model, conv_params=[kernel_size, padding_size], **kwargs)

        # KPL pipeline: FDM -> MultiOrderKAN -> PrototypeAssignment, all over feature_size channels.
        self.fdm = FDM(
            fl_ratio=fl_ratio, fh_ratio=fh_ratio, implementation=fdm_mode
        )
        self.kan_block = MultiOrderKANBlock(
            input_dim=feature_size, taylor_hidden=kan_taylor_hidden, Q=kan_Q, P=kan_P,
        )
        self.prototype_mode = str(prototype_mode)
        if self.prototype_mode == 'multiscale':
            scales = sorted({int(scale) for scale in prototype_scales if 1 < int(scale) <= seq_length})
            if not scales:
                scales = [seq_length]
            self.kpa = MultiScaleTemporalPrototypeAssignment(
                d_model=feature_size,
                scales=scales,
                num_prototypes=num_prototypes,
                stride_ratio=prototype_stride_ratio,
                temperature=prototype_temperature,
            )
        elif self.prototype_mode == 'point':
            self.kpa = PrototypeAssignment(d_model=feature_size, num_prototypes=num_prototypes)
        else:
            raise ValueError(f'unknown prototype mode: {self.prototype_mode}')

        self.prototype_loss_weight = float(prototype_loss_weight)
        self.prototype_balance_weight = float(prototype_balance_weight)
        self.prototype_diversity_weight = float(prototype_diversity_weight)
        self.prototype_consistency_weight = float(prototype_consistency_weight)

        # R-Sampling hyperparameters (only used by full p_sample loop, not DDIM fast_sample).
        self.T_max = int(T_max)
        self.inv_guidance_scale = float(inv_guidance_scale)
        self.lambda_step = float(lambda_step_ratio) * float(timesteps)
        self.sampling_mode = str(sampling_mode)
        self.adaptive_sampling_timesteps = int(adaptive_sampling_timesteps)
        self.reflection_strength = float(reflection_strength)
        self.reflection_threshold = float(reflection_threshold)
        self.reflection_temperature = float(reflection_temperature)
        self.reflection_roughness_threshold = float(reflection_roughness_threshold)
        self.reflection_roughness_temperature = float(reflection_roughness_temperature)
        self.reflection_start_ratio = float(reflection_start_ratio)
        self.reflection_topk_ratio = float(reflection_topk_ratio)
        self.reflection_mask_temperature = float(reflection_mask_temperature)
        self.adaptive_stride_gain = float(adaptive_stride_gain)
        self.last_sampling_stats = None
        self.register_buffer('clean_roughness_ema', torch.tensor(0.0))
        self.register_buffer('clean_roughness_initialized', torch.tensor(False))

        if beta_schedule == 'linear':
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type

        # sampling related parameters

        self.sampling_timesteps = default(
            sampling_timesteps, timesteps)  # default num sampling timesteps to number of timesteps at training

        assert self.sampling_timesteps <= timesteps
        self.fast_sampling = self.sampling_timesteps < timesteps

        # helper function to register buffer from float64 to float32

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))
        register_buffer('timesteps', torch.arange(timesteps, 0, -1))
        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min=1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # calculate reweighting
        
        register_buffer('loss_weight', torch.sqrt(alphas) * torch.sqrt(1. - alphas_cumprod) / betas / 100)

    def predict_noise_from_start(self, x_t, t, x0):
        return (
                (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) /
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )
    
    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
                extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped
    
    def _prototype_condition(self, x, compute_aux=False):
        low_freq, mid_freq, high_freq = self.fdm(x)
        H = self.kan_block(low_freq, mid_freq, high_freq)
        if self.prototype_mode == 'multiscale':
            conditioned, context, assignments, aux = self.kpa(H, x, compute_aux=compute_aux)
        else:
            conditioned, assignments = self.kpa(H)
            context, aux = conditioned, None
        return conditioned, context, assignments, aux

    def output(self, x, x_mark, t, padding_masks=None):
        # KPL pipeline: frequency/KAN queries -> multi-scale temporal prototype context.
        conditioned, context, assignments, _ = self._prototype_condition(x, compute_aux=False)
        self._last_proto_attn = assignments
        self._last_proto_context = context
        model_output = self.model(conditioned, t, padding_masks=padding_masks)
        return model_output

    def model_predictions(self, x, t, clip_x_start=False, padding_masks=None, guidance_scale=None):
        """Modified to support guidance scale in predictions"""
        if padding_masks is None:
            padding_masks = torch.ones(x.shape[0], self.seq_length, dtype=bool, device=x.device)

        maybe_clip = partial(torch.clamp, min=-1., max=1.) if clip_x_start else identity

        x_start = self.output(x, None, t, padding_masks)
        x_start = maybe_clip(x_start)
        pred_noise = self.predict_noise_from_start(x, t, x_start)
        
        # Add guidance scale support if provided
        if guidance_scale is not None and hasattr(self, 'unconditional_output'):
            uncond_x_start = self.unconditional_output(x, None, t, padding_masks)
            uncond_x_start = maybe_clip(uncond_x_start)
            uncond_pred_noise = self.predict_noise_from_start(x, t, uncond_x_start)
            x_start = uncond_x_start + guidance_scale * (x_start - uncond_x_start)
            pred_noise = uncond_pred_noise + guidance_scale * (pred_noise - uncond_pred_noise)
            
        return pred_noise, x_start

    def p_mean_variance(self, x, t, clip_denoised=True, guidance_scale=None):
        """Modified to pass guidance_scale to model_predictions"""
        _, x_start = self.model_predictions(x, t, clip_denoised, guidance_scale=guidance_scale)
        if clip_denoised:
            x_start.clamp_(-1., 1.)
        model_mean, posterior_variance, posterior_log_variance = \
            self.q_posterior(x_start=x_start, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance, x_start

    def p_sample(self, x, t: int, clip_denoised=True, cond_fn=None, model_kwargs=None,
                T_max=1, inv_guidance_scale=0.0, lambda_step=None):
        """Main function with z-sampling improvements"""
        b, *_, device = *x.shape, self.betas.device
        batched_times = torch.full((x.shape[0],), t, device=x.device, dtype=torch.long)
        
        # Set default lambda_step if not provided
        if lambda_step is None:
            lambda_step = 0.01 * len(self.timesteps)
        # print(self.timesteps)
        # Original forward process
        model_mean, _, model_log_variance, x_start = \
            self.p_mean_variance(x=x, t=batched_times, clip_denoised=clip_denoised)
        noise = torch.randn_like(x) if t > 0 else 0.
        
        if cond_fn is not None:
            model_mean = self.condition_mean(
                cond_fn, model_mean, model_log_variance, x, t=batched_times, 
                model_kwargs=model_kwargs
            )
        
        pred_series = model_mean + (0.5 * model_log_variance).exp() * noise
        
        # R-Sampling optimization for early timesteps
        if t < lambda_step:
            for _ in range(T_max):
                #  inverse step
           
                inv_mean, inv_var, inv_log_var, _ = self.p_mean_variance(
                    x=pred_series, 
                    t=batched_times+1 if t < len(self.timesteps)-1 else batched_times,
                    clip_denoised=clip_denoised,
                    guidance_scale=inv_guidance_scale
                )
                inv_noise = torch.randn_like(pred_series) if t > 0 else 0.
                inv_pred_series = inv_mean + (0.5 * inv_log_var).exp() * inv_noise
                
                # forward step
                model_mean, _, model_log_variance, x_start = \
                    self.p_mean_variance(x=inv_pred_series, t=batched_times, clip_denoised=clip_denoised)
                noise = torch.randn_like(inv_pred_series) if t > 0 else 0.
                pred_series = model_mean + (0.5 * model_log_variance).exp() * noise
                
                if cond_fn is not None:
                    pred_series = self.condition_mean(
                        cond_fn, pred_series, model_log_variance, 
                        inv_pred_series, t=batched_times, model_kwargs=model_kwargs
                    )
        
        return pred_series, x_start


    @torch.no_grad()
    def sample(self, shape):
        device = self.betas.device
        img = torch.randn(shape, device=device)
        for t in tqdm(reversed(range(0, self.num_timesteps)),
                      desc='sampling loop time step', total=self.num_timesteps):
            img, _ = self.p_sample(
                img, t,
                T_max=self.T_max,
                inv_guidance_scale=self.inv_guidance_scale,
                lambda_step=self.lambda_step,
            )
        return img

    @torch.no_grad()
    def calibrate_reflection(self, clean_series, momentum=0.99):
        """Calibrate reflection against roughness observed in training data."""
        derivatives = (clean_series[:, 1:] - clean_series[:, :-1]).abs().flatten(1)
        roughness = torch.quantile(derivatives, 0.95, dim=1).mean()
        if not bool(self.clean_roughness_initialized):
            self.clean_roughness_ema.copy_(roughness)
            self.clean_roughness_initialized.fill_(True)
        else:
            self.clean_roughness_ema.mul_(momentum).add_(roughness * (1.0 - momentum))

    @torch.no_grad()
    def _adaptive_reflect(self, x_start, timestep):
        """Cheap data-space correction using the learned temporal prototypes."""
        if self.prototype_mode != 'multiscale':
            zeros = torch.zeros(x_start.shape[0], device=x_start.device)
            return x_start, zeros, zeros, torch.ones_like(zeros)

        _, prototype_context, _, _ = self._prototype_condition(x_start, compute_aux=False)
        point_error = (x_start - prototype_context).square().mean(dim=2).sqrt()
        topk_ratio = min(max(self.reflection_topk_ratio, 1e-3), 1.0)
        cutoff = torch.quantile(point_error, 1.0 - topk_ratio, dim=1, keepdim=True)
        local_scale = point_error.std(dim=1, keepdim=True).clamp_min(1e-4)
        local_activation = torch.sigmoid(
            (point_error - cutoff)
            / (max(self.reflection_mask_temperature, 1e-4) * local_scale)
        )
        error = (
            (point_error * local_activation).sum(dim=1)
            / local_activation.sum(dim=1).clamp_min(1.0)
        )
        magnitude = x_start.square().mean(dim=(1, 2)).sqrt().clamp_min(1e-4)
        relative_error = error / magnitude

        if self.num_timesteps <= 1:
            progress = 1.0
        else:
            progress = 1.0 - float(timestep) / float(self.num_timesteps - 1)
        if progress < self.reflection_start_ratio:
            activation = torch.zeros_like(relative_error)
        else:
            temperature = max(self.reflection_temperature, 1e-4)
            activation = torch.sigmoid(
                (relative_error - self.reflection_threshold) / temperature
            )

        # Reflection is useful when accelerated sampling creates excessive
        # local roughness, but can over-smooth data that are already realistic.
        # The training-data calibration makes this gate dataset-adaptive.
        if bool(self.clean_roughness_initialized):
            derivatives = (x_start[:, 1:] - x_start[:, :-1]).abs().flatten(1)
            roughness = torch.quantile(derivatives, 0.95, dim=1)
            roughness_ratio = roughness / self.clean_roughness_ema.clamp_min(1e-6)
            roughness_gate = torch.sigmoid(
                (roughness_ratio - self.reflection_roughness_threshold)
                / max(self.reflection_roughness_temperature, 1e-4)
            )
            activation = activation * roughness_gate
        else:
            roughness_ratio = torch.ones_like(relative_error)

        strength = self.reflection_strength * progress * activation
        reflection_mask = local_activation
        reflected = x_start + strength[:, None, None] * reflection_mask[:, :, None] * (
            prototype_context - x_start
        )
        return reflected, relative_error, activation, roughness_ratio

    @torch.no_grad()
    def adaptive_reflection_sample(self, shape, max_nfe=None, clip_denoised=True):
        """DDIM sampling with prototype-driven reflection and adaptive skipping.

        Reflection reuses the predicted clean series and only evaluates the
        comparatively small prototype learner.  It therefore avoids the two
        extra denoiser evaluations used by inverse-forward re-sampling.
        """
        batch, device = shape[0], self.betas.device
        max_nfe = int(default(max_nfe, self.adaptive_sampling_timesteps))
        max_nfe = max(1, min(max_nfe, self.num_timesteps))
        img = torch.randn(shape, device=device)
        timestep = self.num_timesteps - 1
        nfe = 0
        residual_trace, activation_trace, roughness_trace, timestep_trace = [], [], [], []

        progress_bar = tqdm(
            total=max_nfe, desc='adaptive reflection sampling', leave=False
        )
        while timestep >= 0 and nfe < max_nfe:
            time_cond = torch.full((batch,), timestep, device=device, dtype=torch.long)
            _, x_start = self.model_predictions(
                img, time_cond, clip_x_start=clip_denoised
            )
            nfe += 1
            progress_bar.update(1)

            x_start, residual, activation, roughness_ratio = self._adaptive_reflect(
                x_start, timestep
            )
            residual_trace.append(float(residual.mean().item()))
            activation_trace.append(float(activation.mean().item()))
            roughness_trace.append(float(roughness_ratio.mean().item()))
            timestep_trace.append(int(timestep))

            if timestep == 0 or nfe >= max_nfe:
                img = x_start
                break

            remaining_nfe = max_nfe - nfe
            required_stride = int(math.ceil(timestep / max(remaining_nfe, 1)))
            nominal_stride = max(1, int(math.ceil(self.num_timesteps / max_nfe)))
            adaptive_multiplier = 1.0 + self.adaptive_stride_gain * (
                1.0 - float(activation.mean().item())
            )
            desired_stride = max(1, int(round(nominal_stride * adaptive_multiplier)))
            stride = max(required_stride, desired_stride)
            next_timestep = max(0, timestep - stride)

            # Recompute noise from the reflected clean prediction so that the
            # accelerated DDIM update remains internally consistent.
            pred_noise = self.predict_noise_from_start(img, time_cond, x_start)
            alpha = self.alphas_cumprod[timestep]
            alpha_next = self.alphas_cumprod[next_timestep]
            sigma = self.eta * (
                (1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)
            ).clamp_min(0).sqrt()
            coefficient = (1 - alpha_next - sigma ** 2).clamp_min(0).sqrt()
            noise = torch.randn_like(img) if float(sigma) > 0 else 0.0
            img = x_start * alpha_next.sqrt() + coefficient * pred_noise + sigma * noise
            timestep = next_timestep

        progress_bar.close()
        self.last_sampling_stats = {
            'nfe': nfe,
            'mean_prototype_residual': float(np.mean(residual_trace)),
            'mean_reflection_activation': float(np.mean(activation_trace)),
            'mean_roughness_ratio': float(np.mean(roughness_trace)),
            'timesteps': timestep_trace,
        }
        return img

    @torch.no_grad()
    def fast_sample(self, shape, clip_denoised=True):
        batch, device, total_timesteps, sampling_timesteps, eta = \
            shape[0], self.betas.device, self.num_timesteps, self.sampling_timesteps, self.eta

        # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)

        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))  # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]
        img = torch.randn(shape, device=device)

        for time, time_next in tqdm(time_pairs, desc='sampling loop time step'):
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            pred_noise, x_start, *_ = self.model_predictions(img, time_cond, clip_x_start=clip_denoised)

            if time_next < 0:
                img = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]
            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()
            noise = torch.randn_like(img)
            img = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise

        return img
    
    def generate_mts(self, batch_size=16, model_kwargs=None, cond_fn=None):
        feature_size, seq_length = self.feature_size, self.seq_length
        if cond_fn is not None:
            sample_fn = self.fast_sample_cond if self.fast_sampling else self.sample_cond
            return sample_fn((batch_size, seq_length, feature_size), model_kwargs=model_kwargs, cond_fn=cond_fn)
        if self.sampling_mode == 'adaptive_reflection':
            return self.adaptive_reflection_sample(
                (batch_size, seq_length, feature_size),
                max_nfe=self.adaptive_sampling_timesteps,
            )
        if self.sampling_mode != 'legacy':
            raise ValueError(f'unknown sampling mode: {self.sampling_mode}')
        sample_fn = self.fast_sample if self.fast_sampling else self.sample
        return sample_fn((batch_size, seq_length, feature_size))

    @property
    def loss_fn(self):
        if self.loss_type == 'l1':
            return F.l1_loss
        elif self.loss_type == 'l2':
            return F.mse_loss
        else:
            raise ValueError(f'invalid loss type {self.loss_type}')

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        return (
                extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def _train_loss(self, x_start, t, x_mark=None, target=None, noise=None, padding_masks=None):
        noise = default(noise, lambda: torch.randn_like(x_start))  # x_start : ori data
        if target is None:
            target = x_start
        if self.prototype_mode == 'multiscale' and self.training:
            self.kpa.initialize_from_batch(target)
            self.calibrate_reflection(target)
        x = self.q_sample(x_start=x_start, t=t, noise=noise)   # noise sample
        model_out = self.output(x, None, t, padding_masks)
        train_loss = self.loss_fn(model_out, target, reduction='none')
        fourier_loss = torch.tensor([0.])
        if self.use_ff:
            fft1 = torch.fft.fft(model_out.transpose(1, 2), norm='forward')
            fft2 = torch.fft.fft(target.transpose(1, 2), norm='forward')
            fft1, fft2 = fft1.transpose(1, 2), fft2.transpose(1, 2)
            fourier_loss = self.loss_fn(torch.real(fft1), torch.real(fft2), reduction='none')\
                           + self.loss_fn(torch.imag(fft1), torch.imag(fft2), reduction='none')
            train_loss +=  self.ff_weight * fourier_loss
        
        train_loss = reduce(train_loss, 'b ... -> b (...)', 'mean')
        train_loss = train_loss * extract(self.loss_weight, t, train_loss.shape)
        train_loss = train_loss.mean()

        if self.prototype_mode == 'multiscale' and self.prototype_loss_weight > 0:
            noisy_assignments = self._last_proto_attn
            _, _, clean_assignments, prototype_aux = self._prototype_condition(
                target, compute_aux=True
            )
            consistency_losses = []
            for noisy_assignment, clean_assignment in zip(noisy_assignments, clean_assignments):
                consistency_losses.append(F.mse_loss(noisy_assignment, clean_assignment))
            consistency = torch.stack(consistency_losses).mean()
            prototype_loss = (
                prototype_aux['reconstruction']
                + self.prototype_balance_weight * prototype_aux['balance']
                + self.prototype_diversity_weight * prototype_aux['diversity']
                + self.prototype_consistency_weight * consistency
            )
            train_loss = train_loss + self.prototype_loss_weight * prototype_loss
            self._last_loss_components = {
                'diffusion': float((train_loss - self.prototype_loss_weight * prototype_loss).detach()),
                'prototype': float(prototype_loss.detach()),
                'prototype_reconstruction': float(prototype_aux['reconstruction'].detach()),
                'prototype_balance': float(prototype_aux['balance'].detach()),
                'prototype_diversity': float(prototype_aux['diversity'].detach()),
                'prototype_consistency': float(consistency.detach()),
            }
        else:
            self._last_loss_components = {'diffusion': float(train_loss.detach())}
        return train_loss

    def forward(self, x, **kwargs):
        b, c, n, device, feature_size, = *x.shape, x.device, self.feature_size
        assert n == feature_size, f'number of variable must be {feature_size}'
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        return self._train_loss(x_start=x, t=t, **kwargs)

    def return_components(self, x, t: int):
        b, c, n, device, feature_size, = *x.shape, x.device, self.feature_size
        assert n == feature_size, f'number of variable must be {feature_size}'
        t = torch.tensor([t])
        t = t.repeat(b).to(device)
        x = self.q_sample(x, t)
        trend, season, residual = self.model(x, t, return_res=True)
        return trend, season, residual, x

    def fast_sample_infill(self, shape, target, sampling_timesteps, partial_mask=None, clip_denoised=True, model_kwargs=None):
        batch, device, total_timesteps, eta = shape[0], self.betas.device, self.num_timesteps, self.eta

        # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)

        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))  # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]
        img = torch.randn(shape, device=device)

        for time, time_next in tqdm(time_pairs, desc='conditional sampling loop time step'):
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            pred_noise, x_start, *_ = self.model_predictions(img, time_cond, clip_x_start=clip_denoised)

            if time_next < 0:
                img = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]
            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()
            pred_mean = x_start * alpha_next.sqrt() + c * pred_noise
            noise = torch.randn_like(img)

            img = pred_mean + sigma * noise
            img = self.langevin_fn(sample=img, mean=pred_mean, sigma=sigma, t=time_cond,
                                   tgt_embs=target, partial_mask=partial_mask, **model_kwargs)
            target_t = self.q_sample(target, t=time_cond)
            img[partial_mask] = target_t[partial_mask]

        img[partial_mask] = target[partial_mask]

        return img

    def sample_infill(
        self,
        shape, 
        target,
        partial_mask=None,
        clip_denoised=True,
        model_kwargs=None,
    ):
        """
        Generate samples from the model and yield intermediate samples from
        each timestep of diffusion.
        """
        batch, device = shape[0], self.betas.device
        img = torch.randn(shape, device=device)
        for t in tqdm(reversed(range(0, self.num_timesteps)),
                      desc='conditional sampling loop time step', total=self.num_timesteps):
            img = self.p_sample_infill(x=img, t=t, clip_denoised=clip_denoised, target=target,
                                       partial_mask=partial_mask, model_kwargs=model_kwargs)
        
        img[partial_mask] = target[partial_mask]
        return img
    
    def p_sample_infill(
        self,
        x,
        target,
        t: int,
        partial_mask=None,
        clip_denoised=True,
        model_kwargs=None
    ):
        b, *_, device = *x.shape, self.betas.device
        batched_times = torch.full((x.shape[0],), t, device=x.device, dtype=torch.long)
        model_mean, _, model_log_variance, _ = \
            self.p_mean_variance(x=x, t=batched_times, clip_denoised=clip_denoised)
        noise = torch.randn_like(x) if t > 0 else 0.  # no noise if t == 0
        sigma = (0.5 * model_log_variance).exp()
        pred_img = model_mean + sigma * noise

        pred_img = self.langevin_fn(sample=pred_img, mean=model_mean, sigma=sigma, t=batched_times,
                                    tgt_embs=target, partial_mask=partial_mask, **model_kwargs)
        
        target_t = self.q_sample(target, t=batched_times)
        pred_img[partial_mask] = target_t[partial_mask]

        return pred_img

    def langevin_fn(
        self,
        coef,
        partial_mask,
        tgt_embs,
        learning_rate,
        sample,
        mean,
        sigma,
        t,
        coef_=0.
    ):
    
        if t[0].item() < self.num_timesteps * 0.05:
            K = 0
        elif t[0].item() > self.num_timesteps * 0.9:
            K = 3
        elif t[0].item() > self.num_timesteps * 0.75:
            K = 2
            learning_rate = learning_rate * 0.5
        else:
            K = 1
            learning_rate = learning_rate * 0.25

        input_embs_param = torch.nn.Parameter(sample)

        with torch.enable_grad():
            for i in range(K):
                optimizer = torch.optim.Adagrad([input_embs_param], lr=learning_rate)
                optimizer.zero_grad()

                x_start = self.output(x=input_embs_param, t=t)

                if sigma.mean() == 0:
                    logp_term = coef * ((mean - input_embs_param) ** 2 / 1.).mean(dim=0).sum()
                    infill_loss = (x_start[partial_mask] - tgt_embs[partial_mask]) ** 2
                    infill_loss = infill_loss.mean(dim=0).sum()
                else:
                    logp_term = coef * ((mean - input_embs_param)**2 / sigma).mean(dim=0).sum()
                    infill_loss = (x_start[partial_mask] - tgt_embs[partial_mask]) ** 2
                    infill_loss = (infill_loss/sigma.mean()).mean(dim=0).sum()
            
                loss = logp_term + infill_loss
                loss.backward()
                optimizer.step()
                epsilon = torch.randn_like(input_embs_param.data)
                input_embs_param = torch.nn.Parameter((input_embs_param.data + coef_ * sigma.mean().item() * epsilon).detach())

        sample[~partial_mask] = input_embs_param.data[~partial_mask]
        return sample
    
    def condition_mean(self, cond_fn, mean, log_variance, x, t, model_kwargs=None):
        """
        Compute the mean for the previous step, given a function cond_fn that
        computes the gradient of a conditional log probability with respect to
        x. In particular, cond_fn computes grad(log(p(y|x))), and we want to
        condition on y.

        This uses the conditioning strategy from Sohl-Dickstein et al. (2015).
        """
        gradient = cond_fn(x=x, t=t, **model_kwargs)
        new_mean = (
            mean.float() + torch.exp(log_variance) * gradient.float()
        )
        return new_mean
    
    def condition_score(self, cond_fn, x_start, x, t, model_kwargs=None):
        """
        Compute what the p_mean_variance output would have been, should the
        model's score function be conditioned by cond_fn.

        See condition_mean() for details on cond_fn.

        Unlike condition_mean(), this instead uses the conditioning strategy
        from Song et al (2020).
        """
        alpha_bar = extract(self.alphas_cumprod, t, x.shape)

        eps = self.predict_noise_from_start(x, t, x_start)
        eps = eps - (1 - alpha_bar).sqrt() * cond_fn(x, t, **model_kwargs)

        pred_xstart = self.predict_start_from_noise(x, t, eps)
        model_mean, _, _ = self.q_posterior(x_start=pred_xstart, x_t=x, t=t)
        return model_mean, pred_xstart
    
    def sample_cond(
        self,
        shape,
        clip_denoised=True,
        model_kwargs=None,
        cond_fn=None
    ):
        """
        Generate samples from the model and yield intermediate samples from
        each timestep of diffusion.
        """
        batch, device = shape[0], self.betas.device
        img = torch.randn(shape, device=device)
        for t in tqdm(reversed(range(0, self.num_timesteps)),
                      desc='sampling loop time step', total=self.num_timesteps):
            img, x_start = self.p_sample(img, t, clip_denoised=clip_denoised, cond_fn=cond_fn,
                                         model_kwargs=model_kwargs)
        return img

    def fast_sample_cond(
        self,
        shape,
        clip_denoised=True,
        model_kwargs=None,
        cond_fn=None
    ):
        batch, device, total_timesteps, sampling_timesteps, eta = \
            shape[0], self.betas.device, self.num_timesteps, self.sampling_timesteps, self.eta

        # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)

        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))  # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]
        img = torch.randn(shape, device=device)
        x_start = None

        for time, time_next in tqdm(time_pairs, desc='sampling loop time step'):
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            pred_noise, x_start, *_ = self.model_predictions(img, time_cond, clip_x_start=clip_denoised)

            if cond_fn is not None:
                _, x_start = self.condition_score(cond_fn, x_start, img, time_cond, model_kwargs=model_kwargs)
                pred_noise = self.predict_noise_from_start(img, time_cond, x_start)

            if time_next < 0:
                img = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]
            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()
            noise = torch.randn_like(img)
            img = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise

        return img




class TimeFeature:
    def __init__(self):
        pass

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        pass

    def __repr__(self):
        return self.__class__.__name__ + "()"


class SecondOfMinute(TimeFeature):
    """Minute of hour encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.second / 59.0 - 0.5


class MinuteOfHour(TimeFeature):
    """Minute of hour encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.minute / 59.0 - 0.5


class HourOfDay(TimeFeature):
    """Hour of day encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.hour / 23.0 - 0.5


class DayOfWeek(TimeFeature):
    """Hour of day encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.dayofweek / 6.0 - 0.5


class DayOfMonth(TimeFeature):
    """Day of month encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.day - 1) / 30.0 - 0.5


class DayOfYear(TimeFeature):
    """Day of year encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.dayofyear - 1) / 365.0 - 0.5


class MonthOfYear(TimeFeature):
    """Month of year encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.month - 1) / 11.0 - 0.5


class WeekOfYear(TimeFeature):
    """Week of year encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.isocalendar().week - 1) / 52.0 - 0.5


def time_features_from_frequency_str(freq_str: str) -> List[TimeFeature]:
    """
    Returns a list of time features that will be appropriate for the given frequency string.
    Parameters
    ----------
    freq_str
        Frequency string of the form [multiple][granularity] such as "12H", "5min", "1D" etc.
    """

    features_by_offsets = {
        offsets.YearEnd: [],  # No features for YearEnd
        offsets.QuarterEnd: [MonthOfYear],  # Only MonthOfYear
        offsets.MonthEnd: [MonthOfYear],  # Only MonthOfYear
        offsets.Week: [DayOfMonth, WeekOfYear],  # DayOfMonth, WeekOfYear
        offsets.Day: [DayOfWeek, DayOfMonth, DayOfYear],  # DayOfWeek, DayOfMonth, DayOfYear
        offsets.BusinessDay: [DayOfWeek, DayOfMonth, DayOfYear],  # DayOfWeek, DayOfMonth, DayOfYear
        offsets.Hour: [HourOfDay, DayOfWeek, DayOfMonth, DayOfYear],  # HourOfDay, DayOfWeek, DayOfMonth, DayOfYear
        offsets.Minute: [MinuteOfHour, HourOfDay, DayOfWeek, DayOfMonth],  # MinuteOfHour, HourOfDay, DayOfWeek, DayOfMonth
        offsets.Second: [SecondOfMinute, MinuteOfHour, HourOfDay, DayOfWeek],  # SecondOfMinute, MinuteOfHour, HourOfDay, DayOfWeek
    }

    offset = to_offset(freq_str)

    for offset_type, feature_classes in features_by_offsets.items():
        if isinstance(offset, offset_type):
            return [cls() for cls in feature_classes]

    supported_freq_msg = f"""
    Unsupported frequency {freq_str}
    The following frequencies are supported:
        Y   - yearly
            alias: A
        M   - monthly
        W   - weekly
        D   - daily
        B   - business days
        H   - hourly
        T   - minutely
            alias: min
        S   - secondly
    """
    raise RuntimeError(supported_freq_msg)


def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)
