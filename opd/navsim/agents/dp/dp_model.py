# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import DDPMScheduler

from navsim.agents.dp.dp_config import DPConfig
from navsim.agents.gtrs_dense.hydra_backbone_bev import HydraBackboneBEV

x_diff_min = -1.2698211669921875
x_diff_max = 7.475563049316406
x_diff_mean = 2.950225591659546

# Y difference statistics
y_diff_min = -5.012081146240234
y_diff_max = 4.8563690185546875
y_diff_mean = 0.0607292577624321

# Calculate scaling factors for differences
x_diff_scale = max(abs(x_diff_max - x_diff_mean), abs(x_diff_min - x_diff_mean))
y_diff_scale = max(abs(y_diff_max - y_diff_mean), abs(y_diff_min - y_diff_mean))

HORIZON = 8
ACTION_DIM = 4
ACTION_DIM_ORI = 3


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class SimpleDiffusionTransformer(nn.Module):
    def __init__(self, d_model, nhead, d_ffn, dp_nlayers, input_dim, obs_len, self_cond=False):
        super().__init__()
        self.dp_transformer = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model, nhead, d_ffn,
                dropout=0.0, batch_first=True
            ), dp_nlayers
        )
        # self_cond: additionally feed the model's previous x1 estimate (flattened, same
        # size as the noisy sample). Zeros at the first step / when disabled at runtime.
        self.self_cond_dim = input_dim if self_cond else 0
        self.input_emb = nn.Linear(input_dim + self.self_cond_dim, d_model)
        self.time_emb = SinusoidalPosEmb(d_model)
        self.ln_f = nn.LayerNorm(d_model)
        self.output_emb = nn.Linear(d_model, input_dim)
        token_len = obs_len + 1
        self.cond_pos_emb = nn.Parameter(torch.zeros(1, token_len, d_model))
        self.pos_emb = nn.Parameter(torch.zeros(1, 1, d_model))
        self.apply(self._init_weights)
        self._register_load_state_dict_pre_hook(self._expand_input_emb_hook)

    def _expand_input_emb_hook(self, state_dict, prefix, local_metadata, strict,
                               missing_keys, unexpected_keys, error_msgs):
        """Make `self_cond=True` loadable from a checkpoint trained with it off.

        Enabling self-conditioning widens `input_emb` from (d_model, input_dim) to
        (d_model, 2*input_dim). Zero-padding the new columns makes the self_cond branch
        contribute exactly nothing at load time, so the expanded model is numerically
        identical to the checkpoint and then *learns* to use the extra input. Registered as a
        load pre-hook rather than handled in the agent so that it also covers Lightning's
        `resume_ckpt_path` path, which loads strictly and never calls Agent.initialize().
        """
        key = prefix + 'input_emb.weight'
        if key not in state_dict:
            return
        have = state_dict[key]
        want = self.input_emb.weight
        if tuple(have.shape) == tuple(want.shape):
            return
        if have.dim() != 2 or have.shape[0] != want.shape[0] or have.shape[1] >= want.shape[1]:
            return  # not a column-subset: leave it to the normal shape-mismatch error
        padded = have.new_zeros(want.shape)
        padded[:, :have.shape[1]] = have
        state_dict[key] = padded

    def _init_weights(self, module):
        ignore_types = (nn.Dropout,
                        SinusoidalPosEmb,
                        nn.TransformerEncoderLayer,
                        nn.TransformerDecoderLayer,
                        nn.TransformerEncoder,
                        nn.TransformerDecoder,
                        nn.ModuleList,
                        nn.Mish,
                        nn.Sequential)
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.MultiheadAttention):
            weight_names = [
                'in_proj_weight', 'q_proj_weight', 'k_proj_weight', 'v_proj_weight']
            for name in weight_names:
                weight = getattr(module, name)
                if weight is not None:
                    torch.nn.init.normal_(weight, mean=0.0, std=0.02)

            bias_names = ['in_proj_bias', 'bias_k', 'bias_v']
            for name in bias_names:
                bias = getattr(module, name)
                if bias is not None:
                    torch.nn.init.zeros_(bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
        elif isinstance(module, SimpleDiffusionTransformer):
            torch.nn.init.normal_(module.pos_emb, mean=0.0, std=0.02)
        elif isinstance(module, ignore_types):
            # no param
            pass
        else:
            raise RuntimeError("Unaccounted module {}".format(module))

    def forward(self,
                sample,
                timestep,
                cond,
                self_cond=None):
        B, HORIZON, DIM = sample.shape
        sample = sample.view(B, -1).float()
        if self.self_cond_dim > 0:
            if self_cond is None:
                self_cond = torch.zeros_like(sample)
            sample = torch.cat([sample, self_cond.float()], dim=-1)
        input_emb = self.input_emb(sample)

        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])
        time_emb = self.time_emb(timesteps).unsqueeze(1)
        # (B,To,n_emb)
        cond_embeddings = torch.cat([time_emb, cond], dim=1)
        tc = cond_embeddings.shape[1]
        position_embeddings = self.cond_pos_emb[
                              :, :tc, :
                              ]  # each position maps to a (learnable) vector
        x = cond_embeddings + position_embeddings
        memory = x
        # (B,T_cond,n_emb)

        # decoder
        token_embeddings = input_emb.unsqueeze(1)
        t = token_embeddings.shape[1]
        position_embeddings = self.pos_emb[
                              :, :t, :
                              ]  # each position maps to a (learnable) vector
        x = token_embeddings + position_embeddings
        # (B,T,n_emb)
        x = self.dp_transformer(
            tgt=x,
            memory=memory,
        )
        # (B,T,n_emb)
        x = self.ln_f(x)
        x = self.output_emb(x)
        return x.squeeze(1).view(B, HORIZON, DIM)


def diff_traj(traj):
    B, L, _ = traj.shape
    sin = traj[..., -1:].sin()
    cos = traj[..., -1:].cos()
    zero_pad = torch.zeros((B, 1, 1), dtype=traj.dtype, device=traj.device)
    x_diff = traj[..., 0:1].diff(n=1, dim=1, prepend=zero_pad)
    x_diff = x_diff - x_diff_mean
    x_diff_range = max(abs(x_diff_max - x_diff_mean), abs(x_diff_min - x_diff_mean))
    x_diff_norm = x_diff / x_diff_range

    zero_pad = torch.zeros((B, 1, 1), dtype=traj.dtype, device=traj.device)
    y_diff = traj[..., 1:2].diff(n=1, dim=1, prepend=zero_pad)
    y_diff = y_diff - y_diff_mean
    y_diff_range = max(abs(y_diff_max - y_diff_mean), abs(y_diff_min - y_diff_mean))
    y_diff_norm = y_diff / y_diff_range

    return torch.cat([x_diff_norm, y_diff_norm, sin, cos], -1)


def cumsum_traj(norm_trajs):
    B, L, _ = norm_trajs.shape
    sin_values = norm_trajs[..., 2:3]
    cos_values = norm_trajs[..., 3:4]
    heading = torch.atan2(sin_values, cos_values)

    # Denormalize x differences
    x_diff_range = max(abs(x_diff_max - x_diff_mean), abs(x_diff_min - x_diff_mean))
    x_diff = norm_trajs[..., 0:1] * x_diff_range + x_diff_mean

    # Denormalize y differences
    y_diff_range = max(abs(y_diff_max - y_diff_mean), abs(y_diff_min - y_diff_mean))
    y_diff = norm_trajs[..., 1:2] * y_diff_range + y_diff_mean

    # Cumulative sum to get absolute positions
    x = x_diff.cumsum(dim=1)
    y = y_diff.cumsum(dim=1)

    return torch.cat([x, y, heading], -1)


class DPHead(nn.Module):
    """Trajectory decoder. Architecture is the official SimpleDiffusionTransformer.

    Official weights predict DDPM epsilon; an epsilon-prediction is NOT a Flow-Matching
    velocity field (they differ by a t-dependent linear transform). Training with
    use_flow_matching=True therefore requires reinit_traj_head=True so the decoder is
    trained from scratch on the FM velocity target — never "reinterpret" official weights.
    """

    def __init__(self, num_poses: int, d_ffn: int, d_model: int, vocab_path: str,
                 nhead: int, nlayers: int, config: DPConfig = None
                 ):
        super().__init__()
        self.config = config
        self.use_flow_matching = bool(getattr(config, 'use_flow_matching', False))
        self.fm_num_inference_steps = int(getattr(config, 'fm_num_inference_steps', 20))
        self.fm_sigma_min = float(getattr(config, 'fm_sigma_min', 1e-4))
        # 第3章改造：辅助监督 + 训推对齐
        self.fm_traj_aux_weight = float(getattr(config, 'fm_traj_aux_weight', 0.0))
        self.fm_traj_aux_pos_weight = float(getattr(config, 'fm_traj_aux_pos_weight', 1.0))
        self.fm_traj_aux_head_weight = float(getattr(config, 'fm_traj_aux_head_weight', 0.5))
        self.fm_time_align = bool(getattr(config, 'fm_time_align', True))
        self.fm_lattice_t_prob = float(getattr(config, 'fm_lattice_t_prob', 0.0))
        # 训练时在这一组 K 的 Euler 网格并集上采 t；评测扫到的每个 K 都应在集合内，
        # 否则 {k/K} 与训练见过的格点只在 t=1 相交，步数扫描曲线会被 OOD artifact 污染。
        self.fm_lattice_k_set = tuple(int(k) for k in getattr(
            config, 'fm_lattice_k_set', (2, 3, 4, 5, 8, 10, 20)) if int(k) > 0)
        if not self.fm_lattice_k_set:
            self.fm_lattice_k_set = (self.fm_num_inference_steps,)
        self.fm_self_conditioning = bool(getattr(config, 'fm_self_conditioning', False))
        self.fm_self_cond_p = float(getattr(config, 'fm_self_cond_p', 0.5))
        # x̂1 裁剪：DDPM 的 clip_sample 作用在 pred_original_sample 上，FM 下的正确对应
        # 是裁剪隐含的 x̂1（归一化增量的合法范围恰为 [-1,1]）再反解速度，而不是裁剪 x_t。
        self.fm_clip_sample = bool(getattr(config, 'fm_clip_sample', True))
        self.fm_clip_range = float(getattr(config, 'fm_clip_range', 1.0))
        # 辅助/运动学损失把残差除以 t_eff 做归一，这里给分母一个下界防止 t→0 时放大噪声
        self.fm_aux_t_floor = float(getattr(config, 'fm_aux_t_floor', 0.05))
        # 自洽监督：以该概率用"模型自己积分出来的中间态"替代真值插值点
        self.fm_self_consistency_p = float(getattr(config, 'fm_self_consistency_p', 0.0))
        self.fm_sc_max_steps = int(getattr(config, 'fm_sc_max_steps', 4))
        self.fm_sc_warmup_steps = int(getattr(config, 'fm_sc_warmup_steps', 2000))
        self.fm_sc_target_clip = float(getattr(config, 'fm_sc_target_clip', 6.0))
        # 运动学（增量的一/二阶差分）损失权重
        self.fm_kin_weight = float(getattr(config, 'fm_kin_weight', 0.0))
        # 纯 Python 计数器（不是 buffer，不进 state_dict，resume 后重新 warmup）
        self._sc_calls = 0

        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=config.denoising_timesteps,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule='squaredcos_cap_v2',
            variance_type='fixed_small',
            clip_sample=True,
            clip_sample_range=1.0,
            prediction_type='epsilon'
        )
        img_num = 2 if config.use_back_view else 1

        self.transformer_dp = SimpleDiffusionTransformer(
            d_model, nhead, d_ffn, config.dp_layers,
            input_dim=ACTION_DIM * HORIZON,
            obs_len=config.img_vert_anchors * config.img_horz_anchors * img_num + 1,
            self_cond=self.fm_self_conditioning,
        )
        # DDPM inference steps: decoupled from num_train_timesteps so the DDPM baseline can
        # be compared with FM at the same sampling budget. Default 100 = official behavior.
        self.num_inference_steps = int(getattr(config, 'ddpm_num_inference_steps',
                                               self.noise_scheduler.config.num_train_timesteps))

    def _fm_time_for_transformer(self, t01: torch.Tensor) -> torch.Tensor:
        """Map Flow-Matching t in [0, 1] onto the official DDPM timestep scale.

        Official time embeddings were trained with integer t in [0, denoising_timesteps).
        With fm_time_align=True we map to [0, T-1]: the K-step Euler sampler queries
        t01=1.0 at its first step, which would otherwise land on the never-seen index T.
        """
        if self.fm_time_align:
            return t01 * float(self.config.denoising_timesteps - 1)
        return t01 * float(self.config.denoising_timesteps)

    def _fm_interpolate(self, x1: torch.Tensor, noise: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t = t.view(-1, 1, 1)
        t = t * (1.0 - self.fm_sigma_min) + self.fm_sigma_min
        return (1.0 - t) * x1 + t * noise

    def _fm_t_eff(self, t01: torch.Tensor) -> torch.Tensor:
        """Effective interpolation ratio t' = t(1-sigma_min)+sigma_min, matching _fm_interpolate."""
        return t01 * (1.0 - self.fm_sigma_min) + self.fm_sigma_min

    def _fm_sample_t01(self, B: int, device) -> torch.Tensor:
        """Sample training times in (0, 1].

        With probability fm_lattice_t_prob sample on a K-step Euler grid {k/K, k=1..K},
        drawing K per-sample from fm_lattice_k_set. Using a *set* of K (rather than a single
        hard-coded one) is what makes the inference-step sweep valid: with one K only, any
        evaluated K that does not divide it queries times the model never saw.
        """
        t_uniform = torch.rand(B, device=device).clamp_min(1e-3)
        if self.fm_lattice_t_prob <= 0.0:
            return t_uniform
        ks = torch.tensor(self.fm_lattice_k_set, device=device, dtype=torch.float)
        K = ks[torch.randint(0, ks.numel(), (B,), device=device)]
        # floor(U[0,1)*K)+1 is uniform on {1, ..., K}
        t_lattice = ((torch.rand(B, device=device) * K).floor() + 1.0) / K
        use_lattice = torch.rand(B, device=device) < self.fm_lattice_t_prob
        return torch.where(use_lattice, t_lattice, t_uniform)

    def _fm_clip_velocity(self, x_t: torch.Tensor, t01: torch.Tensor,
                          v_pred: torch.Tensor) -> torch.Tensor:
        """DDPM `clip_sample` semantics, expressed for Flow Matching.

        diffusers clips the *predicted clean sample*, not the running state. Here we clamp
        the implied x̂1 to the normalized data range and re-derive the velocity from it, so
        few-step sampling cannot extrapolate far outside the increment distribution (which
        cumsum_traj would then blow up by ~4.5x per step).
        """
        if not self.fm_clip_sample:
            return v_pred
        t_eff = self._fm_t_eff(t01).view(-1, 1, 1).clamp_min(1e-3)
        x1_est = x_t + t_eff * v_pred / (1.0 - self.fm_sigma_min)
        x1_est = x1_est.clamp(-self.fm_clip_range, self.fm_clip_range)
        return (x1_est - x_t) * (1.0 - self.fm_sigma_min) / t_eff

    def _fm_sc_alpha(self) -> float:
        """Self-consistency probability with a linear warmup.

        Early in training the model's own rollout is far off-manifold, so the dynamic target
        (x1 - x̂_t)/t is both large and uninformative. Ramping alpha keeps the first steps on
        the standard CFM target. Counter is a plain attribute, so a resume restarts the ramp.
        """
        if self.fm_self_consistency_p <= 0.0:
            return 0.0
        if self.fm_sc_warmup_steps <= 0:
            return self.fm_self_consistency_p
        ratio = min(1.0, self._sc_calls / float(self.fm_sc_warmup_steps))
        return self.fm_self_consistency_p * ratio

    def _fm_rollout_to_t(self, noise: torch.Tensor, t01: torch.Tensor, cond: torch.Tensor,
                         n_roll: int) -> torch.Tensor:
        """Euler-integrate the *current* model from pure noise (t=1) down to per-sample t01.

        This reproduces the state distribution the sampler actually visits, which is the
        whole point of self-consistency: standard CFM supervises at the ground-truth
        interpolation point x_t, but at inference the state is produced by the model's own
        integration and drifts away from it. Runs under no_grad (the Word's sg[.]), so only
        the single graded forward below contributes gradients — DDP-safe.
        """
        x = noise
        if n_roll <= 0:
            return x
        step = ((1.0 - t01) / float(n_roll)).view(-1, 1, 1)
        cur = torch.ones_like(t01)
        self_cond = None
        with torch.no_grad():
            for _ in range(n_roll):
                v = self.transformer_dp(x, self._fm_time_for_transformer(cur), cond,
                                        self_cond=self_cond)
                v = self._fm_clip_velocity(x, cur, v)
                if self.fm_self_conditioning:
                    self_cond = self._fm_x1_pred(x, cur, v).view(x.shape[0], -1)
                x = x + v * step
                cur = cur - step.view(-1)
        return x.detach()

    def _fm_x1_pred(self, x_t: torch.Tensor, t01: torch.Tensor, v_pred: torch.Tensor) -> torch.Tensor:
        """One-shot estimate of the clean sample. With x_t = x1 - t'*(x1-noise) and the
        sigma_min-consistent target v = (1-sigma_min)*(x1-noise), x1 = x_t + t'/(1-sigma_min)*v."""
        t_eff = self._fm_t_eff(t01).view(-1, 1, 1)
        return x_t + t_eff * v_pred / (1.0 - self.fm_sigma_min)

    def forward(self, kv) -> Dict[str, torch.Tensor]:
        B = kv.shape[0]
        result = {}
        if not self.training:
            NUM_PROPOSALS = self.config.num_proposals

            condition = kv.repeat_interleave(NUM_PROPOSALS, dim=0)

            noise = torch.randn(
                size=(B * NUM_PROPOSALS, HORIZON, ACTION_DIM),
                dtype=condition.dtype,
                device=condition.device,
            )

            if self.use_flow_matching:
                num_steps = self.fm_num_inference_steps
                dt = 1.0 / num_steps
                x_t = noise
                n = B * NUM_PROPOSALS
                self_cond = None  # self-conditioning: previous step's x1 estimate (zeros first)
                for i in range(num_steps):
                    t_val = 1.0 - i * dt
                    t01 = torch.full((n,), t_val, dtype=condition.dtype, device=condition.device)
                    v_pred = self.transformer_dp(x_t, self._fm_time_for_transformer(t01), condition,
                                                 self_cond=self_cond)
                    v_pred = self._fm_clip_velocity(x_t, t01, v_pred)
                    if self.fm_self_conditioning:
                        self_cond = self._fm_x1_pred(x_t, t01, v_pred).view(n, -1).detach()
                    x_t = x_t + v_pred * dt
                traj = cumsum_traj(x_t)
            else:
                self.noise_scheduler.set_timesteps(self.num_inference_steps)
                for t in self.noise_scheduler.timesteps:
                    model_output = self.transformer_dp(noise, t, condition)
                    noise = self.noise_scheduler.step(model_output, t, noise).prev_sample
                traj = cumsum_traj(noise)
            result['dp_pred'] = traj.view(B, NUM_PROPOSALS, HORIZON, ACTION_DIM_ORI)

        return result

    def get_dp_loss(self, kv, gt_trajectory):
        """Returns (loss, extras). extras: detached scalars for logging only."""
        B = kv.shape[0]
        device = kv.device
        gt_trajectory = gt_trajectory.float()
        x1 = diff_traj(gt_trajectory)

        noise = torch.randn(x1.shape, device=device, dtype=torch.float)

        if self.use_flow_matching:
            self._sc_calls += 1
            t01 = self._fm_sample_t01(B, device)
            t_eff = self._fm_t_eff(t01).view(-1, 1, 1)

            # 自洽监督：以 alpha 的概率用模型自己积分出的中间态 x̂_t 替代真值插值点，
            # 并把监督换成"从 x̂_t 出发、在剩余时间内到达 x1 所需的速度"。
            # 恒定目标 (x1-noise) 指向的是真值插值线，无法纠正已经产生的偏差。
            sc_alpha = self._fm_sc_alpha()
            use_sc = sc_alpha > 0.0 and bool(torch.rand((), device=device) < sc_alpha)
            if use_sc:
                n_roll = int(torch.randint(1, max(self.fm_sc_max_steps, 1) + 1, ()).item())
                x_t = self._fm_rollout_to_t(noise, t01, kv, n_roll)
                # 动态剩余速度目标。当 x̂_t 恰好等于真值插值点时，
                # (1-s)(x1-x_t)/t_eff == (1-s)(x1-noise)，精确退化为标准 CFM 目标。
                # 这里的分母只给一个极小下界（t01 已 clamp 到 ≥1e-3）：用 fm_aux_t_floor
                # 那样的大下界会把小 t 的目标整体缩小几十倍，等于在教模型欠冲——
                # 真正该挡住的是 x̂_t 偏差过大导致的目标爆炸，由下面的幅度 clip 负责。
                v_target = ((1.0 - self.fm_sigma_min) * (x1 - x_t) / t_eff.clamp_min(1e-3))
                v_target = v_target.clamp(-self.fm_sc_target_clip, self.fm_sc_target_clip)
            else:
                x_t = self._fm_interpolate(x1, noise, t01)
                # sigma_min-consistent rectified-flow target: for x_t=(1-t')x1 + t'noise with
                # t'=t(1-s_min)+s_min the reverse-time velocity is -(dx_t/dt) = (1-s_min)(x1-noise).
                v_target = (1.0 - self.fm_sigma_min) * (x1 - noise)
            t_in = self._fm_time_for_transformer(t01)

            self_cond = None
            if self.fm_self_conditioning and bool(torch.rand((), device=device) < self.fm_self_cond_p):
                # teacher-forced self-conditioning: first pass (no grad, zeros) -> x1 estimate
                with torch.no_grad():
                    v0 = self.transformer_dp(x_t, t_in, kv, self_cond=None)
                    self_cond = self._fm_x1_pred(x_t, t01, v0).view(B, -1)

            v_pred = self.transformer_dp(x_t, t_in, kv, self_cond=self_cond)
            fm_loss = F.mse_loss(v_pred, v_target)
            extras = {'fm_sc_alpha': torch.tensor(sc_alpha, device=device)}

            if self.fm_traj_aux_weight > 0.0 or self.fm_kin_weight > 0.0:
                # 三个损失覆盖误差的三个频段，互不重复：
                #   FM 主损失 = 逐点速度误差（中频）
                #   drift     = 增量累积误差（低频，对应 ADE/FDE 的漂移）
                #   d1/d2     = 增量的一/二阶差分误差（高频，对应加速度/jerk → comfort）
                # 三者都在*归一化增量*空间且都除以 t_eff，所以与主损失同量纲、权重可解释。
                # 不除 t_eff 的话残差恒 ∝ t_eff，t→0（推理最后几步、决定落点）时梯度消失。
                x1_pred = self._fm_x1_pred(x_t, t01, v_pred)
                t_n = t_eff.clamp_min(self.fm_aux_t_floor)
                resid = (x1_pred[..., :2] - x1[..., :2]) / t_n

                drift = resid.cumsum(dim=1)
                pos_loss = F.smooth_l1_loss(drift, torch.zeros_like(drift))
                # 航向：先单位化再比内积，等价于 1-cos(Δθ)。原来的 MSE(x1_pred[2:], x1[2:])
                # 在代数上只是主损失在 sin/cos 两通道上的 t² 重加权（因为两者共用 x_t），
                # 不提供新监督，而且可以靠缩小幅度而非对准角度来降低。
                n_pred = x1_pred[..., 2:4]
                n_pred = n_pred / n_pred.norm(dim=-1, keepdim=True).clamp_min(1e-4)
                head_loss = (1.0 - (n_pred * x1[..., 2:4]).sum(-1)).mean()
                fm_loss = fm_loss + self.fm_traj_aux_weight * (
                        self.fm_traj_aux_pos_weight * pos_loss
                        + self.fm_traj_aux_head_weight * head_loss)

                kin_loss = None
                if self.fm_kin_weight > 0.0 and resid.shape[1] >= 3:
                    d1 = resid.diff(dim=1)          # 加速度误差（增量的一阶差分）
                    d2 = d1.diff(dim=1)             # jerk 误差
                    kin_loss = (F.smooth_l1_loss(d1, torch.zeros_like(d1))
                                + F.smooth_l1_loss(d2, torch.zeros_like(d2)))
                    fm_loss = fm_loss + self.fm_kin_weight * kin_loss

                with torch.no_grad():
                    # 诊断量仍用真实米制，便于和 PDM 指标对照
                    traj_pred = cumsum_traj(x1_pred.detach())
                    traj_gt = cumsum_traj(x1)
                    pos_err = (traj_pred[..., :2] - traj_gt[..., :2]).norm(dim=-1)
                    extras.update({
                        'fm_aux_pos_loss': pos_loss.detach(),
                        'fm_aux_head_loss': head_loss.detach(),
                        'fm_x1_ade': pos_err.mean(),
                        'fm_x1_fde': pos_err[:, -1].mean(),
                    })
                    if kin_loss is not None:
                        extras['fm_kin_loss'] = kin_loss.detach()
            return fm_loss, extras

        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (B,), device=device
        ).long()
        noisy_dp_input = self.noise_scheduler.add_noise(x1, noise, timesteps)
        pred = self.transformer_dp(noisy_dp_input, timesteps, kv)
        return F.mse_loss(pred, noise), {}


class DPModel(nn.Module):
    def __init__(self, config: DPConfig):
        super().__init__()
        self._config = config
        self._backbone = HydraBackboneBEV(config)

        kv_len = self._backbone.bev_w * self._backbone.bev_h
        emb_len = kv_len + 1
        if self._config.use_hist_ego_status:
            emb_len += 1
        self._keyval_embedding = nn.Embedding(
            emb_len, config.tf_d_model
        )  # 8x8 feature grid + trajectory

        # usually, the BEV features are variable in size.
        self.downscale_layer = nn.Linear(self._backbone.img_feat_c, config.tf_d_model)
        self._bev_semantic_head = nn.Sequential(
            nn.Conv2d(
                config.bev_features_channels,
                config.bev_features_channels,
                kernel_size=(3, 3),
                stride=1,
                padding=(1, 1),
                bias=True,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                config.bev_features_channels,
                config.num_bev_classes,
                kernel_size=(1, 1),
                stride=1,
                padding=0,
                bias=True,
            ),
            nn.Upsample(
                size=(
                    config.lidar_resolution_height // 2,
                    config.lidar_resolution_width,
                ),
                mode="bilinear",
                align_corners=False,
            ),
        )

        self._status_encoding = nn.Linear((4 + 2 + 2) * config.num_ego_status, config.tf_d_model)
        if self._config.use_hist_ego_status:
            self._hist_status_encoding = nn.Linear((2 + 2 + 3) * 3, config.tf_d_model)

        self._trajectory_head = DPHead(
            num_poses=config.trajectory_sampling.num_poses,
            d_ffn=config.tf_d_ffn,
            d_model=config.tf_d_model,
            nhead=config.vadv2_head_nhead,
            nlayers=config.vadv2_head_nlayers,
            vocab_path=config.vocab_path,
            config=config
        )
        if self._config.use_temporal_bev_kv:
            self.temporal_bev_fusion = nn.Conv2d(
                config.tf_d_model * 2,
                config.tf_d_model,
                kernel_size=(1, 1),
                stride=1,
                padding=0,
                bias=True,
            )

        # guard: the decoder's cond positional table must match (time token + kv tokens).
        # obs_len was derived from img anchors which only coincidentally equals bev_h*bev_w;
        # changing backbone/resolution silently shifts the table (no runtime error otherwise).
        cond_table_len = self._trajectory_head.transformer_dp.cond_pos_emb.shape[1]
        assert cond_table_len == emb_len + 1, (
            f"cond_pos_emb len {cond_table_len} != time(1)+kv({emb_len}); "
            f"check img_vert/horz_anchors ({config.img_vert_anchors}x{config.img_horz_anchors}) "
            f"vs bev grid ({self._backbone.bev_h}x{self._backbone.bev_w})"
        )

    def forward(self, features: Dict[str, torch.Tensor],
                interpolated_traj=None) -> Dict[str, torch.Tensor]:
        camera_feature: torch.Tensor = features["camera_feature"]
        camera_feature_back: torch.Tensor = features["camera_feature_back"]
        status_feature: torch.Tensor = features["status_feature"][0]

        batch_size = status_feature.shape[0]
        assert (camera_feature[-1].shape[0] == batch_size)

        camera_feature_curr = camera_feature[-1]
        if isinstance(camera_feature_back, list):
            camera_feature_back_curr = camera_feature_back[-1]
        else:
            camera_feature_back_curr = camera_feature_back
        img_tokens, bev_tokens, up_bev = self._backbone(camera_feature_curr, camera_feature_back_curr)
        keyval = self.downscale_layer(bev_tokens)
        assert not self._config.use_temporal_bev_kv
        if self._config.use_temporal_bev_kv:
            with torch.no_grad():
                camera_feature_prev = camera_feature[-2]
                camera_feature_back_prev = camera_feature_back[-2]
                img_tokens, bev_tokens, up_bev = self._backbone(camera_feature_prev, camera_feature_back_prev)
                keyval_prev = self.downscale_layer(bev_tokens)
            # grad for fusion layer
            C = keyval.shape[-1]
            keyval = self.temporal_bev_fusion(
                torch.cat([
                    keyval.permute(0, 2, 1).view(batch_size, C, self._backbone.bev_h, self._backbone.bev_w),
                    keyval_prev.permute(0, 2, 1).view(batch_size, C, self._backbone.bev_h, self._backbone.bev_w)
                ], 1)
            ).view(batch_size, C, -1).permute(0, 2, 1).contiguous()

        bev_semantic_map = self._bev_semantic_head(up_bev)
        if self._config.num_ego_status == 1 and status_feature.shape[1] == 32:
            status_encoding = self._status_encoding(status_feature[:, :8])
        else:
            status_encoding = self._status_encoding(status_feature)

        keyval = torch.concatenate([keyval, status_encoding[:, None]], dim=1)
        if self._config.use_hist_ego_status:
            hist_status_encoding = self._hist_status_encoding(features['hist_status_feature'])
            keyval = torch.concatenate([keyval, hist_status_encoding[:, None]], dim=1)

        keyval += self._keyval_embedding.weight[None, ...]

        output: Dict[str, torch.Tensor] = {}
        trajectory = self._trajectory_head(keyval)

        output.update(trajectory)

        output['env_kv'] = keyval
        output['bev_semantic_map'] = bev_semantic_map

        return output
