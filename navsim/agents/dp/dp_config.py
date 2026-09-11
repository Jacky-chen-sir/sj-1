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

import os
from dataclasses import dataclass
from typing import Tuple

from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.common.maps.abstract_map import SemanticMapLayer
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from navsim.agents.transfuser.transfuser_config import TransfuserConfig

NAVSIM_DEVKIT_ROOT = os.environ.get("NAVSIM_DEVKIT_ROOT")


@dataclass
class DPConfig(TransfuserConfig):
    scheduler: str = 'default'

    num_proposals: int = 100

    dp_layers: int = 5
    dp_loss_weight: float = 10.0
    bev_loss_weight: float = 10.0

    # whether to optimize hydra
    use_hist_ego_status: bool = False

    norm_accel: bool = False
    denoising_timesteps: int = 100
    # DDPM baseline 的推理步数（默认 100 = 官方）；和 FM 对比时建议设成相同步数预算
    ddpm_num_inference_steps: int = 100
    use_temporal_bev_kv: bool = False

    # Flow Matching / Rectified Flow (trajectory decoder only; architecture unchanged).
    # Default False keeps official DDPM behavior so existing eval scripts still work.
    use_flow_matching: bool = False
    fm_num_inference_steps: int = 20
    fm_sigma_min: float = 1e-4
    freeze_except_traj_head: bool = False
    # When True: load official ckpt for backbone/BEV/etc, but randomly re-init
    # `_trajectory_head` so FM is trained from scratch (do not reuse DDPM epsilon weights).
    reinit_traj_head: bool = False

    # —— FM 辅助监督（归一化增量空间，残差除以 t_eff 后与主损失同量纲），0 = 关闭 ——
    # 低频：增量累积误差（漂移 → ADE/FDE）
    fm_traj_aux_weight: float = 0.5
    fm_traj_aux_pos_weight: float = 1.0
    # 航向：单位化后的 1-cos(Δθ)，有界且已 wrap
    fm_traj_aux_head_weight: float = 1.0
    # 高频：增量的一阶（加速度）与二阶（jerk）差分误差，直接对应 comfort 子指标
    fm_kin_weight: float = 0.5
    # 辅助/运动学损失里 1/t_eff 的分母下界，防 t→0 放大噪声
    fm_aux_t_floor: float = 0.05

    # —— FM 训推对齐 ——
    # t∈[0,1] 映射到 [0, T-1]（推理首步 t=100 > 训练见过的 0..99，属 OOD）
    fm_time_align: bool = True
    # 以该概率在推理 Euler 网格 {k/K} 上采样 t；0 = 纯均匀采样
    fm_lattice_t_prob: float = 0.5
    # 训练时 K 从这个集合里逐样本抽取。**评测要扫的每个 K 都必须在这里**，
    # 否则该 K 的 {k/K} 与训练格点只在 t=1 相交，步数扫描曲线会被 OOD artifact 污染。
    fm_lattice_k_set: Tuple[int, ...] = (2, 3, 4, 5, 8, 10, 20)
    # x̂1 裁剪（= DDPM clip_sample 在 FM 下的正确对应：裁剪隐含的干净样本再反解速度）。
    # 归一化增量的合法范围恰为 [-1,1]；少步采样外推后 cumsum 会按 ~4.5x/步 放大。
    fm_clip_sample: bool = True
    fm_clip_range: float = 1.0

    # —— 自洽监督：用模型自己积分出的中间态替代真值插值点 ——
    # 标准 CFM 在真值插值点 x_t 上监督恒定目标 (x1-noise)，但推理时状态由模型自回归积分
    # 产生并偏离该点，恒定目标无法纠偏，误差沿积分步累积。这里以 p 的概率改为：先 no_grad
    # 积分到 x̂_t，再监督动态剩余速度 (1-s)(x1-x̂_t)/t_eff（x̂_t 准确时精确退化为标准目标）。
    # 建议在已收敛的 ckpt 上续训开启；从零训练时靠 warmup 兜底。
    fm_self_consistency_p: float = 0.5
    fm_sc_max_steps: int = 4
    fm_sc_warmup_steps: int = 2000
    fm_sc_target_clip: float = 6.0
    # 自条件（self-conditioning）：解码器额外输入上一轮的 x1 估计，缓解少步采样外推误差。
    # 改变 input_emb 的 in_features；dp_agent 加载时会把旧权重零填充扩列，故可直接续训。
    fm_self_conditioning: bool = True
    fm_self_cond_p: float = 0.5

    seq_len: int = 2
    trajectory_imi_weight: float = 1.0
    trajectory_pdm_weight = {
        'noc': 3.0,
        'da': 3.0,
        'dd': 3.0,
        'ttc': 2.0,
        'progress': 1.0,
        'comfort': 1.0,
    }
    progress_weight: float = 2.0
    ttc_weight: float = 2.0

    inference_imi_weight: float = 0.1
    inference_da_weight: float = 1.0
    decouple: bool = False
    vocab_size: int = 4096
    vocab_path: str = None
    normalize_vocab_pos: bool = False
    num_ego_status: int = 1
    fusion_layers: int = 3

    ckpt_path: str = None
    sigma: float = 0.5
    use_pers_bev_embed: bool = False
    type: str = 'center'
    rel: bool = False
    use_nerf: bool = False
    extra_traj_layer: bool = False

    use_back_view: bool = False
    use_lr_view: bool = False


    extra_tr: bool = False
    vadv2_head_nhead: int = 8
    vadv2_head_nlayers: int = 3

    trajectory_sampling: TrajectorySampling = TrajectorySampling(
        time_horizon=4, interval_length=0.1
    )

    # img backbone
    use_final_fpn: bool = False
    use_img_pretrained: bool = False
    # image_architecture: str = "vit_large_patch14_dinov2.lvd142m"
    image_architecture: str = "resnet34"
    backbone_type: str = 'resnet'
    vit_ckpt: str = ''
    intern_ckpt: str = ''
    vov_ckpt: str = ''
    eva_ckpt: str = ''
    swin_ckpt: str = ''

    sptr_ckpt: str = ''
    map_ckpt: str = ''

    lr_mult_backbone: float = 1.0
    backbone_wd: float = 0.0
    weight_decay: float = 0.0

    # lidar backbone
    lidar_architecture: str = "resnet34"

    max_height_lidar: float = 100.0
    pixels_per_meter: float = 4.0
    hist_max_per_pixel: int = 5

    lidar_min_x: float = -32
    lidar_max_x: float = 32
    lidar_min_y: float = -32
    lidar_max_y: float = 32

    lidar_split_height: float = 0.2
    use_ground_plane: bool = False

    # new
    lidar_seq_len: int = 1

    camera_width: int = 2048
    camera_height: int = 512
    lidar_resolution_width: int = 256
    lidar_resolution_height: int = 256

    img_vert_anchors: int = camera_height // 32
    img_horz_anchors: int = camera_width // 32
    lidar_vert_anchors: int = lidar_resolution_height // 32
    lidar_horz_anchors: int = lidar_resolution_width // 32

    block_exp = 4
    n_layer = 2  # Number of transformer layers used in the vision backbone
    n_head = 4
    n_scale = 4
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1
    # Mean of the normal distribution initialization for linear layers in the GPT
    gpt_linear_layer_init_mean = 0.0
    # Std of the normal distribution initialization for linear layers in the GPT
    gpt_linear_layer_init_std = 0.02
    # Initial weight of the layer norms in the gpt.
    gpt_layer_norm_init_weight = 1.0

    perspective_downsample_factor = 1
    transformer_decoder_join = True
    detect_boxes = True
    use_bev_semantic = True
    use_semantic = False
    use_depth = False
    add_features = True

    # Transformer
    tf_d_model: int = 256
    tf_d_ffn: int = 1024
    tf_num_layers: int = 3
    tf_num_head: int = 8
    tf_dropout: float = 0.0

    # detection
    num_bounding_boxes: int = 30

    # loss weights
    agent_class_weight: float = 10.0
    agent_box_weight: float = 1.0
    bev_semantic_weight: float = 10.0

    # BEV mapping
    bev_semantic_classes = {
        1: ("polygon", [SemanticMapLayer.LANE, SemanticMapLayer.INTERSECTION]),  # road
        2: ("polygon", [SemanticMapLayer.WALKWAYS]),  # walkways
        3: ("linestring", [SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR]),  # centerline
        4: (
            "box",
            [
                TrackedObjectType.CZONE_SIGN,
                TrackedObjectType.BARRIER,
                TrackedObjectType.TRAFFIC_CONE,
                TrackedObjectType.GENERIC_OBJECT,
            ],
        ),  # static_objects
        5: ("box", [TrackedObjectType.VEHICLE]),  # vehicles
        6: ("box", [TrackedObjectType.PEDESTRIAN]),  # pedestrians
    }

    bev_pixel_width: int = lidar_resolution_width
    bev_pixel_height: int = lidar_resolution_height // 2
    bev_pixel_size: float = 1 / pixels_per_meter

    num_bev_classes = 7
    bev_features_channels: int = 64
    bev_down_sample_factor: int = 4
    bev_upsample_factor: int = 2

    @property
    def bev_semantic_frame(self) -> Tuple[int, int]:
        return (self.bev_pixel_height, self.bev_pixel_width)

    @property
    def bev_radius(self) -> float:
        values = [self.lidar_min_x, self.lidar_max_x, self.lidar_min_y, self.lidar_max_y]
        return max([abs(value) for value in values])
