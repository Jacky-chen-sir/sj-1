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
from dataclasses import dataclass, field
from typing import Optional, Tuple

from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.common.maps.abstract_map import SemanticMapLayer
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from navsim.agents.transfuser.transfuser_config import TransfuserConfig

NAVSIM_DEVKIT_ROOT = os.environ.get("NAVSIM_DEVKIT_ROOT")


def get_opd_config(config):
    """Read config.opd without crashing OmegaConf struct (missing key is not getattr-safe)."""
    from omegaconf import DictConfig, OmegaConf

    if isinstance(config, DictConfig):
        return OmegaConf.select(config, "opd", default=None)
    return getattr(config, "opd", None)


def is_opd_offline(config) -> bool:
    opd = get_opd_config(config)
    if opd is None:
        return False
    enable = opd.get("enable") if isinstance(opd, dict) else getattr(opd, "enable", False)
    mode = opd.get("teacher_mode") if isinstance(opd, dict) else getattr(opd, "teacher_mode", None)
    return bool(enable) and mode == "offline"


def sync_drivesuprim_config_aliases(config: "HydraConfigAug") -> "HydraConfigAug":
    """Map DriveSuprim flat fields onto nested gtrs_aug config (safe to call multiple times)."""
    if config.ego_perturb.n_student_rotation_ensemble is not None and config.ego_perturb.n_student_rotation_ensemble >= 0:
        config.student_rotation_ensemble = int(config.ego_perturb.n_student_rotation_ensemble)
    if config.ego_perturb.offline_aug_angle_boundary is not None and config.ego_perturb.offline_aug_angle_boundary >= 0:
        config.ego_perturb.rotation.offline_aug_angle_boundary = float(
            config.ego_perturb.offline_aug_angle_boundary
        )
        config.ego_perturb.rotation.enable = True

    config.lab.use_imi_learning_in_refinement = bool(config.refinement.use_imi_learning_in_refinement)
    config.lab.use_first_stage_traj_in_infer = bool(config.inference.use_first_stage_traj_in_infer)
    config.lab.save_pickle = bool(config.inference.save_pickle)

    if config.refinement.refinement_approach == "transformer_decoder":
        config.refinement.use_offset_refinement_v2 = False

    # OPD：先把 hydra `--convert all` 下的 dict 强制挽回 dataclass（`++agent.config.opd.*`
    # 在无 `_target_` 块的 yaml 上会把 opd 变成普通 dict，随后点属性访问必崩）。
    if getattr(config, "opd", None) is not None and not isinstance(config.opd, OPDConfig):
        config.opd = OPDConfig(**dict(config.opd))

    if config.opd is not None and config.opd.enable:
        assert config.opd.teacher_mode in ("offline", "ema", "none"), (
            f"opd.teacher_mode 只能是 offline/ema/none，收到 {config.opd.teacher_mode!r}")
        assert config.opd.lambda_decay in ("none", "cosine"), (
            f"opd.lambda_decay 只能是 none/cosine，收到 {config.opd.lambda_decay!r}")
        if config.opd.teacher_mode == "offline":
            has_ema = bool(config.opd.ema_eval or config.opd.ema_soft_label)
            # 软标签需要进程内 EMA 教师；没有它时 compute_loss_soft_teacher 会对 teacher_pred=None 下标崩。
            config.lab.ban_soft_label_loss = not bool(config.opd.ema_soft_label)
            if not has_ema:
                # 无 EMA 副本的 ckpt 没有 teacher.*，若 inference 还指向 teacher，eval 会静默
                # 用随机初始化的 teacher 输出垃圾分数。强制 student。
                config.inference.model = "student"
            # fused_coarse_score 的非 safe 分支走 `softmax(-1).log()` / `sigmoid().log()`，
            # 词表里必然存在被压到下溢的条目 → -inf。蒸馏要对这个张量做 gather/log_softmax，
            # -inf 会经 `0 * -inf` 变 NaN（valid=0 的样本尤其）。OPD 下必须走 safe 版。
            config.opd.safe_fused_score = True
            assert not config.lab.optimize_prev_frame_traj_for_ec, (
                "optimize_prev_frame_traj_for_ec requires an in-process teacher; incompatible with OPD offline.")
            if config.training:
                assert config.opd.teacher_score_dir, (
                    "OPD offline 训练需要 agent.config.opd.teacher_score_dir 指向教师 per-token pickle 目录。")
                if config.opd.on_policy_rounds > 0:
                    assert config.opd.teacher_onpolicy_score_dir, (
                        "on_policy_rounds>0 需要 opd.teacher_onpolicy_score_dir")
        elif config.opd.teacher_mode == "ema":
            # EMA 自蒸馏对照（消融 B1）：恢复原在线软标签教师，完全不读缓存。
            # 不能 assert teacher_score_dir is None——sweep 复用同一份 yaml，该键总是被写上；
            # 这里直接清空，保证 `_load_teacher_score` 不会被误调用。
            config.opd.teacher_score_dir = None
            config.opd.teacher_onpolicy_score_dir = None
            config.lab.ban_soft_label_loss = False
        else:  # "none"：去掉 ViT-L 蒸馏、其余（EMA 软标签/EMA 评测/硬拷贝期）与 OPD 完全同配置的对照
            config.opd.teacher_score_dir = None
            config.opd.teacher_onpolicy_score_dir = None
            config.lab.ban_soft_label_loss = not bool(config.opd.ema_soft_label)
            if not (config.opd.ema_eval or config.opd.ema_soft_label):
                config.inference.model = "student"
    return config


@dataclass
class InferConfig:
    model: str = "teacher"  # teacher or student
    use_aug: bool = True  # whether using teacher augmentation
    # DriveSuprim flat aliases (synced onto LabConfig in HydraConfigAug.__post_init__)
    use_first_stage_traj_in_infer: bool = False
    save_pickle: bool = False


@dataclass
class RotationConfig:
    enable: bool = False
    fixed_angle: float = 0  # degree, positive: turn left
    offline_aug_angle_boundary: float = 0
    change_camera: bool = False
    crop_from_panoramic: bool = False


@dataclass
class VAConfig:
    # vel and acc perturb
    enable: bool = False
    offline_aug_boundary: float = 0


@dataclass
class EgoPerturbConfig:
    mode: str = 'fixed'  # 'fixed' or 'load_from_offline'
    ensemble_aug: bool = False
    offline_aug_file: str = '???'
    rotation: RotationConfig = RotationConfig()
    va: VAConfig = VAConfig()
    # DriveSuprim flat aliases (synced in HydraConfigAug.__post_init__)
    n_student_rotation_ensemble: int = -1  # >=0 overrides HydraConfigAug.student_rotation_ensemble
    offline_aug_angle_boundary: float = -1.0  # >=0 overrides rotation.offline_aug_angle_boundary


@dataclass
class CameraProblemConfig:
    shutdown_enable: bool = False
    shutdown_mode: int = 1
    shutdown_probability: float = 0

    noise_enable: bool = False  # randomly set the pixel value
    noise_percentage: float = 0

    gaussian_enable: bool = False
    gaussian_mode: str = 'random'  # random or load_from_offline
    gaussian_probability: float = 0.0  # probability of applying gaussian noise to an image
    gaussian_mean: float = 0.0  # mean of gaussian noise
    gaussian_min_std: float = 0.05  # minimum std when using random std
    gaussian_max_std: float = 0.25  # maximum std when using random std
    gaussian_offline_file: str = ''  # file path for offline gaussian noise parameters

    # Weather augmentation settings
    weather_enable: bool = False
    weather_aug_mode: str = 'random'  # random or load_from_offline
    fog_prob: float = 0.2  # probability of applying fog effect
    rain_prob: float = 0.2  # probability of applying rain effect
    snow_prob: float = 0.2  # probability of applying snow effect


@dataclass
class DinoConfig:
    loss_weight: float = 1.0
    head_n_prototypes: int = 65536
    head_bottleneck_dim: int = 256
    head_nlayers: int = 3
    head_hidden_dim: int = 2048
    koleo_loss_weight: float = 0.1


@dataclass
class IbotConfig:
    loss_weight: float = 1.0
    mask_sample_probability: float = 0.5
    mask_ratio_min_max: Tuple[float, float] = (0.1, 0.5)
    separate_head: bool = True
    head_n_prototypes: int = 65536
    head_bottleneck_dim: int = 256
    head_nlayers: int = 3
    head_hidden_dim: int = 2048


@dataclass
class RefinementConfig:
    use_multi_stage: bool = False
    # "transformer_decoder" is DriveSuprim alias for absolute TrajOffsetHead
    # (use_offset_refinement_v2=False). "offset_decoder" is the GTRS name.
    refinement_approach: str = "offset_decoder"
    num_refinement_stage: int = 1  # 2
    stage_layers: str = "3"  # "3+3"
    topks: str = "256"  # "256+64"

    use_mid_output: bool = True
    use_offset_refinement: bool = True  # abandoned
    use_offset_refinement_v2: bool = False
    use_separate_stage_heads: bool = True

    traj_expansion_in_infer: bool = False
    n_total_traj: int = 1024
    # DriveSuprim alias → synced to lab.use_imi_learning_in_refinement
    use_imi_learning_in_refinement: bool = True


@dataclass
class LabConfig:
    check_top_k_traj: bool = False
    num_top_k: int = 64
    test_full_vocab_pdm_score_path: str = "???"
    use_first_stage_traj_in_infer: bool = False

    change_loss_weight: bool = False
    use_imi_learning_in_refinement: bool = True
    adjust_refinement_loss_weight: bool = False  # change refinement loss weight: 256 / 8192.0
    adjust_refinement_score_weight: bool = False  # change dac, ep, lk score weight to 2 times
    ban_soft_label_loss: bool = False
    optimize_prev_frame_traj_for_ec: bool = False
    refinement_metrics: str = "all"  # 'all' or 'dac_ep_lk' or 'dac_ep_lk_pdms'
    use_higher_res_feat_in_refinement: bool = False

    use_cosine_ema_scheduler: bool = False
    ema_momentum_start: float = 0.99
    update_buffer_in_ema: bool = False
    save_pickle: bool = False


@dataclass
class OPDConfig:
    """OPD 离线蒸馏（论文第 4 章）。

    教师 = 冻结的 ViT-L，离线打分、per-token 缓存；学生 = R34，训练时 ViT-L 完全不前向。
    offline 模式下默认仍保留一份学生的 EMA 副本（ema_eval / ema_soft_label），
    关掉这两个开关即退化为纯离线蒸馏（此时 sync 强制 ban_soft_label_loss + inference=student）。
    """
    enable: bool = False
    teacher_mode: str = "offline"          # offline | ema | none   (ema = 复现原行为，作消融对照)
    teacher_score_dir: Optional[str] = None         # per-token pickle 目录（原视图）
    teacher_onpolicy_score_dir: Optional[str] = None  # per-token pickle 目录（on-policy 视图）

    # 温度按头类型拆分：imi 是 8192-way softmax，PDM 头是逐轨迹 Bernoulli(sigmoid)，
    # 融合分数是 log 空间且跨度 ~100 nats，三者有效区间不同，不能共用一个 tau。
    tau_imi: float = 2.0                   # 粗筛 imi 分布蒸馏
    tau_head: float = 2.0                  # 8 个 PDM 头的逐轨迹二元 KL
    tau_list: float = 2.0                  # 精排/召回 listwise（基于融合分数）

    lambda_imi: float = 1.0                # 粗筛 imi 分布蒸馏
    lambda_head: float = 1.0               # 8 个 PDM 头逐轨迹二元 KL
    lambda_refine: float = 1.0             # 精排 listwise（教师 Top-K 子集上的融合分数）
    lambda_recall: float = 0.5             # 召回集合监督（教师 Top-K' 质量不低于阈值）

    topk_refine: int = 256                 # 精排 listwise 的 Top-K（≤ 教师缓存落盘的 topk，超出会被截断）
    topk_recall: int = 32                  # 召回集合监督的 Top-K

    # **默认 False**：`opd` 是 field(default_factory=OPDConfig)，永不为 None，所以这个默认值
    # 会落到所有 gtrs_aug agent（含 baseline `gtrs_aug_drivesuprim_r34/_vit`）的 eval 上。
    # 默认 True 会静默改掉既有 ckpt 的融合分数公式（safe 版与原版不 bit-wise 相等）。
    # OPD offline 下由 sync_drivesuprim_config_aliases 强制置 True。
    safe_fused_score: bool = False
    # 教师缓存是否包含 8 头 logits。False 时训练端自动把 lambda_head 路置零（valid_head=0），
    # 不会 KeyError；缓存工具侧由 OPD_STORE_HEADS 控制，两边必须一致。
    store_heads: bool = True

    on_policy_rounds: int = 0              # 0 = 关闭该分支
    on_policy_weight: float = 0.5
    # on-policy 教师是在「旋转了 -dθ 的观测」上打分的，所以必须配对学生**同一旋转**的视图，
    # 不能配 predictions[0]（原视图）。collect 脚本把 dθ 写进 offline_aug_file 的 view 0，
    # 对应学生 predictions[1]。缓存 payload 里的 view_idx 会与这个值做一致性断言。
    on_policy_view_idx: int = 1

    # —— 学生权重 EMA（offline 模式下也保留一份与学生同构的 EMA 副本）——
    # 官方配方评测的是 EMA 教师（inference.model=teacher），纯 offline 评测的是裸学生：
    # epoch≥3 后 EMA 的权重平均加成只有 base 吃得到，师生对比不同口径。
    # ema_eval：建 EMA 副本并默认评测它（不增加前向，只是每步一次权重滑动平均）。
    ema_eval: bool = True
    # ema_soft_label：同时恢复原配方的在线软标签损失（EMA 教师在原视图上的限幅软目标）。
    # 与离线 ViT-L 蒸馏互补：前者前期强（冻结强教师），后者后期强（随学生共同进化的集成）。
    # 代价：每步多一次 R34 无梯度前向（即原配方的开销）。
    ema_soft_label: bool = True
    # R34 原配方前 3 个 epoch m=0（EMA=学生硬拷贝）。OPD 的学生在 ViT-L 监督下 1 个 epoch
    # 就已进入可用区间，EMA 可以更早开始集成；非 ResNet 骨干的原配方本来就从 epoch 0 起 m=0.992。
    ema_hardcopy_epochs: int = 1

    # —— 蒸馏权重调度 ——
    # 教师在它自己的训练集上的输出 ≈ 标签本身，蒸馏的边际价值随学生逼近教师而递减；
    # 前期保持全权重吃冷启动红利，后期衰减到 lambda_final_ratio，把学生交还给 GT + 软标签。
    lambda_decay: str = "cosine"           # none | cosine（按 optimizer step 计）
    lambda_final_ratio: float = 0.3


@dataclass
class HydraConfigAug(TransfuserConfig):
    seq_len: int = 2
    trajectory_imi_weight: float = 1.0
    trajectory_pdm_weight = {
        'no_at_fault_collisions': 3.0,
        'drivable_area_compliance': 3.0,
        'time_to_collision_within_bound': 4.0,
        'ego_progress': 2.0,
        'driving_direction_compliance': 1.0,
        'lane_keeping': 2.0,
        'traffic_light_compliance': 3.0,
        'history_comfort': 1.0,
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

    ckpt_path: str = None
    sigma: float = 0.5
    use_pers_bev_embed: bool = False
    type: str = 'center'
    rel: bool = False
    use_nerf: bool = False
    extra_traj_layer: bool = False

    use_back_view: bool = False

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

    n_camera: int = 3  # 1 or 3 or 5

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

    # robust setting
    training: bool = True
    ego_perturb: EgoPerturbConfig = EgoPerturbConfig()
    camera_problem: CameraProblemConfig = CameraProblemConfig()
    only_ori_input: bool = False  # 如果是 True，说明是原来的训练设置
    student_rotation_ensemble: int = 3
    ori_vocab_pdm_score_full_path: str = "???"
    aug_vocab_pdm_score_dir: str = "???"
    pdm_closed_traj_path: str = "???"
    weakly_supervised_imi_learning: bool = False  # 直接不学 augmented 之后的 traj
    pdm_close_traj_for_augmented_gt: bool = False
    traj_smoothing: bool = False  # pdm_close_traj_for_augmented_gt 为 false 时，本来应该直接对原来的 traj 做旋转变换，但是 smoothing 可以将轨迹跟车的运动速度更加贴合

    only_imi_learning: bool = False

    soft_label_traj: str = 'first'  # first or final
    soft_label_imi_diff_thresh: float = 1.0
    soft_label_score_diff_thresh: float = 0.15

    use_rotation_loss: bool = False
    use_mask_loss: bool = False
    dino: DinoConfig = DinoConfig()
    ibot: IbotConfig = IbotConfig()
    refinement: RefinementConfig = RefinementConfig()

    inference: InferConfig = InferConfig()

    lab: LabConfig = LabConfig()
    opd: OPDConfig = field(default_factory=OPDConfig)

    def __post_init__(self):
        sync_drivesuprim_config_aliases(self)

    @property
    def bev_semantic_frame(self) -> Tuple[int, int]:
        return (self.bev_pixel_height, self.bev_pixel_width)

    @property
    def bev_radius(self) -> float:
        values = [self.lidar_min_x, self.lidar_max_x, self.lidar_min_y, self.lidar_max_y]
        return max([abs(value) for value in values])
