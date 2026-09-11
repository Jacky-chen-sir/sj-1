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

import json
import logging
import os
import pickle
from typing import Any, Optional, Union
from typing import Dict, List

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

logger = logging.getLogger(__name__)

from navsim.agents.abstract_agent import AbstractAgent
from navsim.agents.gtrs_aug.aug_meta_arch import AugMetaArch
from navsim.agents.gtrs_aug.hydra_config_aug import HydraConfigAug, sync_drivesuprim_config_aliases
from navsim.agents.gtrs_aug.hydra_features_aug import HydraAugFeatureBuilder, HydraAugTargetBuilder
from navsim.agents.gtrs_aug.hydra_loss_fn_aug import hydra_kd_imi_agent_loss_robust, opd_distill_loss, \
    hydra_kd_imi_agent_loss_single_stage
from navsim.agents.gtrs_aug.hydra_model import HydraModel

_OPD_PDM_HEAD_KEYS = ('no_at_fault_collisions', 'drivable_area_compliance',
                      'time_to_collision_within_bound', 'ego_progress', 'driving_direction_compliance',
                      'lane_keeping', 'traffic_light_compliance', 'history_comfort')
# 三路损失都必须有的键（见 `opd_distill_loss`）。8 个 PDM 头是可选的：教师缓存
# 不带 heads 时把 head 那一路整体置零，而不是 KeyError。
_OPD_REQUIRED_KEYS = ('imi', 'coarse', 'topk_idx')


def _ops_teacher_to_tensors(items: List[Optional[Dict[str, Any]]], device,
                            vocab_size: int, topk: int,
                            expect_safe_fused: bool = True) -> Dict[str, torch.Tensor]:
    """把 per-token 教师缓存列表堆成本 batch 的张量字典。

    - `valid` [B]：该 token 的必需键是否齐全。缺失样本以零张量占位，损失端乘 0。
    - `valid_head` [B]：该 token 是否还带了 8 个 PDM 头（`OPD_STORE_HEADS=1`）。

    **不 assert "整 batch 全缺失"**：训练跑在 `DDPStrategy(static_graph=True, timeout=3600s)`
    下，某个 rank 抛异常不会让别的 rank 同步失败，而是所有人卡在 NCCL allreduce 上等满
    1 小时才超时。整 batch 缺失时返回全 0 的 valid + 零张量，损失为 0、计算图仍连通。
    """
    out: Dict[str, torch.Tensor] = {}

    def has(it, keys):
        return it is not None and all(k in it for k in keys)

    ok = [has(it, _OPD_REQUIRED_KEYS) for it in items]
    ok_head = [ok[i] and has(items[i], _OPD_PDM_HEAD_KEYS) for i in range(len(items))]
    out['valid'] = torch.tensor([1.0 if f else 0.0 for f in ok], dtype=torch.float32, device=device)
    out['valid_head'] = torch.tensor([1.0 if f else 0.0 for f in ok_head], dtype=torch.float32, device=device)

    if not any(ok):
        logger.warning(
            "整个 batch (%d 个 token) 的 OPD 教师缓存都缺失/不完整；本 step 蒸馏损失置 0。"
            "若持续出现请检查 teacher_score_dir 是否已生成完毕。", len(items))

    ref = next((items[i] for i in range(len(items)) if ok[i]), None)
    if ref is not None and expect_safe_fused and 'safe_fused_score' in ref:
        assert bool(ref['safe_fused_score']), (
            "教师缓存是用 opd.safe_fused_score=false 生成的（coarse 含 -inf），"
            "与训练端强制的 safe 版融合分数公式不一致；必须用 safe=true 重跑缓存")

    def stack(key, dtype, shape, valid_flags):
        vals = []
        for i, it in enumerate(items):
            if valid_flags[i]:
                t = torch.from_numpy(np.asarray(it[key]).copy())
            else:
                t = torch.zeros(shape)
            vals.append(t.to(dtype))
        return torch.stack(vals, 0).to(device)

    # 零占位张量必须和真实缓存同形，否则 torch.stack 直接炸。K 以缓存里的实际长度为准
    # （opd.topk_refine 只在损失端做截断，不要求与落盘 K 相等）。
    if ref is not None:
        topk = int(np.asarray(ref['topk_idx']).shape[-1])
        assert int(np.asarray(ref['coarse']).shape[-1]) == int(vocab_size), (
            f"教师缓存 coarse 维度 {np.asarray(ref['coarse']).shape[-1]} != config.vocab_size {vocab_size}")
    v_shape, k_shape = (vocab_size,), (topk,)
    for k in ('imi', 'coarse'):
        out[k] = stack(k, torch.float32, v_shape, ok)
    out['topk_idx'] = stack('topk_idx', torch.int64, k_shape, ok)
    for k in _OPD_PDM_HEAD_KEYS:
        out[k] = stack(k, torch.float32, v_shape, ok_head)
    if ref is not None and 'topk_score' in ref:
        out['topk_score'] = stack('topk_score', torch.float32, k_shape, ok)
    out['view_idx'] = torch.tensor(
        [-1 if not ok[i] else int(items[i].get('view_idx', 0)) for i in range(len(items))],
        dtype=torch.long, device=device)
    return out
from navsim.common.dataclasses import SensorConfig
from navsim.planning.training.abstract_feature_target_builder import (
    AbstractFeatureBuilder,
    AbstractTargetBuilder,
)


class GTRSAugAgent(AbstractAgent):
    def __init__(
            self,
            config: HydraConfigAug,
            lr: float,
            checkpoint_path: str = None,
            metrics=None,
    ):
        super().__init__(
            trajectory_sampling=config.trajectory_sampling
        )
        sync_drivesuprim_config_aliases(config)
        config.trajectory_pdm_weight = {
            'no_at_fault_collisions': 3.0,
            'drivable_area_compliance': 3.0,
            'time_to_collision_within_bound': 4.0,
            'ego_progress': 2.0,
            'driving_direction_compliance': 1.0,
            'lane_keeping': 2.0,
            'traffic_light_compliance': 3.0,
            'history_comfort': 1.0,
        }
        if config.lab.change_loss_weight:
            config.trajectory_pdm_weight = {
                'no_at_fault_collisions': 1.5,
                'drivable_area_compliance': 1.5,
                'time_to_collision_within_bound': 1.5,
                'ego_progress': 2.0,
                'driving_direction_compliance': 1.0,
                'lane_keeping': 2.0,
                'traffic_light_compliance': 1.0,
                'history_comfort': 1.0,
            }
        self._config = config
        self._lr = lr
        self.metrics = metrics
        self._checkpoint_path = checkpoint_path
        # OPD 离线蒸馏：训练时不建教师、不跑教师、不做 EMA，teacher_model=None。
        # teacher_mode='ema' 才复现原在线软标签教师（消融对照）。
        opd = getattr(config, 'opd', None)
        opd_offline = opd is not None and opd.enable and opd.teacher_mode == 'offline'
        if opd_offline:
            teacher_model = None
            student_model = HydraModel(config)
        else:
            teacher_model = HydraModel(config)
            student_model = HydraModel(config)
        self._opd_offline = opd_offline
        self.model = AugMetaArch(config, teacher_model, student_model)
        self.vocab_size = config.vocab_size
        self.backbone_wd = config.backbone_wd
        self.ensemble_aug = config.ego_perturb.ensemble_aug
        self.training = config.training

        if self.training:
            self.ori_vocab_pdm_score_full = pickle.load(
                open(f'{config.ori_vocab_pdm_score_full_path}', 'rb'))
            self.aug_vocab_pdm_score_dir = config.aug_vocab_pdm_score_dir

            with open(config.ego_perturb.offline_aug_file, 'r') as f:
                aug_data = json.load(f)
            assert aug_data['param']['rot'] == config.ego_perturb.rotation.offline_aug_angle_boundary
            self.aug_info = aug_data['tokens']

            if config.opd.enable and config.opd.teacher_mode == 'offline':
                assert config.opd.teacher_score_dir, (
                    "opd.enable 且 teacher_mode='offline' 时必须配置 opd.teacher_score_dir"
                )
                assert os.path.isdir(config.opd.teacher_score_dir), (
                    f"opd.teacher_score_dir 不存在：{config.opd.teacher_score_dir}"
                )
                # 启动时抽样校验缓存的元信息（词表 sha + offline_aug_file sha），
                # 否则换一份 8192.npy（与 test_8192_kmeans.npy 同尺寸但内容不同）会静默错位 Top-K。
                self._teacher_score_meta = self._load_teacher_meta(config.opd.teacher_score_dir)
                self._expected_vocab_sha1 = self._sha1_file_prefix(config.vocab_path)
                self._expected_offline_aug_sha1 = self._sha1_file_prefix(config.ego_perturb.offline_aug_file)

        self.only_ori_input = config.only_ori_input
        self.n_rotation_crop = config.student_rotation_ensemble

    @staticmethod
    def _sha1_file_prefix(path: Optional[str], nbytes: int = 4096) -> str:
        import hashlib
        if not path:
            return ""
        with open(path, 'rb') as f:
            buf = f.read(nbytes)
        try:
            size = os.path.getsize(path)
        except OSError:
            size = -1
        return hashlib.sha1(buf + str(size).encode()).hexdigest()

    def _load_teacher_meta(self, score_dir: str) -> Dict[str, Any]:
        """读取教师缓存目录里第一份 pickle 的元字段；没有元字段则按空 dict 返回（向后兼容旧缓存）。"""
        try:
            sample = next(f for f in os.listdir(score_dir) if f.endswith('.pkl'))
        except StopIteration:
            logger.warning("opd.teacher_score_dir=%s is empty; every token will be valid=0", score_dir)
            return {}
        with open(os.path.join(score_dir, sample), 'rb') as f:
            data = pickle.load(f)
        if not isinstance(data, dict):
            logger.warning("teacher cache %s is not a dict (legacy format); skipping meta validation", sample)
            return {}
        return {k: data[k] for k in ('vocab_sha1', 'vocab_size', 'offline_aug_file_sha1') if k in data}

    def _load_teacher_score(self, token: str, score_dir: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """per-token 懒加载教师蒸馏缓存；缺失/损坏返回 None（由调用方把该样本 valid 置 0）。

        不做任何 fallback——蒸馏信号缺失（置 0）比蒸馏错目标安全。
        第一次命中有效缓存时校验元信息（vocab_sha1 / offline_aug_file_sha1），不一致直接 raise。
        默认读原视图目录；on-policy 目录用 score_dir 显式传入。
        """
        if score_dir is None:
            score_dir = self._config.opd.teacher_score_dir
        path = os.path.join(score_dir, f"{token}.pkl")
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
        except Exception as e:
            logger.warning(
                "Missing/corrupt OPD teacher cache for token=%s (%s: %s); this sample gets valid=0",
                token, type(e).__name__, e,
            )
            return None
        if not getattr(self, '_teacher_meta_checked', False):
            v_sha = data.get('vocab_sha1')
            v_size = data.get('vocab_size')
            aug_sha = data.get('offline_aug_file_sha1')
            if v_sha is not None:
                assert v_sha == self._expected_vocab_sha1, (
                    f"教师缓存词表与研究词表不一致: cache={v_sha} expected={self._expected_vocab_sha1} ({self._config.vocab_path})"
                )
            if v_size is not None:
                assert int(v_size) == int(self._config.vocab_size), (
                    f"教师缓存词表大小 {v_size} 与 config.vocab_size {self._config.vocab_size} 不一致"
                )
            if aug_sha is not None and self._expected_offline_aug_sha1:
                assert aug_sha == self._expected_offline_aug_sha1, (
                    "教师缓存用的 offline_aug_file 与训练端不一致；旋转角按 (token,view) 静默错配，必须重跑缓存"
                )
            self._teacher_meta_checked = True
        return data

    def _load_aug_vocab_pdm_score(self, token: str):
        """Load per-token aug PDM scores; fall back to ori if pickle is corrupt/truncated."""
        path = os.path.join(self.aug_vocab_pdm_score_dir, f"{token}.pkl")
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            logger.warning(
                "Corrupt aug vocab PDM pickle for token=%s (%s: %s); "
                "falling back to ori scores for all rotation ensembles",
                token,
                type(e).__name__,
                e,
            )
            ori = self.ori_vocab_pdm_score_full[token]
            return [ori for _ in range(self._config.student_rotation_ensemble)]

    def name(self) -> str:
        """Inherited, see superclass."""

        return self.__class__.__name__

    def initialize(self) -> None:
        """Inherited, see superclass."""
        state_dict: Dict[str, Any] = torch.load(self._checkpoint_path, map_location=torch.device("cpu"))["state_dict"]
        incompatible = self.load_state_dict(
            {k.replace("agent.", ""): v for k, v in state_dict.items()}, strict=False)
        missing = list(incompatible.missing_keys)
        unexpected = list(incompatible.unexpected_keys)
        # OPD 离线蒸馏 ckpt 里没有 teacher.*；学生键缺失（结构改变/没加载上）必须报错而不是随机初始化。
        student_missing = [k for k in missing if k.startswith("model.student.")]
        assert not student_missing, (
            f"加载 ckpt 时学生侧缺失参数（结构已变 / 权重没加载上）：{student_missing[:8]}{' ...' if len(student_missing) > 8 else ''}"
        )
        teacher_missing = [k for k in missing if k.startswith("model.teacher.")]
        if teacher_missing and self._config.inference.model == "teacher":
            logger.warning(
                "加载的 ckpt 缺少 %d 个 teacher.* 参数，但 inference.model='teacher'——"
                "将用随机初始化的教师打分，分数不可信；请改用 inference.model=student 或加载完整 ckpt",
                len(teacher_missing),
            )
        if student_missing or unexpected:
            logger.info("initialize: unexpected_keys=%s", unexpected[:8])

    def get_sensor_config(self) -> SensorConfig:
        """Inherited, see superclass."""
        return SensorConfig(
            cam_f0=[0, 1, 2, 3],
            cam_l0=[0, 1, 2, 3],
            cam_l1=[0, 1, 2, 3],
            cam_l2=[0, 1, 2, 3],
            cam_r0=[0, 1, 2, 3],
            cam_r1=[0, 1, 2, 3],
            cam_r2=[0, 1, 2, 3],
            cam_b0=[0, 1, 2, 3],
            lidar_pc=[],
        )

    def get_target_builders(self) -> List[AbstractTargetBuilder]:
        return [HydraAugTargetBuilder(config=self._config)]

    def get_feature_builders(self) -> List[AbstractFeatureBuilder]:
        return [HydraAugFeatureBuilder(config=self._config)]

    def forward(self, batch) -> Dict[str, torch.Tensor]:
        features, targets, tokens = batch
        kwargs = {'tokens': tokens}

        teacher_ori_features = dict()
        student_ori_features = dict()
        teacher_ori_features['camera_feature'] = features['ori_teacher']
        teacher_ori_features['status_feature'] = features['status_feature']

        student_feat_dict_lst = []

        student_ori_features['camera_feature'] = features['ori']
        student_ori_features['status_feature'] = features['status_feature']
        student_feat_dict_lst.append(student_ori_features)
        if not self.only_ori_input and self._config.training:
            for i in range(self.n_rotation_crop):
                student_feat_dict_lst.append(
                    {
                        'camera_feature': features['rotated'][i],
                        'status_feature': features['status_feature'],
                    }
                )

            if self._config.use_mask_loss:
                kwargs = {
                    'collated_masks': features['collated_masks'],
                    "mask_indices_list": features['mask_indices_list'],
                    "masks_weight": features['masks_weight'],
                    "upperbound": features['upperbound'],
                    "n_masked_patches": features['n_masked_patches'],
                }

        teacher_pred, student_preds, loss_dict = self.model(teacher_ori_features, student_feat_dict_lst, **kwargs)
        return teacher_pred, student_preds, loss_dict

    def forward_train(self, features, interpolated_traj):
        return self.vadv2_model(features, interpolated_traj)

    def compute_loss(
            self,
            features: Dict[str, torch.Tensor],
            targets: Dict[str, torch.Tensor],
            predictions: List[Dict[str, torch.Tensor]],
            tokens=None
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        # ori
        ori_targets = {'trajectory': targets['ori_trajectory']}
        ori_predictions = predictions[0]
        scores = {}
        for k in self.metrics:
            tmp = [self.ori_vocab_pdm_score_full[token][k][None] for token in tokens]
            scores[k] = (torch.from_numpy(np.concatenate(tmp, axis=0))
                         .to(ori_predictions['trajectory'].device))
        ori_loss = hydra_kd_imi_agent_loss_robust(ori_targets, ori_predictions, self._config, scores)
        if self._config.only_ori_input:
            return {"ori": ori_loss}

        # aug
        _aug_vocab_pdm_score = {token: self._load_aug_vocab_pdm_score(token) for token in tokens}
        aug_loss = []
        for idx in range(self._config.student_rotation_ensemble):
            aug_targets = {'trajectory': targets['rotated_trajectories'][idx]}
            scores = {}
            for k in self.metrics:
                tmp = [_aug_vocab_pdm_score[token][idx][k][None] for token in tokens]
                scores[k] = (torch.from_numpy(np.concatenate(tmp, axis=0))
                             .to(predictions[idx + 1]['trajectory'].device))
            aug_loss.append(hydra_kd_imi_agent_loss_robust(aug_targets, predictions[idx + 1], self._config, scores))

        # Calculate average loss and loss dict
        avg_aug_loss = torch.mean(torch.stack([loss[0] for loss in aug_loss]))
        avg_aug_loss_dict = {}
        for key in aug_loss[0][1].keys():
            avg_aug_loss_dict[key] = torch.mean(torch.stack([loss[1][key] for loss in aug_loss]))
        return {
            "ori": ori_loss,
            "aug": (avg_aug_loss, avg_aug_loss_dict),
        }

    def compute_loss_soft_teacher(
            self,
            teacher_pred: Dict[str, torch.Tensor],
            student_pred: Dict[str, torch.Tensor],
            targets,
            tokens=None
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        sampled_timepoints = [5 * ii - 1 for ii in range(1, 9)]
        if self._config.soft_label_traj == 'first':
            traj_diff = teacher_pred['trajectory'][:, sampled_timepoints] - targets['ori_trajectory']
        elif self._config.soft_label_traj == 'final':
            traj_diff = teacher_pred['final_traj'][:, sampled_timepoints] - targets['ori_trajectory']

        clamped_traj_diff = torch.clamp(traj_diff, min=-self._config.soft_label_imi_diff_thresh,
                                        max=self._config.soft_label_imi_diff_thresh)
        # Apply clamped adjustment to original trajectory
        revised_targets = {'trajectory': targets['ori_trajectory'] + clamped_traj_diff}

        scores = {}
        revised_scores = {}
        for k in self.metrics:
            tmp = [self.ori_vocab_pdm_score_full[token][k][None] for token in tokens]
            scores[k] = torch.from_numpy(np.concatenate(tmp, axis=0)).to(teacher_pred['trajectory'].device).float()
            # Calculate difference and clamp to max 0.2
            diff = teacher_pred[k].sigmoid() - scores[k]
            _soft_label_score_diff_thresh = self._config.soft_label_score_diff_thresh
            clamped_diff = torch.clamp(diff, min=-_soft_label_score_diff_thresh, max=_soft_label_score_diff_thresh)
            # Apply clamped adjustment to original scores
            revised_scores[k] = scores[k] + clamped_diff

        soft_loss = hydra_kd_imi_agent_loss_robust(revised_targets, student_pred, self._config, revised_scores)
        return soft_loss

    def compute_loss_distill(
            self,
            features,
            targets: Dict[str, torch.Tensor],
            predictions: List[Dict[str, torch.Tensor]],
            tokens=None
    ):
        """OPD 蒸馏总入口：on-policy（若有）与 ori 各算一份，按权重合并。"""
        cfg_opd = self._config.opd

        def build_cache(score_dir: str) -> Dict[str, torch.Tensor]:
            items = [self._load_teacher_score(t, score_dir=score_dir) for t in tokens]
            return _ops_teacher_to_tensors(
                items, device=predictions[0]['imi'].device,
                vocab_size=int(self._config.vocab_size),
                topk=int(cfg_opd.topk_refine),
            )

        # 原视图蒸馏：教师缓存的 view_idx 必须是 0（原视图），配 predictions[0]
        cache_ori = build_cache(cfg_opd.teacher_score_dir)
        self._assert_view_match(cache_ori, 0, cfg_opd.teacher_score_dir)
        loss_ori, details = opd_distill_loss(predictions[0], cache_ori, self._config)

        total = loss_ori
        if cfg_opd.on_policy_rounds > 0:
            assert cfg_opd.teacher_onpolicy_score_dir, "on_policy_rounds>0 需要 opd.teacher_onpolicy_score_dir"
            # on-policy 教师在「旋转了 -dθ 的观测」上打分，必须配学生**同一旋转**的视图。
            # 原来这里用 predictions[0]（原视图）：师生看的是两张不同的图，蒸馏目标静默错配
            # 且完全不会报错——这是最危险的一类 bug，所以改成显式索引 + 双向断言。
            vi = int(cfg_opd.on_policy_view_idx)
            assert 0 <= vi < len(predictions), (
                f"opd.on_policy_view_idx={vi} 越界：本 batch 只有 {len(predictions)} 个学生视图"
                f"（predictions[0]=原视图，1.. 为 ego_perturb 旋转视图）。"
                f"请确认 ego_perturb.student_rotation_ensemble 已开启。")
            cache_op = build_cache(cfg_opd.teacher_onpolicy_score_dir)
            self._assert_view_match(cache_op, vi, cfg_opd.teacher_onpolicy_score_dir)
            loss_op2, details_op = opd_distill_loss(predictions[vi], cache_op, self._config)
            total = loss_ori + cfg_opd.on_policy_weight * loss_op2
            for k, v in details_op.items():
                details[f'onpolicy_{k}'] = v

        return total, details

    @staticmethod
    def _assert_view_match(cache: Dict[str, torch.Tensor], expected_view: int, score_dir: str) -> None:
        """教师缓存落盘的 view_idx 必须与它被配对的学生视图一致（-1 = 该样本缺失，跳过）。"""
        vi = cache.get('view_idx')
        if vi is None:
            return
        bad = vi[(vi >= 0) & (vi != expected_view)]
        assert bad.numel() == 0, (
            f"教师缓存 {score_dir} 的 view_idx={bad[0].item()}，但它被配到学生 predictions"
            f"[{expected_view}]。师生看的不是同一张图，蒸馏目标会静默错配；"
            f"请用 OPD_VIEW_IDX={expected_view} 重跑该缓存，或改 opd.on_policy_view_idx。")

    def compute_loss_multi_stage(
            self,
            features,
            targets: Dict[str, torch.Tensor],
            predictions: List[Dict[str, torch.Tensor]],
            tokens=None
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        _refinement_metrics = self.metrics
        if self._config.lab.refinement_metrics == 'dac_ep_lk_pdms':
            _refinement_metrics = _refinement_metrics + ['pdm_score']

        trajectory_vocab = predictions[0]['trajectory_vocab']

        result_dict = dict()
        # ori
        ori_loss_lst = []
        ori_predictions = predictions[0]['refinement']
        num_stage = len(ori_predictions)
        for i in range(num_stage):
            pred_i = ori_predictions[i]
            selected_indices_i = pred_i['indices_absolute']
            scores = {}
            for k in _refinement_metrics:
                tmp = [self.ori_vocab_pdm_score_full[token][k][None] for token in tokens]
                full_scores = torch.from_numpy(np.concatenate(tmp, axis=0)).to(
                    selected_indices_i.device)  # [bs, vocab_size]
                # Extract scores based on selected indices [bs, topk_stage_i]
                batch_size, topk = selected_indices_i.shape
                batch_indices = torch.arange(batch_size, device=selected_indices_i.device).unsqueeze(1).expand(-1, topk)
                scores[k] = full_scores[batch_indices, selected_indices_i]  # [bs, topk_stage_i]
            _kwargs = {}
            if self._config.lab.use_imi_learning_in_refinement:
                _kwargs['targets'] = {'trajectory': targets['ori_trajectory']}
                pred_i['trajectory_vocab'] = trajectory_vocab
            ori_loss_i = hydra_kd_imi_agent_loss_single_stage(pred_i, self._config, scores, **_kwargs)
            ori_loss_lst.append(ori_loss_i)
        total_ori_loss = sum([loss_tup[0] for loss_tup in ori_loss_lst])
        total_ori_loss_dict = {}
        for i, loss_tup in enumerate(ori_loss_lst):
            loss_dict = loss_tup[1]
            for _key, _value in loss_dict.items():
                total_ori_loss_dict[f"stage_{i + 2}_{_key}"] = _value
        result_dict['ori'] = (total_ori_loss, total_ori_loss_dict)
        if self._config.only_ori_input:
            return result_dict

        # aug
        _aug_vocab_pdm_score = {token: self._load_aug_vocab_pdm_score(token) for token in tokens}
        aug_loss_all_mode_lst = []
        for idx in range(self._config.student_rotation_ensemble):
            aug_loss_lst = []
            aug_idx_predictions = predictions[idx + 1]['refinement']
            for i in range(num_stage):
                aug_idx_pred_i = aug_idx_predictions[i]
                aug_idx_selected_indices_i = aug_idx_pred_i['indices_absolute']
                scores = {}
                for k in _refinement_metrics:
                    tmp = [_aug_vocab_pdm_score[token][idx][k][None] for token in tokens]
                    full_scores = torch.from_numpy(np.concatenate(tmp, axis=0)).to(aug_idx_selected_indices_i.device)
                    batch_size, topk = aug_idx_selected_indices_i.shape
                    batch_indices = torch.arange(batch_size, device=aug_idx_selected_indices_i.device).unsqueeze(
                        1).expand(-1, topk)
                    scores[k] = full_scores[batch_indices, aug_idx_selected_indices_i]
                _kwargs_idx = {}
                if self._config.lab.use_imi_learning_in_refinement:
                    _kwargs_idx['targets'] = {'trajectory': targets['rotated_trajectories'][idx]}
                    aug_idx_pred_i['trajectory_vocab'] = trajectory_vocab
                aug_loss_lst.append(
                    hydra_kd_imi_agent_loss_single_stage(aug_idx_pred_i, self._config, scores, **_kwargs_idx))
            aug_loss_single_mode = sum([loss_tup[0] for loss_tup in aug_loss_lst])
            aug_loss_single_mode_dict = {}
            for i, loss_tup in enumerate(aug_loss_lst):
                loss_dict = loss_tup[1]
                for _key, _value in loss_dict.items():
                    aug_loss_single_mode_dict[f"stage_{i + 2}_{_key}"] = _value
            aug_loss_all_mode_lst.append((aug_loss_single_mode, aug_loss_single_mode_dict))

        # Calculate average loss and loss dict
        avg_aug_loss = torch.mean(torch.stack([loss[0] for loss in aug_loss_all_mode_lst]))
        avg_aug_loss_dict = {}
        for key in aug_loss_all_mode_lst[0][1].keys():
            avg_aug_loss_dict[key] = torch.mean(torch.stack([loss[1][key] for loss in aug_loss_all_mode_lst]))
        result_dict['aug'] = (avg_aug_loss, avg_aug_loss_dict)

        return result_dict

    def get_optimizers(self) -> Union[Optimizer, Dict[str, Union[Optimizer, LRScheduler]]]:
        backbone_params_name = '_backbone.image_encoder'
        img_backbone_params = list(
            filter(lambda kv: backbone_params_name in kv[0], self.model.student.model.named_parameters()))
        default_params = list(
            filter(lambda kv: backbone_params_name not in kv[0], self.model.student.model.named_parameters()))
        params_lr_dict = [
            {'params': [tmp[1] for tmp in default_params]},
            {
                'params': [tmp[1] for tmp in img_backbone_params],
                'lr': self._lr * self._config.lr_mult_backbone,
                'weight_decay': self.backbone_wd
            }
        ]
        return torch.optim.Adam(params_lr_dict, lr=self._lr)

    def get_training_callbacks(self) -> List[pl.Callback]:
        ckpt_dir = f"{os.environ.get('NAVSIM_EXP_ROOT')}/{self._config.ckpt_path}/"
        return [
            ModelCheckpoint(
                save_top_k=30,
                monitor="val/loss-ori",
                mode="min",
                dirpath=ckpt_dir,
                filename="{epoch:02d}-{step:04d}",
            ),
            # Mid-epoch snapshots so a late-epoch crash does not lose ~hours of work.
            ModelCheckpoint(
                every_n_train_steps=1000,
                save_on_train_epoch_end=False,
                save_top_k=-1,
                dirpath=ckpt_dir,
                filename="step-{step:06d}",
            ),
        ]


def _get_norm_tensor(t):
    norms = torch.norm(t, p=2, dim=1, keepdim=True)
    return t / norms
