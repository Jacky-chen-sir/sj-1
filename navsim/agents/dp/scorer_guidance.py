# -*- coding: utf-8 -*-
"""第 3 章创新点 A/B：评价器引导的轨迹生成。

论文位置：3.4.7（评价器引导的采样）与 3.5.5（指标结构感知的生成）。

## 为什么需要这个模块

GTRS（CVPR 2025）与 GoalFlow（CVPR 2025）都把「生成」与「评价」**解耦**：生成器只负责
产出多样的提案，评价器负责从中选优。但候选集的质量决定了评价器的性能上限——一个对下游
评价器无感知的生成器，会产出一批「几何上多样、但对评价器而言难以区分」的候选。

本章的方法（A）把第 4 章的评分头接进生成训练：评分头是**可微的**，且其融合准则在八项
可预测指标上**恒等于 log EPDMS**（见 4.2.4 节）。因此可以直接用它的分数作为生成器的
引导信号，使提案偏向高分区。

方法（B）进一步利用 EPDMS 的**聚合结构**：安全项是乘性合取（任一因子归零即整分归零），
舒适项是加性析取（可互补）。据此，生成器应当
  · 保证提案**不落入安全不可行区**（乘性 → 一票否决）
  · 在**舒适维度**上保持多样性（加性 → 这才是评价器能区分的地方）

## 实现要点

评分器的 `HydraTrajHead.forward` 已经支持对**任意轨迹**打分（`interpolated_traj` 分支，
见 `hydra_model.py:375-390`），本模块复用同一条通路：把生成轨迹当作额外的「候选」送进
`pos_embed → encoder → transformer → heads`。

特征适配：第 3 章的 `HydraFeatureBuilder` 与评分器的 `HydraAugFeatureBuilder` 产出的
ego status 构造方式**完全一致**（`driving_command + velocity + acceleration` = 8 维），
只是键名不同（`ego_status` vs `status_feature`），因此只需一次改名即可复用同一份输入，
无需重复跑特征构建。

显存：评分器（ResNet34 学生）冻结后仅前向，参数量与第 3 章骨干同级；BEV 特征每个 batch
只算一次，被 N 条提案共享。
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from navsim.agents.gtrs_aug.hydra_model import (
    HydraModel, fused_coarse_score, _SAFE_HEAD_KEYS,
)

logger = logging.getLogger(__name__)


class ScorerGuidance(nn.Module):
    """把第 4 章的评分器包装成第 3 章可调用的「轨迹打分器」。

    只做两件事：
      1. `encode_scene(features)` —— 跑一次评分器骨干，得到 BEV 特征与状态编码（每个 batch 只算一次）
      2. `score_traj(scene, traj_abs)` —— 对任意 [B, N, H, 3] 的绝对轨迹打分

    评分器默认**冻结**（`requires_grad=False`）。梯度仍然会穿过它回传到生成器——这正是
    引导信号起作用的方式；冻结只是不更新评分器自身的参数。
    """

    def __init__(self, scorer_config, ckpt_path: str = '', freeze: bool = True,
                 use_dual_stream: bool = True):
        super().__init__()
        self.scorer = HydraModel(scorer_config)
        self._dual_stream = bool(use_dual_stream)
        self._num_poses = int(scorer_config.trajectory_sampling.num_poses)

        if ckpt_path:
            self._load_ckpt(ckpt_path)
        if freeze:
            for p in self.scorer.parameters():
                p.requires_grad_(False)
            self.scorer.eval()
        logger.info(
            f'ScorerGuidance: ckpt={ckpt_path or "<random>"} freeze={freeze} '
            f'dual_stream={use_dual_stream}'
        )

    # ------------------------------------------------------------------
    def _load_ckpt(self, path: str) -> None:
        """加载评分器权重。兼容 Lightning 的 `state_dict` 外层包装与 `model.` 前缀。"""
        sd = torch.load(path, map_location='cpu')
        for key in ('state_dict', 'model', 'model_state_dict'):
            if isinstance(sd, dict) and key in sd:
                sd = sd[key]
                break
        sd = {k[len('model.'):]: v for k, v in sd.items() if k.startswith('model.')} or sd
        missing, unexpected = self.scorer.load_state_dict(sd, strict=False)
        if missing:
            logger.warning(f'ScorerGuidance: 缺 {len(missing)} 个键，前 5 个: {missing[:5]}')
        if unexpected:
            logger.warning(f'ScorerGuidance: 多 {len(unexpected)} 个键，前 5 个: {unexpected[:5]}')

    # ------------------------------------------------------------------
    @staticmethod
    def adapt_features(features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """把第 3 章的 features 改成评分器 `forward` 认得的键。

        两边的相机特征键名相同（都叫 `camera_feature`），只有 ego status 的键名不同
        （`ego_status` vs `status_feature`）。评分器内部按 `status_feature[0]` 取当前帧，
        所以这里包成单元素 list。
        """
        out = {'camera_feature': features['camera_feature']}
        if 'status_feature' in features:
            out['status_feature'] = features['status_feature']
        else:
            es = features['ego_status']
            if isinstance(es, list):
                es = es[0] if torch.is_tensor(es[0]) else torch.stack(es, 0)
            if es.dim() == 3:          # [B, T, 8] → 取最后一帧
                es = es[:, -1]
            out['status_feature'] = [es]
        return out

    # ------------------------------------------------------------------
    def encode_scene(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """跑一次评分器骨干，得到 BEV 特征（keyval）与状态编码。

        每个 batch 只需调用一次，N 条提案共享结果——这是引导信号不显著增加开销的关键。
        """
        f = self.adapt_features(features)
        camera_feature = f['camera_feature']
        if isinstance(camera_feature, list):
            camera_feature = camera_feature[-1]
        status_feature = f['status_feature'][0]

        sc = self.scorer
        img_feat_dict = sc.img_feat_blc_dict(camera_feature, masks=None, return_class_token=True)
        keyval = img_feat_dict['avg_feat'] + sc._keyval_embedding.weight[None, ...]
        status_encoding = sc._status_encoding(status_feature)
        return {'keyval': keyval, 'status_encoding': status_encoding}

    # ------------------------------------------------------------------
    def score_traj(self, scene: Dict[str, torch.Tensor],
                   traj_abs: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """对 [B, N, H, 3] 的**绝对**轨迹打分。

        返回 `(score [B, N], head_out)`。`score` 即论文式 (4-8) 的对数融合分数，在八项
        可预测指标上等于 log EPDMS（未含跨帧的 EC）。

        走的通路与 `HydraTrajHead.forward` 的 `dp_preds` 分支完全一致：
            pos_embed → encoder → transformer(kv) → +status → 逐头线性 → fused
        """
        head = self.scorer._trajectory_head
        B, N, H, C = traj_abs.shape
        assert C == 3, f'轨迹最后一维应为 (x, y, θ)，得到 {C}'
        assert H == self._num_poses, f'轨迹长度应为 {self._num_poses}，得到 {H}'

        flat = traj_abs.reshape(B * N, H, C).reshape(B * N, -1)
        emb = head.pos_embed(flat)
        emb = head.encoder(emb)

        keyval = scene['keyval']
        tr_out = head.transformer(emb.view(B, N, -1), keyval)          # [B, N, C]
        dist_status = tr_out + scene['status_encoding'].unsqueeze(1)    # [B, N, C]

        head_out = {k: h(dist_status).squeeze(-1) for k, h in head.heads.items()}
        score = fused_coarse_score(
            head_out, self.scorer._config, safe=True, dual_stream=self._dual_stream)
        return score, head_out

    # ------------------------------------------------------------------
    def safety_product(self, head_out: Dict[str, torch.Tensor]) -> torch.Tensor:
        """四项安全因子的概率乘积 `[B, N]`——EPDMS 的乘性部分。

        这就是「安全不可行区」的判据：乘积越小，该候选的 EPDMS 越可能被整体抹零。
        """
        psafe = None
        for k in _SAFE_HEAD_KEYS:
            p = head_out[k].sigmoid()
            psafe = p if psafe is None else psafe * p
        return psafe

    def comfort_score(self, head_out: Dict[str, torch.Tensor],
                      weights: Optional[Dict[str, float]] = None) -> torch.Tensor:
        """舒适组的加权加性聚合值 `[B, N]`（EPDMS 的加性部分，不含 EC）。

        默认权重与 EPDMS 一致：TTC=5, EP=5, LK=2, HC=2（W=14，即 4.2.3 节的
        `M_add^net` 口径）。
        """
        w = weights or {
            'time_to_collision_within_bound': 5.0,
            'ego_progress': 5.0,
            'lane_keeping': 2.0,
            'history_comfort': 2.0,
        }
        num = None
        den = 0.0
        for k, wk in w.items():
            if k not in head_out:
                continue
            p = head_out[k].sigmoid()
            num = wk * p if num is None else num + wk * p
            den += wk
        if num is None:
            raise KeyError(f'head_out 里没有任何舒适头，现有: {list(head_out)}')
        return num / max(den, 1e-6)


# ======================================================================
# 损失项（创新点 A / B）
# ======================================================================
def guidance_loss(score: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """创新点 A：评分引导损失。

    对同一场景内 N 条提案的评分作 softmax，取负的期望分数（即最大化平均分）。
    用 softmax 加权而非直接取 max，是为了避免生成器只优化「最好的一条」而放弃其余，
    从而保持提案集合的整体质量——毕竟评价器要从整池里选。
    """
    w = F.softmax(score / max(temperature, 1e-6), dim=1)
    return -(w * score).sum(dim=1).mean()


def safety_feasibility_loss(psafe: torch.Tensor, margin: float = 0.5) -> torch.Tensor:
    """创新点 B-1：安全可行性。

    EPDMS 的安全项是**乘性合取**，任一因子归零即整分归零。因此对生成器而言，产出
    「安全乘积低于 margin」的提案是纯浪费——它在评价阶段必然拿零分。

    这里用 hinge 形式：只惩罚低于 margin 的部分，高于 margin 的不再优化（避免把生成器
    推向极端保守）。margin 取 0.5 对应「四项安全概率的几何均值约 0.84」。
    """
    return F.relu(margin - psafe).mean()


def comfort_diversity_loss(comfort: torch.Tensor, target_std: float = 1.0) -> torch.Tensor:
    """创新点 B-2：舒适多样性。

    EPDMS 的舒适项是**加性析取**，评价器正是在这一维度上作区分。若生成器产出的提案在
    舒适维度上高度雷同，评价器的判别能力就被浪费了。

    这里鼓励同一场景内提案的舒适分数标准差接近 `target_std`（默认 1.0，即 log 空间下
    约 2.7 倍的分数跨度）。用「接近目标」而非「越大越好」，是为了避免生成器为了刷大
    方差而产出明显不合理的轨迹。
    """
    s = comfort.std(dim=1)
    return ((s - target_std) ** 2).mean()
