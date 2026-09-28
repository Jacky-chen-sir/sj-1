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

from typing import Dict

import torch
import torch.nn.functional as F

from navsim.agents.gtrs_aug.hydra_config_aug import HydraConfigAug


def hydra_kd_imi_agent_loss_robust(
        targets: Dict[str, torch.Tensor], predictions: Dict[str, torch.Tensor], config: HydraConfigAug,
        vocab_pdm_score
):
    """
    Helper function calculating complete loss of Transfuser
    :param targets: dictionary of name tensor pairings
    :param predictions: dictionary of name tensor pairings
    :param config: global Transfuser config
    :return: combined loss value
    """
    # if os.getenv('ROBUST_HYDRA_DEBUG') == 'true':
    #     import pdb; pdb.set_trace()

    no_at_fault_collisions, drivable_area_compliance, time_to_collision_within_bound, ego_progress = (
        predictions['no_at_fault_collisions'],
        predictions['drivable_area_compliance'],
        predictions['time_to_collision_within_bound'],
        predictions['ego_progress']
    )
    driving_direction_compliance, lane_keeping, traffic_light_compliance = (
        predictions['driving_direction_compliance'],
        predictions['lane_keeping'],
        predictions['traffic_light_compliance']
    )
    history_comfort = predictions['history_comfort']
    imi = predictions['imi']
    # if os.getenv('ROBUST_HYDRA_DEBUG') == 'true':
    #     import pdb; pdb.set_trace()
    _dtype = imi.dtype

    # 2 cls
    da_loss = F.binary_cross_entropy_with_logits(drivable_area_compliance,
                                                 vocab_pdm_score['drivable_area_compliance'].to(_dtype))
    ttc_loss = F.binary_cross_entropy_with_logits(time_to_collision_within_bound,
                                                  vocab_pdm_score['time_to_collision_within_bound'].to(_dtype))
    noc_loss = F.binary_cross_entropy_with_logits(no_at_fault_collisions, three_to_two_classes(
        vocab_pdm_score['no_at_fault_collisions'].to(_dtype)))
    progress_loss = F.binary_cross_entropy_with_logits(ego_progress, vocab_pdm_score['ego_progress'].to(_dtype))
    # expansion
    ddc_loss = F.binary_cross_entropy_with_logits(driving_direction_compliance, three_to_two_classes(
        vocab_pdm_score['driving_direction_compliance'].to(_dtype)))
    lk_loss = F.binary_cross_entropy_with_logits(lane_keeping, vocab_pdm_score['lane_keeping'].to(_dtype))
    tl_loss = F.binary_cross_entropy_with_logits(traffic_light_compliance,
                                                 vocab_pdm_score['traffic_light_compliance'].to(_dtype))

    comfort_loss = F.binary_cross_entropy_with_logits(history_comfort,
                                                      vocab_pdm_score['history_comfort'].to(_dtype))
    vocab = predictions["trajectory_vocab"]
    # B, 8 (4 secs, 0.5Hz), 3
    target_traj = targets["trajectory"]
    # 4, 9, ..., 39
    sampled_timepoints = [5 * k - 1 for k in range(1, 9)]
    B = target_traj.shape[0]
    l2_distance = -((vocab[:, sampled_timepoints][None].repeat(B, 1, 1, 1) - target_traj[:, None]) ** 2) / config.sigma
    """
    vocab: [vocab_size, 40, 3]
    vocab[:, sampled_timepoints]: [vocab_size, 8, 3]
    vocab[:, sampled_timepoints][None].repeat(B, 1, 1, 1): [b, vocab_size, 8, 3]
    target_traj[:, None]: [b, 1, 8, 3]
    l2_distance: [b, vocab_size, 8, 3]
    """
    imi_loss = F.cross_entropy(imi, l2_distance.sum((-2, -1)).softmax(1))

    imi_loss_final = config.trajectory_imi_weight * imi_loss

    noc_loss_final = config.trajectory_pdm_weight['no_at_fault_collisions'] * noc_loss
    da_loss_final = config.trajectory_pdm_weight['drivable_area_compliance'] * da_loss
    ttc_loss_final = config.trajectory_pdm_weight['time_to_collision_within_bound'] * ttc_loss
    progress_loss_final = config.trajectory_pdm_weight['ego_progress'] * progress_loss
    ddc_loss_final = config.trajectory_pdm_weight['driving_direction_compliance'] * ddc_loss
    lk_loss_final = config.trajectory_pdm_weight['lane_keeping'] * lk_loss
    tl_loss_final = config.trajectory_pdm_weight['traffic_light_compliance'] * tl_loss
    comfort_loss_final = config.trajectory_pdm_weight['history_comfort'] * comfort_loss

    # agent_class_loss, agent_box_loss = _agent_loss(targets, predictions, config)

    # agent_class_loss_final = config.agent_class_weight * agent_class_loss
    # agent_box_loss_final = config.agent_box_weight * agent_box_loss
    loss = (
            imi_loss_final
            + noc_loss_final
            + da_loss_final
            + ttc_loss_final
            + progress_loss_final
            + ddc_loss_final
            + lk_loss_final
            + tl_loss_final
            + comfort_loss_final
    )
    return loss, {
        'imi_loss': imi_loss_final,
        'pdm_noc_loss': noc_loss_final,
        'pdm_da_loss': da_loss_final,
        'pdm_ttc_loss': ttc_loss_final,
        'pdm_progress_loss': progress_loss_final,
        'pdm_ddc_loss': ddc_loss_final,
        'pdm_lk_loss': lk_loss_final,
        'pdm_tl_loss': tl_loss_final,
        'pdm_comfort_loss': comfort_loss_final
    }


def hydra_kd_imi_agent_loss_single_stage(
        predictions: Dict[str, torch.Tensor], config: HydraConfigAug, vocab_pdm_score, targets=None
):
    """
    Helper function calculating complete loss of Transfuser
    :param targets: dictionary of name tensor pairings
    :param predictions: dictionary of name tensor pairings
    :param config: global Transfuser config
    :return: combined loss value
    """

    # if os.getenv('ROBUST_HYDRA_DEBUG') == 'true':
    #     import pdb; pdb.set_trace()

    layer_results = predictions['layer_results']
    losses = {}
    total_loss = 0.0

    refinement_metrics = config.lab.refinement_metrics

    for layer, layer_result in enumerate(layer_results):

        if refinement_metrics == 'all':
            no_at_fault_collisions, drivable_area_compliance, time_to_collision_within_bound, ego_progress = (
                layer_result['no_at_fault_collisions'],
                layer_result['drivable_area_compliance'],
                layer_result['time_to_collision_within_bound'],
                layer_result['ego_progress']
            )
            driving_direction_compliance, lane_keeping, traffic_light_compliance = (
                layer_result['driving_direction_compliance'],
                layer_result['lane_keeping'],
                layer_result['traffic_light_compliance']
            )
            history_comfort = layer_result['history_comfort']

            _dtype = drivable_area_compliance.dtype

            da_loss = F.binary_cross_entropy_with_logits(drivable_area_compliance,
                                                         vocab_pdm_score['drivable_area_compliance'].to(_dtype))
            ttc_loss = F.binary_cross_entropy_with_logits(time_to_collision_within_bound,
                                                          vocab_pdm_score['time_to_collision_within_bound'].to(_dtype))
            noc_loss = F.binary_cross_entropy_with_logits(no_at_fault_collisions, three_to_two_classes(
                vocab_pdm_score['no_at_fault_collisions'].to(_dtype)))
            progress_loss = F.binary_cross_entropy_with_logits(ego_progress, vocab_pdm_score['ego_progress'].to(_dtype))
            # expansion
            ddc_loss = F.binary_cross_entropy_with_logits(driving_direction_compliance, three_to_two_classes(
                vocab_pdm_score['driving_direction_compliance'].to(_dtype)))
            lk_loss = F.binary_cross_entropy_with_logits(lane_keeping, vocab_pdm_score['lane_keeping'].to(_dtype))
            tl_loss = F.binary_cross_entropy_with_logits(traffic_light_compliance,
                                                         vocab_pdm_score['traffic_light_compliance'].to(_dtype))

            comfort_loss = F.binary_cross_entropy_with_logits(history_comfort,
                                                              vocab_pdm_score['history_comfort'].to(_dtype))

            noc_loss_final = config.trajectory_pdm_weight['no_at_fault_collisions'] * noc_loss
            da_loss_final = config.trajectory_pdm_weight['drivable_area_compliance'] * da_loss
            ttc_loss_final = config.trajectory_pdm_weight['time_to_collision_within_bound'] * ttc_loss
            progress_loss_final = config.trajectory_pdm_weight['ego_progress'] * progress_loss
            ddc_loss_final = config.trajectory_pdm_weight['driving_direction_compliance'] * ddc_loss
            lk_loss_final = config.trajectory_pdm_weight['lane_keeping'] * lk_loss
            tl_loss_final = config.trajectory_pdm_weight['traffic_light_compliance'] * tl_loss
            comfort_loss_final = config.trajectory_pdm_weight['history_comfort'] * comfort_loss

            loss = (
                    noc_loss_final
                    + da_loss_final
                    + ttc_loss_final
                    + progress_loss_final
                    + ddc_loss_final
                    + lk_loss_final
                    + tl_loss_final
                    + comfort_loss_final
            )

        else:
            drivable_area_compliance, ego_progress = (
                layer_result['drivable_area_compliance'],
                layer_result['ego_progress']
            )
            lane_keeping = layer_result['lane_keeping']

            _dtype = drivable_area_compliance.dtype

            da_loss = F.binary_cross_entropy_with_logits(drivable_area_compliance,
                                                         vocab_pdm_score['drivable_area_compliance'].to(_dtype))
            progress_loss = F.binary_cross_entropy_with_logits(ego_progress, vocab_pdm_score['ego_progress'].to(_dtype))
            # expansion
            lk_loss = F.binary_cross_entropy_with_logits(lane_keeping, vocab_pdm_score['lane_keeping'].to(_dtype))

            da_loss_final = config.trajectory_pdm_weight['drivable_area_compliance'] * da_loss
            progress_loss_final = config.trajectory_pdm_weight['ego_progress'] * progress_loss
            lk_loss_final = config.trajectory_pdm_weight['lane_keeping'] * lk_loss

            loss = (
                    da_loss_final
                    + progress_loss_final
                    + lk_loss_final
            )

            if refinement_metrics == 'dac_ep_lk_pdms':
                pdm = layer_result['pdm_score']
                pdm_loss = F.binary_cross_entropy_with_logits(pdm,
                                                              vocab_pdm_score['pdm_score'].to(_dtype))
                pdm_loss_final = 2.0 * pdm_loss
                loss += pdm_loss_final

        if config.lab.use_imi_learning_in_refinement:
            imi = layer_result['imi']
            vocab = predictions["trajectory_vocab"]
            # B, 8 (4 secs, 0.5Hz), 3
            target_traj = targets["trajectory"]
            # 4, 9, ..., 39
            sampled_timepoints = [5 * k - 1 for k in range(1, 9)]
            indices_absolute = predictions['indices_absolute']
            l2_distance = -((vocab[:, sampled_timepoints][indices_absolute] - target_traj[:, None]) ** 2) / config.sigma
            """
            vocab: [vocab_size, 40, 3]
            vocab[:, sampled_timepoints]: [vocab_size, 8, 3]
            vocab[:, sampled_timepoints][None].repeat(B, 1, 1, 1): [b, vocab_size, 8, 3]
            target_traj[:, None]: [b, 1, 8, 3]
            l2_distance: [b, vocab_size, 8, 3]
            """
            imi_loss = F.cross_entropy(imi, l2_distance.sum((-2, -1)).softmax(1))

            imi_loss_final = config.trajectory_imi_weight * imi_loss

            loss += imi_loss_final

        if config.lab.adjust_refinement_loss_weight:
            n_cur_traj = drivable_area_compliance.shape[1]
            loss *= n_cur_traj / config.vocab_size

        total_loss += loss
        losses[f'layer_{layer + 1}'] = loss

    return total_loss, losses


def three_to_two_classes(x):
    x[x == 0.5] = 0.0
    return x


# 8 个 PDM 头的名字按 yaml 的 `metrics:` 列表顺序（HydraConfigAug.trajectory_pdm_weight 的键）
_PDM_HEADS = (
    'no_at_fault_collisions', 'drivable_area_compliance', 'time_to_collision_within_bound',
    'ego_progress', 'driving_direction_compliance', 'lane_keeping',
    'traffic_light_compliance', 'history_comfort',
)

# 四项乘性（合取）安全头，论文式 (4-22) / (4-23) 的作用域。
# 必须与 `hydra_model._SAFE_HEAD_KEYS` 一致；不直接 import 是为了避免 loss 模块反向依赖模型模块。
_SAFE_HEAD_KEYS = (
    'no_at_fault_collisions', 'drivable_area_compliance',
    'driving_direction_compliance', 'traffic_light_compliance',
)


def opd_distill_loss(student_pred: Dict[str, torch.Tensor], teacher_cache: Dict[str, torch.Tensor],
                     config: HydraConfigAug):
    """OPD 蒸馏，返回 (total, dict)。

    每一路都可独立置零（对应消融的 λ 独立性与 on/off 开关）：
      1. imi 分布蒸馏：`KL(softmax(s_imi/τ_imi) ‖ softmax(t_imi/τ_imi)) · τ_imi²`
      2. 逐轨迹二元 KL：`KL(Bern(σ(t/τ_h)) ‖ Bern(σ(s/τ_h)))`，逐元素按 trajectory_pdm_weight 加权。
         遍历范围由 `opd.head_scope` 决定：'all'（默认）= 八项可预测指标，
         'safe' = 只蒸四项乘性安全头（论文式 (4-22) 的原始写法）。
      3. 精排 listwise（教师 Top-K 子集上的融合分数）+ 召回集合监督（教师 Top-K' 内部的 logsumexp）
      4. 乘积一致性约束（论文式 (4-23)，创新点 2）：对四项安全因子的**乘积**整体施加监督。
         第 2 路逐项独立，保证不了乘积；而乘积恰是 EPDMS 安全项的全部内容。

    约定（与教师缓存 `run_teacher_opd_cache.py` 对齐）：
      - teacher_cache['valid']: [B] float，0=该 token 的缓存缺失/损坏。乘 0 而非跳过分支，
        否则 DDP static_graph 会因某 step 少走一条 loss 分支而 RuntimeError。
      - teacher_cache['valid_head']: [B] float，0=该 token 的缓存不含 8 头 logits
        （`OPD_STORE_HEADS=0`）。第 2、4 路按这个 mask 置零，其余路不受影响。
      - teacher_cache['coarse']: [B, V] 数值安全融合分数（必须是 safe 版；非 safe 版含 -inf）。
      - teacher_cache['imi'] / 8 个 per-head 键：[B, V] 原始 logits（**顶层逐头键**，
        不是一个堆叠数组——落盘端 `run_teacher_opd_cache.py` 必须与此一致）。
      - 全部内部 `.float()`，杜绝 16-mixed 未来把 log_softmax/logsumexp 掉精度。
    """
    opd = config.opd
    device = student_pred['imi'].device
    valid = teacher_cache['valid'].float().to(device).view(-1)          # [B]
    n_valid = valid.sum().clamp(min=1.0)                                # 避免除 0
    # 8 头 logits 是可选缓存（OPD_STORE_HEADS）。缺失时这一路按样本置零，而不是 KeyError。
    valid_head = teacher_cache.get('valid_head')
    valid_head = valid if valid_head is None else valid_head.float().to(device).view(-1) * valid
    n_valid_head = valid_head.sum().clamp(min=1.0)

    def masked_mean(per_sample):                                        # [B] -> scalar，乘 mask 而非跳过
        return (per_sample * valid).sum() / n_valid

    def masked_mean_head(per_sample):
        return (per_sample * valid_head).sum() / n_valid_head

    out: Dict[str, torch.Tensor] = {}

    # ---------- 1. imi 分布蒸馏（前向 KL，均值=batchmean，乘 τ²） ----------
    s_log_imi = F.log_softmax(student_pred['imi'].float() / opd.tau_imi, dim=-1)      # [B,V]
    t_log_imi = F.log_softmax(teacher_cache['imi'].float().to(device) / opd.tau_imi, dim=-1)
    loss_imi = F.kl_div(s_log_imi, t_log_imi, reduction='none', log_target=True).sum(-1) * (opd.tau_imi ** 2)
    out['distill_imi'] = masked_mean(loss_imi)

    # ---------- 2. PDM 头逐轨迹二元 KL（等价 F.binary_cross_entropy_with_logits(t_logit, σ(s))），logsigmoid 安全化 ----------
    # 遍历范围由 head_scope 决定：'all' = 八项可预测指标（默认，现有行为）；
    # 'safe' = 只蒸四项乘性安全头，对应论文式 (4-22) 的原始写法。
    head_scope = str(getattr(opd, 'head_scope', 'all'))
    scope_heads = _SAFE_HEAD_KEYS if head_scope == 'safe' else _PDM_HEADS
    loss_head = torch.zeros(student_pred['imi'].shape[0], device=device)
    w = config.trajectory_pdm_weight
    for name in scope_heads:
        t_logit = teacher_cache[name].float().to(device) / opd.tau_head   # 教师 logit，当 soft 目标温度
        s_logit = student_pred[name].float() / opd.tau_head
        # BCE-with-soft-target(s_logit, σ(t_logit)) * τ² 即逐轨迹 Bernoulli KL 的原值
        bce = F.binary_cross_entropy_with_logits(s_logit, torch.sigmoid(t_logit), reduction='none')  # [B,V]
        kl = bce.mean(-1) * (opd.tau_head ** 2) * float(w.get(name, 1.0))
        loss_head = loss_head + kl
        out[f'distill_head_{name}'] = masked_mean_head(kl)
    out['distill_head'] = masked_mean_head(loss_head)

    # ---------- 2b. 乘积一致性约束（论文式 (4-23)，创新点 2）----------
    # 第 2 路逐项独立，每一项都对不代表**乘积**对；而乘积恰是 EPDMS 安全项的全部内容。
    # 在 log 空间度量乘积的一致性：连乘的比值取对数后变成差，各因子贡献可加、可比较，
    # 同时避免连乘在 fp32 下溢。abs 在 0 点用次梯度（PyTorch 取 0），不影响收敛。
    s_safe_log = sum(F.logsigmoid(student_pred[k].float()) for k in _SAFE_HEAD_KEYS)   # [B,V]
    t_safe_log = sum(F.logsigmoid(teacher_cache[k].float().to(device)) for k in _SAFE_HEAD_KEYS)
    loss_prod = (s_safe_log - t_safe_log).abs()                                        # [B,V]
    out['distill_prod'] = masked_mean_head(loss_prod.mean(-1))

    # ---------- 3a. 精排 listwise：教师 Top-K 子集上、师生双方都用粗筛融合分数做 listwise ----------
    # （不练精排头：学生精排分数的索引空间是它自己的 top-K，与教师 topk_idx 对不齐，见评审 7.3）
    student_fused = student_pred['coarse_fused_score'].float()          # [B,V]
    t_idx = teacher_cache['topk_idx'].long().to(device)                 # [B,K_cached]
    # topk_refine 是一个**消融旋钮**：落盘 K 固定（256），这里截断出实际参与 listwise 的长度。
    # 不截断的话 opd.topk_refine 就是个 no-op，消融表里那几行会完全一样。
    k_refine = max(1, min(int(opd.topk_refine), t_idx.shape[1]))
    t_idx = t_idx[:, :k_refine]
    t_fused = teacher_cache['coarse'].float().to(device)                # [B,V]
    s_top = torch.gather(student_fused, 1, t_idx) / opd.tau_list        # [B,K]
    t_top = torch.gather(t_fused, 1, t_idx) / opd.tau_list              # [B,K]
    s_log_top = F.log_softmax(s_top, dim=-1)
    t_log_top = F.log_softmax(t_top, dim=-1)
    loss_refine = F.kl_div(s_log_top, t_log_top, reduction='none', log_target=True).sum(-1) * (opd.tau_list ** 2)
    out['distill_refine'] = masked_mean(loss_refine)

    # ---------- 3b. 召回：教师 Top-K' 内部的学生质量 logsumexp ----------
    k_recall = max(1, min(int(opd.topk_recall), t_idx.shape[1]))
    t_idx_recall = t_idx[:, :k_recall]
    # 与 listwise 同温度：融合分数是 log 空间、跨度 ~100 nats，不除 tau 的话 softmax 几乎是
    # one-hot，recall_mass 要么 ≈0 要么 ≈-100，梯度只在极少数样本上有效。
    log_p_s = F.log_softmax(student_fused / opd.tau_list, dim=-1)       # [B,V]
    recall_mass = torch.logsumexp(torch.gather(log_p_s, 1, t_idx_recall), dim=-1)  # [B]
    loss_recall = -recall_mass
    out['distill_recall'] = masked_mean(loss_recall)

    total = (opd.lambda_imi * out['distill_imi']
             + opd.lambda_head * out['distill_head']
             + opd.lambda_refine * out['distill_refine']
             + opd.lambda_recall * out['distill_recall']
             + float(getattr(opd, 'lambda_prod', 0.0)) * out['distill_prod'])
    total = total + 0.0 * student_fused.sum()  # 保持计算图连通（valid 全 0 时 DDP static_graph 不报错）
    out['loss_opd'] = total
    return total, out
