"""OPD 三路蒸馏的纯张量数值单测（依赖 torch，本机跑不了，在远程服务器上跑）：

    cd sj-1 && python -m pytest navsim/agents/gtrs_aug/tests/test_opd.py -q

覆盖的回归点（对应 OPD 设计评审里的具体风险）：
  1. 融合分数 safe 版不产生 -inf（否则 KL 会 0*(-inf)=NaN）。
  2. reduction 用 'none'+sum+mask 等价 batchmean——若被改成默认 'mean'，量级差 ~8192 倍。
  3. teacher==student 时三路 KL 全为 0；recall 取到 K'/V 的下界。
  4. valid mask 全 0 时 loss==0 且 backward 不报 DDP static_graph 的 unreduced-parameter 错。
"""
import math

import pytest
import torch
import torch.nn.functional as F

torch.manual_seed(0)

V = 8192
B = 4

METRICS = [
    'no_at_fault_collisions', 'drivable_area_compliance', 'time_to_collision_within_bound',
    'ego_progress', 'driving_direction_compliance', 'lane_keeping',
    'traffic_light_compliance', 'history_comfort',
]


def _cfg(**over):
    from navsim.agents.gtrs_aug.hydra_config_aug import HydraConfigAug
    c = HydraConfigAug()
    c.opd.enable = True
    c.opd.teacher_score_dir = '/tmp/opd_fake'
    for k, v in over.items():
        setattr(c.opd, k, v)
    c.__post_init__()
    return c


def _student_preds(with_head_grad=True):
    p = {'imi': torch.randn(B, V, requires_grad=with_head_grad)}
    for m in METRICS:
        p[m] = torch.randn(B, V, requires_grad=with_head_grad)
    return p


def _teacher_cache(cfg, match_student=None):
    """构造一份 batch 级教师缓存张量字典，键集必须与 `_ops_teacher_to_tensors` 的输出一致。

    注意 `coarse` 与 `valid_head` 是必需的：前者是 refine/recall 两路的教师侧分数，
    后者标记 8 头 logits 是否存在（`OPD_STORE_HEADS=0` 时为 0）。
    """
    K = cfg.opd.topk_refine
    cache = {'valid': torch.ones(B), 'valid_head': torch.ones(B), 'imi': torch.randn(B, V)}
    for m in METRICS:
        cache[m] = match_student[m].detach().clone() if match_student is not None else torch.randn(B, V)
    if match_student is not None:
        cache['imi'] = match_student['imi'].detach().clone()
        cache['coarse'] = match_student['coarse_fused_score'].detach().clone()
    else:
        cache['coarse'] = torch.randn(B, V)
    cache['topk_idx'] = torch.randint(0, V, (B, K))
    cache['topk_score'] = torch.randn(B, K)
    return cache


def _peaked(idx_sets):
    """构造尖峰分数：idx_sets 里的位置给 +15，其余 -15。

    用于 recall 下界测试——随机 randn 分数经 τ=2 软化后 top-32 的概率质量只有几个百分点，
    拿它断言 "recall≈0" 是不成立的；要断言的是"教师 Top-K' 恰为学生高分集合时 recall→0"。
    注意 τ=2 会把峰谷差减半：±10 时理论下界 = -log(1-255·e^{-10}) ≈ 0.0115，
    会顶穿 1e-2 的断言阈值；±15 时下界 ≈ 255·e^{-15} ≈ 8e-5，留出足够裕量。
    """
    s = torch.full((B, V), -15.0)
    for b, idx in enumerate(idx_sets):
        s[b, idx] = 15.0
    return s


def _import_loss():
    from navsim.agents.gtrs_aug.hydra_loss_fn_aug import opd_distill_loss
    return opd_distill_loss


def test_shapes_and_finite():
    cfg = _cfg()
    opd = _import_loss()
    preds = _student_preds()
    preds['coarse_fused_score'] = torch.randn(B, V, requires_grad=True)
    total, details = opd(preds, _teacher_cache(cfg), cfg)
    assert total.ndim == 0
    for k, v in details.items():
        assert torch.isfinite(v).all(), f'{k} not finite'


def test_no_inf_from_safe_fused_score():
    """制造会下溢的极端 logit，断言 safe 版不给 -inf（不走 KL→NaN）。"""
    from navsim.agents.gtrs_aug.hydra_model import fused_coarse_score
    cfg = _cfg()
    head_out = {m: torch.randn(1, V) for m in METRICS}
    head_out['imi'] = torch.full((1, V), -1e4)          # softmax(-1e4) -> 0 -> log -> -inf (非 safe 版)
    head_out['imi'][0, :10] = 0.0
    head_out['time_to_collision_within_bound'][0] = -1e4
    safe = fused_coarse_score(head_out, cfg, safe=True)
    assert torch.isfinite(safe).all(), 'safe fused score still produced non-finite'


def test_head_soft_bce_sanity():
    """逐头 soft-BCE 的最小性：teacher logit == student logit 时该头贡献最小；
    teacher logit 反向时贡献严格更大。"""
    cfg = _cfg()
    opd = _import_loss()
    preds = _student_preds()
    preds['coarse_fused_score'] = torch.randn(B, V)
    cache = _teacher_cache(cfg, match_student=preds)
    _, d_same = opd(preds, cache, cfg)
    cache2 = _teacher_cache(cfg)
    for m in METRICS:
        cache2[m] = -cache2[m]          # 反向
    _, d_opp = opd(preds, cache2, cfg)
    assert d_opp['distill_head'] > d_same['distill_head'], (d_same['distill_head'], d_opp['distill_head'])


def test_teacher_equals_student_zero_kl():
    """teacher 与 student 完全一致时，imi / refine 两路 KL 应≈0。"""
    cfg = _cfg()
    opd = _import_loss()
    preds = _student_preds()
    preds['coarse_fused_score'] = torch.randn(B, V, requires_grad=False)
    cache = _teacher_cache(cfg, match_student=preds)
    # 让 refine 的 top-K 恰好是学生分数的前 K 名（教师侧 coarse 已与学生相同，KL→0）
    s = preds['coarse_fused_score'].float()
    topk = torch.topk(s, k=cfg.opd.topk_refine, dim=1)
    cache['topk_idx'] = topk.indices
    cache['topk_score'] = topk.values
    total, d = opd(preds, cache, cfg)
    assert d['distill_imi'].abs() < 1e-3, d['distill_imi']
    # 逐头 soft-BCE 在 teacher==student 时取最小值（非 0），但应显著小于随机 logits 下的值
    rand_cache = _teacher_cache(cfg)
    _, d_rand = opd(preds, rand_cache, cfg)
    assert d['distill_head'] < d_rand['distill_head'], (d['distill_head'], d_rand['distill_head'])
    assert d['distill_refine'].abs() < 1e-2, d['distill_refine']


def test_recall_goes_to_zero_when_student_mass_on_teacher_topk():
    """学生把概率质量全压在教师 Top-K' 上时 recall→0（这才是该项的最小值条件）。"""
    cfg = _cfg()
    opd = _import_loss()
    preds = _student_preds()
    kr = cfg.opd.topk_recall
    idx_sets = [torch.arange(kr) + b for b in range(B)]
    preds['coarse_fused_score'] = _peaked(idx_sets)
    cache = _teacher_cache(cfg)
    # 教师 topk_idx 的前 K' 就是学生的尖峰位置
    t_idx = torch.randint(0, V, (B, cfg.opd.topk_refine))
    for b in range(B):
        t_idx[b, :kr] = idx_sets[b]
    cache['topk_idx'] = t_idx
    _, d = opd(preds, cache, cfg)
    assert d['distill_recall'].abs() < 1e-2, d['distill_recall']


def test_missing_heads_zeroes_head_path_without_keyerror():
    """OPD_STORE_HEADS=0 的缓存：8 头为零占位且 valid_head=0 → head 路为 0，且不 KeyError。"""
    cfg = _cfg()
    opd = _import_loss()
    preds = _student_preds()
    preds['coarse_fused_score'] = torch.randn(B, V, requires_grad=True)
    cache = _teacher_cache(cfg)
    cache['valid_head'] = torch.zeros(B)
    for m in METRICS:
        cache[m] = torch.zeros(B, V)        # `_ops_teacher_to_tensors` 的零占位
    total, d = opd(preds, cache, cfg)
    assert d['distill_head'].abs() < 1e-8, d['distill_head']
    assert d['distill_imi'] > 0, 'imi 路不应被 head 缺失影响'
    total.backward()
    assert preds['coarse_fused_score'].grad is not None


def test_topk_refine_is_not_a_noop():
    """topk_refine 是消融旋钮：截断不同长度必须给出不同的 listwise 损失。"""
    opd = _import_loss()
    cfg_big = _cfg(topk_refine=256)
    preds = _student_preds(with_head_grad=False)
    preds['coarse_fused_score'] = torch.randn(B, V)
    cache = _teacher_cache(cfg_big)
    _, d_big = opd(preds, cache, cfg_big)
    cfg_small = _cfg(topk_refine=16)
    # 同一份缓存（落盘 K=256）喂给 topk_refine=16 的配置，损失端应截断到 16
    _, d_small = opd(preds, cache, cfg_small)
    assert not torch.allclose(d_big['distill_refine'], d_small['distill_refine']), (
        'topk_refine 没有生效，listwise 仍在全部 256 上算')


def test_reduction_is_batchmean_scale_not_mean():
    """kl_div 必须按样本（sum over V）。若被误用默认 'mean'，梯度会小 ~V 倍——量级守卫。"""
    cfg = _cfg()
    s = torch.randn(B, V)
    t = torch.randn(B, V)
    s_log = F.log_softmax(s / cfg.opd.tau_imi, -1)
    t_log = F.log_softmax(t / cfg.opd.tau_imi, -1)
    per_sample = F.kl_div(s_log, t_log, reduction='none', log_target=True).sum(-1)
    wrong_mean = F.kl_div(s_log, t_log, reduction='mean', log_target=True)
    ratio = (per_sample.mean() / (wrong_mean + 1e-12)).abs()
    assert 0.5 * V < ratio < 2.0 * V, f'mean-vs-batchmean gap should be ~V, got {ratio}'


def test_valid_mask_zero_gives_zero_and_backprop_ok():
    """valid 全 0：loss==0，且 backward 后学生各头 grad 非 None（static_graph 不变 used-set）。"""
    cfg = _cfg()
    opd = _import_loss()
    preds = _student_preds()
    preds['coarse_fused_score'] = torch.randn(B, V, requires_grad=True)
    cache = _teacher_cache(cfg)
    cache['valid'] = torch.zeros(B)
    total, d = opd(preds, cache, cfg)
    assert total.abs() < 1e-8, total
    total.backward()
    for name, p in preds.items():
        if p.requires_grad:
            assert p.grad is not None, f'{name} has no grad after all-invalid backward'


def test_tau_large_kl_vanishes_but_tau2_scaled_hits_fisher_limit():
    """τ→∞ 时分布退化成均匀：未乘 τ² 的 KL→0，但 τ²·KL 收敛到 Fisher（梯度匹配）极限
    ½·Var_v(s−t)——这正是 KD 要乘 τ² 的原因（τ 再大监督信号也不消失）。

    实现内部 `.float()` 强制 fp32，τ 不能取太大：logit 差 ~1/τ 被 fp32 舍入噪声
    （~1e-6 量级）淹没后 ×τ² 会放大成垃圾（τ=1e4 已偏 10 倍，τ=1e6 甚至算出负"KL"）。
    实测 τ∈[50,200] 时 fp32 与 Fisher 极限吻合到 <1%，故取 τ=100。
    """
    tau = 100.0
    cfg = _cfg(tau_imi=tau)
    opd = _import_loss()
    preds = _student_preds()
    preds['coarse_fused_score'] = torch.randn(B, V)
    cache = _teacher_cache(cfg)
    total, d = opd(preds, cache, cfg)
    # 1) 未缩放的 KL 已退化到 ~1e-4（τ=1 时是 O(1)），确证分布→均匀
    assert (d['distill_imi'].abs() / tau ** 2) < 5e-4, d['distill_imi']
    # 2) τ² 缩放后的值 ≈ Fisher 极限 ½·E[Var_v(s−t)]（s,t~randn ⇒ 期望 ≈1，不是 0）
    delta = preds['imi'] - cache['imi']
    delta = delta - delta.mean(dim=-1, keepdim=True)
    fisher = 0.5 * delta.pow(2).mean(dim=-1).mean()  # ½·Var_v，batch 取均值
    assert torch.allclose(d['distill_imi'], fisher, rtol=0.05), (d['distill_imi'], fisher)


def test_recall_floor_at_uniform():
    """学生分布均匀时，recall 原值 loss = -log(K'/V)（details 里返回的是未乘 λ 的原值）。

    注意 topk_idx 是 randint 采的，可能撞重复索引；前 K' 去重后实际集合大小 ≤ K'，
    所以这里用 topk_idx[:, :K'] 的唯一元素数算期望，而不是直接用 K'。
    """
    cfg = _cfg()
    opd = _import_loss()
    preds = _student_preds()
    preds['coarse_fused_score'] = torch.zeros(B, V)     # 均匀（除以 τ 仍均匀）
    cache = _teacher_cache(cfg)
    kr = cfg.opd.topk_recall
    cache['topk_idx'][:, :kr] = torch.stack([torch.arange(kr) + b * kr for b in range(B)])
    expected = -math.log(kr / V)
    total, d = opd(preds, cache, cfg)
    assert abs(d['distill_recall'].item() - expected) < 1e-3, (d['distill_recall'].item(), expected)


SAFE_HEADS = ['no_at_fault_collisions', 'drivable_area_compliance',
              'driving_direction_compliance', 'traffic_light_compliance']


def test_distill_prod_zero_when_heads_match_and_positive_when_one_safe_head_flips():
    """式 (4-23)：师生逐头相同 → 乘积一致性损失为 0；只翻一个安全头 → 严格为正。"""
    opd = _import_loss()
    cfg = _cfg(lambda_prod=1.0)
    preds = _student_preds(with_head_grad=False)
    preds['coarse_fused_score'] = torch.randn(B, V)
    cache = _teacher_cache(cfg, match_student=preds)
    _, d_match = opd(preds, cache, cfg)
    assert d_match['distill_prod'].abs() < 1e-6, d_match['distill_prod']

    # 只把「闯红灯」这一个安全头的教师 logit 大幅拉低，其余全同 → 乘积失配
    cache['traffic_light_compliance'] = cache['traffic_light_compliance'].clone()
    cache['traffic_light_compliance'][:, :] -= 8.0
    _, d_flip = opd(preds, cache, cfg)
    assert d_flip['distill_prod'] > 1e-3, d_flip['distill_prod']


def test_lambda_prod_zero_is_bitwise_noop_on_total():
    """默认路径回归守卫：λ_prod=0 时总损失与不含该项逐位相同（新增项默认不改变训练行为）。"""
    opd = _import_loss()
    cfg = _cfg(lambda_prod=0.0)
    preds = _student_preds(with_head_grad=False)
    preds['coarse_fused_score'] = torch.randn(B, V)
    cache = _teacher_cache(cfg)
    total, d = opd(preds, cache, cfg)
    manual = (cfg.opd.lambda_imi * d['distill_imi']
              + cfg.opd.lambda_head * d['distill_head']
              + cfg.opd.lambda_refine * d['distill_refine']
              + cfg.opd.lambda_recall * d['distill_recall'])
    assert torch.equal(total, manual), (total.item(), manual.item())


def test_head_scope_safe_supervises_four_heads_only():
    """head_scope='safe' 只蒸四项乘性安全头；'all' 覆盖全部八项。"""
    opd = _import_loss()
    preds = _student_preds(with_head_grad=False)
    preds['coarse_fused_score'] = torch.randn(B, V)

    cfg_all = _cfg(head_scope='all')
    _, d_all = opd(preds, _teacher_cache(cfg_all), cfg_all)
    cfg_safe = _cfg(head_scope='safe')
    cache_safe = _teacher_cache(cfg_safe)
    _, d_safe = opd(preds, cache_safe, cfg_safe)

    # safe 作用域 == 手算的四项加权和
    w = cfg_safe.trajectory_pdm_weight
    manual = torch.zeros(B)
    for m in SAFE_HEADS:
        t_logit = cache_safe[m].float() / cfg_safe.opd.tau_head
        s_logit = preds[m].float() / cfg_safe.opd.tau_head
        bce = F.binary_cross_entropy_with_logits(s_logit, torch.sigmoid(t_logit), reduction='none')
        manual = manual + bce.mean(-1) * (cfg_safe.opd.tau_head ** 2) * float(w.get(m, 1.0))
    assert torch.allclose(d_safe['distill_head'], manual.mean(), rtol=1e-5), (
        d_safe['distill_head'], manual.mean())
    # 作用域不同 → 值不同（八项里多出的四项贡献非零）
    assert not torch.allclose(d_all['distill_head'], d_safe['distill_head'], rtol=1e-6)


def test_dual_stream_fusion_is_log_epdms_and_penalizes_zero_safe_head():
    """创新点 1 的机理断言（纯数值，不依赖教师）：

    - 双流融合 = log(安全项乘积 · 舒适项加权平均 / W) + β·log p^imi，与手算一致；
    - 构造一个安全头归零（σ→0）的候选，其双流分数应显著低于安全头全 1 的候选——
      这是「安全是乘性硬约束」的核心，基线式 (4-5) 的加性形式做不到。
    """
    from navsim.agents.gtrs_aug.hydra_model import fused_coarse_score
    cfg = _cfg(dual_stream_score=True)
    cfg.opd.beta_imi = 0.0                      # 隔离 imi 先验，只看八项
    head_out = {m: torch.full((1, V), 6.0) for m in METRICS}   # σ(6)≈0.9975，接近 1
    head_out['imi'] = torch.zeros(1, V)

    dual = fused_coarse_score(head_out, cfg, safe=True, dual_stream=True)

    # 手算：Σ_mul logσ + log((Σ_add w·σ)/14)
    w_add = [5.0, 5.0, 2.0, 2.0]
    add_heads = ['time_to_collision_within_bound', 'ego_progress',
                 'lane_keeping', 'history_comfort']
    safe_log = sum(torch.logsigmoid(head_out[m]) for m in SAFE_HEADS)
    comfort = sum(wi * head_out[m].sigmoid() for wi, m in zip(w_add, add_heads)) / 14.0
    manual = safe_log + comfort.clamp_min(1e-6).log()
    assert torch.allclose(dual, manual, atol=1e-5), (dual[0, :3], manual[0, :3])

    # 安全头归零的候选：双流分数必须显著低于安全头全 1 的候选
    bad = {k: v.clone() for k, v in head_out.items()}
    bad['no_at_fault_collisions'] = torch.full((1, V), -30.0)
    dual_bad = fused_coarse_score(bad, cfg, safe=True, dual_stream=True)
    assert (dual[0, 0] - dual_bad[0, 0]).item() > 10.0, (dual[0, 0].item(), dual_bad[0, 0].item())


def test_dual_stream_default_off_matches_baseline_branch():
    """dual_stream=False 时融合分数与既有 safe 分支逐位相同（默认路径零回归）。"""
    from navsim.agents.gtrs_aug.hydra_model import fused_coarse_score
    cfg = _cfg(dual_stream_score=False)
    head_out = {m: torch.randn(2, V) for m in METRICS}
    head_out['imi'] = torch.randn(2, V)
    a = fused_coarse_score(head_out, cfg, safe=True)
    b = fused_coarse_score(head_out, cfg, safe=True, dual_stream=False)
    assert torch.equal(a, b)


def test_safety_gate_rejects_unsafe_candidate_that_soft_score_prefers():
    """安全门后处理：软分数会选中的「高舒适 + 预测安全 0.9」候选必须被门排除。

    这是 zero_pct 的直接来源——EPDMS 的安全项是四项二值指标的乘积，软分数下
    一个预测安全 0.9 的候选仍可能靠舒适度胜出，而它有一成概率整分归零。
    """
    from navsim.agents.gtrs_aug.hydra_model import safety_gate_scores

    # 2 个候选：cand0 安全但舒适低，cand1 舒适高但 nc 预测很低
    head_out = {m: torch.full((1, 2), 6.0) for m in METRICS}      # σ(6)≈0.9975
    head_out['history_comfort'] = torch.tensor([[0.0, 6.0]])      # cand1 舒适更高
    head_out['no_at_fault_collisions'] = torch.tensor([[6.0, -4.0]])  # cand1 碰撞风险高
    scores = torch.tensor([[0.0, 1.0]])                           # 软分数偏好 cand1

    assert scores.argmax(1).item() == 1, '前置：软分数确实选中了不安全的 cand1'
    gated = safety_gate_scores(head_out, scores, 0.8)
    assert gated.argmax(1).item() == 0, '安全门没有排除不安全的候选'
    assert torch.isneginf(gated[0, 1]), '被排除的候选应置 -inf 而不是改数值'

    # 关闭时逐位不变
    assert torch.equal(safety_gate_scores(head_out, scores, 0.0), scores)


def test_safety_gate_degrades_gracefully_when_all_candidates_unsafe():
    """难场景里全体候选都违规时，门按场景内相对阈值自动放宽，不会出现空候选集。"""
    from navsim.agents.gtrs_aug.hydra_model import safety_gate_scores

    head_out = {m: torch.full((1, 3), 6.0) for m in METRICS}
    head_out['no_at_fault_collisions'] = torch.tensor([[-6.0, -4.0, -8.0]])   # 全部违规
    scores = torch.tensor([[1.0, 3.0, 2.0]])
    gated = safety_gate_scores(head_out, scores, 0.8)
    assert torch.isfinite(gated).any(), '全体违规时不应把候选集清空'
    assert gated.argmax(1).item() == 1, '应保留相对最安全的那个（nc 最高的 cand1）'


if __name__ == '__main__':
    import sys
    sys.exit(pytest.main([__file__, '-q']))
