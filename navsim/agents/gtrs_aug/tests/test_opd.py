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
    """构造尖峰分数：idx_sets 里的位置给 +10，其余 -10。

    用于 recall 下界测试——随机 randn 分数经 τ=2 软化后 top-32 的概率质量只有几个百分点，
    拿它断言 "recall≈0" 是不成立的；要断言的是"教师 Top-K' 恰为学生高分集合时 recall→0"。
    """
    s = torch.full((B, V), -10.0)
    for b, idx in enumerate(idx_sets):
        s[b, idx] = 10.0
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


def test_tau_to_inf_coarse_kl_goes_to_zero():
    """τ→∞ 时分布退化成均匀，KL→0。"""
    cfg = _cfg(tau_imi=1e6)
    opd = _import_loss()
    preds = _student_preds()
    preds['coarse_fused_score'] = torch.randn(B, V)
    total, d = opd(preds, _teacher_cache(cfg), cfg)
    assert d['distill_imi'].abs() < 1e-2, d['distill_imi']


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


if __name__ == '__main__':
    import sys
    sys.exit(pytest.main([__file__, '-q']))
