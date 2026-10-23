# -*- coding: utf-8 -*-
"""Phase 1（中心一致性训练）的单元测试。

远程跑：
    python -m pytest navsim/agents/dp/tests/test_center_consistency.py -q

覆盖点：
  1. 权重为 0 时恒为 0（get_dp_loss 里整块跳过，训练行为与改动前一致）；
  2. warmup 线性爬升，到点后保持 fm_center_weight；
  3. 中心轨迹与真值完全一致时，中心损失为 0；
  4. 中心轨迹偏离真值时，损失为正（漂移项生效）。
"""
import torch

from navsim.agents.dp.dp_model import DPHead


def _shim(weight=0.5, warmup=2000, calls=0):
    """绕过 __init__（不建 transformer），只测纯函数逻辑。"""
    m = object.__new__(DPHead)
    m.fm_center_weight = float(weight)
    m.fm_center_warmup_steps = int(warmup)
    m._sc_calls = int(calls)
    m.fm_traj_aux_pos_weight = 1.0
    m.fm_traj_aux_head_weight = 1.0
    m.fm_kin_weight = 0.5
    return m


def _gt(batch=2, horizon=8):
    x1 = torch.zeros(batch, horizon, 4)
    x1[..., 0] = torch.linspace(0.0, 0.5, horizon)
    x1[..., 1] = torch.linspace(0.0, 0.1, horizon)
    x1[..., 2] = 1.0       # cos
    x1[..., 3] = 0.0       # sin
    return x1


def test_zero_weight_is_noop():
    m = _shim(weight=0.0, calls=12345)
    assert m._center_alpha() == 0.0


def test_center_alpha_warmup():
    m = _shim(weight=0.5, warmup=100, calls=0)
    assert m._center_alpha() == 0.0
    m._sc_calls = 50
    assert abs(m._center_alpha() - 0.25) < 1e-9
    m._sc_calls = 100
    assert abs(m._center_alpha() - 0.5) < 1e-9
    m._sc_calls = 500
    assert abs(m._center_alpha() - 0.5) < 1e-9


def test_center_loss_zero_when_matches_gt():
    m = _shim()
    x1 = _gt()
    loss, extras = m._center_consistency_loss(x1, x1)
    assert loss.item() == 0.0
    assert extras['fm_center_pos_loss'].item() == 0.0
    assert extras['fm_center_head_loss'].item() == 0.0


def test_center_loss_positive_when_shifted():
    m = _shim()
    x1 = _gt()
    x1_center = x1.clone()
    x1_center[..., 0] += 0.2
    loss, _ = m._center_consistency_loss(x1_center, x1)
    assert loss.item() > 0.0
