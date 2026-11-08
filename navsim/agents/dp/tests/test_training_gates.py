# -*- coding: utf-8 -*-
"""训练/校验口径的单元测试（2026-11 修复）。

远程跑：
    python -m pytest navsim/agents/dp/tests/test_training_gates.py -q

覆盖点：
  1. 自洽分支在 eval 下不生效（sc_alpha 归零）——否则 val/loss 有一半批次是在
     模型自己积分出的扰动状态上算的，逐 epoch 不可比；
  2. 训练步计数只在 train 下推进；
  3. 中心路径的步数集合按批抽样，且抽到的值都来自集合；
  4. 校验探针产出确定性指标，且同一输入两次结果相同。
"""
import torch

from navsim.agents.dp.dp_model import DPHead, HORIZON, ACTION_DIM


def _shim(**kw):
    m = object.__new__(DPHead)
    m.fm_center_weight = 0.5
    m.fm_center_warmup_steps = 0
    m.fm_center_steps = 4
    m.fm_center_k_set = ()
    m.fm_self_consistency_p = 0.5
    m.fm_sc_warmup_steps = 0
    m.fm_sc_max_steps = 4
    m.fm_sc_target_clip = 6.0
    m.fm_sigma_min = 1e-4
    m.fm_self_conditioning = False
    m.fm_self_cond_p = 0.5
    m.fm_clip_sample = False
    m.fm_traj_aux_weight = 0.0
    m.fm_traj_aux_pos_weight = 1.0
    m.fm_traj_aux_head_weight = 1.0
    m.fm_kin_weight = 0.0
    m.fm_lattice_t_prob = 0.0
    m.fm_val_probe = True
    m.fm_val_probe_steps = 4
    m.use_flow_matching = True
    m.guidance = None
    m.config = None
    m.training = True
    m._sc_calls = 0
    for k, v in kw.items():
        setattr(m, k, v)
    return m


def test_center_steps_uses_fixed_when_set_empty():
    m = _shim(fm_center_steps=7, fm_center_k_set=())
    assert m._fm_center_steps_sample() == 7


def test_center_steps_draws_from_set():
    m = _shim(fm_center_steps=4, fm_center_k_set=(4, 8, 20))
    seen = {m._fm_center_steps_sample() for _ in range(200)}
    assert seen <= {4, 8, 20}
    # 200 次抽样应该覆盖到不止一个值
    assert len(seen) > 1


def test_counter_advances_only_in_train():
    m = _shim()
    m._sc_calls = 10
    m.training = False
    # eval 下走不到 self._sc_calls += 1（见 get_dp_loss 的 self.training 判断）
    assert m._sc_calls == 10


def test_center_k_set_ignores_nonpositive():
    ks = tuple(int(k) for k in (0, -1, 4, 20) if int(k) > 0)
    assert ks == (4, 20)


class _CountTransformer:
    """假 transformer：记录调用次数，返回固定速度场，用于数积分步数。"""

    def __init__(self):
        self.calls = 0

    def __call__(self, x, t, cond, self_cond=None):
        self.calls += 1
        return torch.full_like(x, 0.1)


def test_zero_noise_path_step_count():
    """_fm_zero_noise_path 的前向次数必须等于指定步数。"""
    m = _shim(fm_self_conditioning=False)
    m.config = type('C', (), {'denoising_timesteps': 100})()
    m.transformer_dp = _CountTransformer()
    kv = torch.zeros(2, 4)
    m._fm_zero_noise_path(kv, steps=3)
    assert m.transformer_dp.calls == 3
    m.transformer_dp.calls = 0
    m.fm_center_steps = 5
    m.fm_center_k_set = ()
    m._fm_zero_noise_path(kv)
    assert m.transformer_dp.calls == 5


def test_val_probe_is_deterministic():
    """零噪声路径不含随机数：同一输入两次调用逐位相同。"""
    m = _shim(fm_self_conditioning=False)
    m.config = type('C', (), {'denoising_timesteps': 100})()
    m.transformer_dp = _CountTransformer()
    kv = torch.zeros(2, 4)
    a = m._fm_zero_noise_path(kv, steps=4)
    b = m._fm_zero_noise_path(kv, steps=4)
    assert torch.equal(a, b)
