# SPDX-License-Identifier: Apache-2.0
"""效率剖析：参数量 / FLOPs / 推理延迟 / FPS。

论文里"轻量化"的主张必须用这张表支撑：教师（ViT-L）与学生（ResNet34）在**同一份输入、
同一张卡**上的参数量与端到端前向耗时。只跑前向，不需要训练，也不需要 metric cache。

计时对象是 `agent.forward(batch)` 在 eval 模式下的一整趟前向。这条路径在
`aug_meta_arch.forward` 里按 `inference.model` 只跑**一个**模型（教师或学生），
因此教师/学生两侧的口径天然一致。

用法（远程，单卡）：

    python navsim/planning/script/run_efficiency_profile.py \
        agent=gtrs_aug_opd_r34 \
        agent.checkpoint_path=<R34 ckpt> \
        agent.config.inference.model=student \
        ++profile.tag=student_r34 ++profile.out=$NAVSIM_EXP_ROOT/efficiency.csv

    python navsim/planning/script/run_efficiency_profile.py \
        agent=gtrs_aug_drivesuprim_vit \
        agent.checkpoint_path=<ViT-L ckpt> \
        agent.config.inference.model=teacher \
        ++profile.tag=teacher_vitl ++profile.out=$NAVSIM_EXP_ROOT/efficiency.csv

同一个 `profile.out` 会被追加，多次调用即拼成一张完整表。batch_size 用
`dataloader.params.batch_size=N` 覆盖；延迟按**每样本**折算，便于跨 batch 比较。

FLOPs 用 `torch.utils.flop_counter`（torch>=2.1）统计，只覆盖 conv/matmul 等主要算子，
不含 softmax 等逐元素开销——这是通行做法，但论文里必须写明口径，且师生两侧用同一份代码、
同一个 batch 统计。
"""

import csv
import logging
import os
import platform
import statistics
import time
from pathlib import Path
from typing import Dict, Tuple

import hydra
import torch
import torch.distributed as dist
from hydra.utils import instantiate
from nuplan.planning.script.builders.logging_builder import build_logger
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataloader import SceneFilter, SceneLoader
from navsim.planning.training.dataset_aug import DatasetAug as Dataset

logger = logging.getLogger(__name__)

CONFIG_PATH = "config/pdm_scoring"
CONFIG_NAME = "default_run_pdm_score_gpu"

_FIELDS = ("tag", "agent", "infer_slot", "backbone", "ckpt", "device", "gpu_name", "batch_size",
           "n_params_total_M", "n_params_trainable_M", "flops_g_per_sample",
           "latency_ms_per_sample_mean", "latency_ms_per_sample_p50",
           "latency_ms_per_sample_p95", "latency_ms_per_sample_std", "fps")


def _to_device(obj, device):
    """递归搬运 batch（features 是嵌套 dict，targets 里可能混有非张量）。"""
    if torch.is_tensor(obj):
        return obj.to(device, non_blocking=True)
    if isinstance(obj, dict):
        return {k: _to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_to_device(v, device) for v in obj)
    return obj


def _pick_model(agent: AbstractAgent, cfg: DictConfig):
    """返回推理时实际前向的那个 HydraModel 以及槽位名。

    异构 OPD 里 `agent.model.teacher` 只在 EMA 模式下才存在；离线蒸馏时进程内没有教师，
    此时必须落到 student（这正是 OPD 省显存的原因）。
    """
    infer_cfg = getattr(cfg.agent.config, "inference", None)
    infer_model = str(infer_cfg.get("model", "student")) if infer_cfg is not None else "student"
    meta = agent.model
    has_teacher = getattr(meta, "_has_teacher", False)
    if infer_model == "teacher":
        if has_teacher:
            return meta.teacher.model, "teacher"
        logger.warning("inference.model=teacher 但进程内无教师（离线 OPD），参数量按 student 统计")
    return meta.student.model, "student"


def _count_params(model) -> Tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def _batch_size(batch) -> int:
    _features, targets, _tokens = batch
    for v in targets.values():
        if torch.is_tensor(v) and v.dim() > 0:
            return int(v.shape[0])
    return 1


def _count_flops(agent, batch) -> float:
    """每样本 FLOPs（G）。口径：conv/matmul 等主要算子。"""
    try:
        from torch.utils.flop_counter import FlopCounterMode
    except ImportError:
        logger.warning("torch.utils.flop_counter 不可用（需要 torch>=2.1），FLOPs 记为 NaN")
        return float("nan")

    bs = _batch_size(batch)
    try:
        with FlopCounterMode(display=False) as fcm:
            with torch.no_grad():
                agent.forward(batch)
        return fcm.get_total_flops() / max(bs, 1) / 1e9
    except Exception:
        logger.exception("FLOPs 统计失败，记为 NaN")
        return float("nan")


def _time_forward(agent, batch, device, warmup: int, iters: int) -> Tuple[float, float, float, float]:
    """每样本 (mean, p50, p95, std) 毫秒。CUDA event 计时，避免 host 侧抖动。"""
    bs = _batch_size(batch)
    use_cuda = device.type == "cuda"

    def one():
        with torch.no_grad():
            agent.forward(batch)

    for _ in range(warmup):
        one()
    if use_cuda:
        torch.cuda.synchronize()

    samples = []
    for _ in range(iters):
        if use_cuda:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            one()
            end.record()
            torch.cuda.synchronize()
            samples.append(start.elapsed_time(end))
        else:
            t0 = time.perf_counter()
            one()
            samples.append((time.perf_counter() - t0) * 1e3)

    per_sample = sorted(s / bs for s in samples)
    p50 = per_sample[len(per_sample) // 2]
    p95 = per_sample[min(len(per_sample) - 1, int(0.95 * len(per_sample)))]
    std = statistics.pstdev(per_sample) if len(per_sample) > 1 else 0.0
    return statistics.fmean(per_sample), p50, p95, std


def _append_row(out_path: str, row: Dict) -> None:
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    exists = p.is_file() and p.stat().st_size > 0
    with open(p, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(_FIELDS))
        if not exists:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in _FIELDS})


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    build_logger(cfg)

    prof = cfg.get("profile", {})
    tag = str(prof.get("tag", "unnamed"))
    out_path = str(prof.get("out", "efficiency.csv"))
    warmup = int(prof.get("warmup", 10))
    iters = int(prof.get("iters", 50))

    # 计时必须在 eval 路径上：training=True 会多跑一路教师前向 + 损失，测出来不是推理延迟。
    if bool(cfg.agent.config.training):
        logger.warning("agent.config.training=True → 强制置为 False（只测推理路径）")
        cfg.agent.config.training = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dist.is_available() and dist.is_initialized():
        device = torch.device(f"cuda:{dist.get_rank() % torch.cuda.device_count()}")

    agent: AbstractAgent = instantiate(cfg.agent)
    agent.initialize()
    model, slot = _pick_model(agent, cfg)
    agent.model.to(device).eval()

    scene_filter: SceneFilter = instantiate(cfg.train_test_split.scene_filter)
    scene_loader = SceneLoader(
        synthetic_sensor_path=Path(cfg.synthetic_sensor_path),
        original_sensor_path=Path(cfg.original_sensor_path),
        data_path=Path(cfg.navsim_log_path),
        synthetic_scenes_path=Path(cfg.synthetic_scenes_path),
        scene_filter=scene_filter,
        sensor_config=agent.get_sensor_config(),
    )
    dataset = Dataset(
        scene_loader=scene_loader,
        feature_builders=agent.get_feature_builders(),
        target_builders=agent.get_target_builders(),
        cfg=cfg.agent.config,
        cache_path=None,
        force_cache_computation=False,
        append_token_to_batch=True,
    )
    if len(dataset) == 0:
        raise RuntimeError("dataset 为空——检查 OPENSCENE_DATA_ROOT 与 train_test_split")

    dataloader = DataLoader(dataset, **cfg.dataloader.params, shuffle=False)
    batch = _to_device(next(iter(dataloader)), device)

    n_total, n_trainable = _count_params(model)
    flops = _count_flops(agent, batch)
    mean, p50, p95, std = _time_forward(agent, batch, device, warmup, iters)

    row = {
        "tag": tag,
        "agent": str(cfg.agent.get("_target_", "")).rsplit(".", 1)[-1],
        "infer_slot": slot,
        "backbone": str(getattr(cfg.agent.config, "backbone", "")),
        "ckpt": str(cfg.agent.get("checkpoint_path", "") or ""),
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor(),
        "batch_size": _batch_size(batch),
        "n_params_total_M": f"{n_total / 1e6:.2f}",
        "n_params_trainable_M": f"{n_trainable / 1e6:.2f}",
        "flops_g_per_sample": f"{flops:.2f}",
        "latency_ms_per_sample_mean": f"{mean:.2f}",
        "latency_ms_per_sample_p50": f"{p50:.2f}",
        "latency_ms_per_sample_p95": f"{p95:.2f}",
        "latency_ms_per_sample_std": f"{std:.2f}",
        "fps": f"{1000.0 / mean:.1f}" if mean > 0 else "",
    }
    _append_row(out_path, row)

    logger.info(
        f"""
        效率剖析 [{tag}]  模型槽位={slot}  设备={row['gpu_name']}  batch={row['batch_size']}
          参数量   : {row['n_params_total_M']} M  (可训练 {row['n_params_trainable_M']} M)
          FLOPs    : {row['flops_g_per_sample']} G / 样本
          延迟     : {row['latency_ms_per_sample_mean']} ms/样本
                     (p50 {row['latency_ms_per_sample_p50']}, p95 {row['latency_ms_per_sample_p95']},
                      std {row['latency_ms_per_sample_std']})
          吞吐     : {row['fps']} FPS
        已追加到: {out_path}
        """
    )


if __name__ == "__main__":
    main()
