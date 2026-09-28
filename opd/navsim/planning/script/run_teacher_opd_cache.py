# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# 教师离线打分缓存（OPD 蒸馏）。
#
# 与 run_pdm_score_gpu_v2_aug.py 的三处关键区别：
#   1. 流式 per-token 写盘，不做 all_gather_object；103,288 token 全量聚合必 OOM。
#   2. 存原始 logits（未 .sigmoid()/.log()）而非 log 概率——loss 端要能重选 τ/λ、重算 top-K、
#      且 KL 走 log-space。refinement 子集合的 top-K 也存原始融合分（safe 版）。
#   3. 不与 metric cache 求交集（教师打分只需要 scene + sensor，不需要 metric cache）；
#      按 {token}.pkl 存在与否断点续跑。
#
# 用法（远程，3×3090）：
#   NPROC=3 ./scripts/opd/training/cache_teacher.sh
#   环境变量：TEACHER_AGENT=gtrs_aug_drivesuprim_vit TEACHER_CKPT=<ViT-L ckpt> OPD_STORE_HEADS=0|1

import hashlib
import logging
import os
import pickle
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import hydra
import numpy as np
import pytorch_lightning as pl
import torch
import torch.distributed as dist
from hydra.utils import instantiate
from nuplan.planning.script.builders.logging_builder import build_logger
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import SensorConfig
from navsim.common.dataloader import SceneFilter, SceneLoader
from navsim.planning.training.agent_lightning_module_aug import AgentLightningModuleAug
from navsim.planning.training.dataset_aug import DatasetAug as Dataset

logger = logging.getLogger(__name__)

CONFIG_PATH = "config/pdm_scoring"
CONFIG_NAME = "default_run_pdm_score_gpu"

_METRICS = ('no_at_fault_collisions', 'drivable_area_compliance', 'time_to_collision_within_bound',
            'ego_progress', 'driving_direction_compliance', 'lane_keeping',
            'traffic_light_compliance', 'history_comfort')


def _sha1_prefix(path, nbytes=4096):
    try:
        with open(path, 'rb') as f:
            buf = f.read(nbytes)
        size = os.path.getsize(path)
    except OSError:
        return ""
    return hashlib.sha1(buf + str(size).encode()).hexdigest()


def _atomic_dump(obj, final_path: str, rank: int) -> None:
    tmp = f"{final_path}.tmp.{rank}.{os.getpid()}.{datetime.now().microsecond}"
    with open(tmp, 'wb') as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, final_path)  # POSIX 原子


class OPDCaptureModule(AgentLightningModuleAug):
    """只做前向、per-token 原子写盘；返回 None（return_predictions=False 忽略返回值）。"""

    def __init__(self, cfg, agent, out_dir: str, vocab_sha1: str, offline_aug_sha1: str, store_heads: bool,
                 safe_fused_score: bool, teacher_ckpt: str = '', view_idx: int = 0):
        super().__init__(cfg=cfg, agent=agent)
        self._teacher_ckpt = teacher_ckpt
        self._out_dir = out_dir
        self._vocab_sha1 = vocab_sha1
        self._offline_aug_sha1 = offline_aug_sha1
        self._store_heads = store_heads
        self._safe_fused_score = safe_fused_score
        self._view_idx = view_idx

    def predict_step(self, batch: Tuple[Dict, Dict, list], batch_idx: int):
        features, targets, tokens = batch
        self.agent.eval()
        with torch.no_grad():
            predictions, _, _ = self.agent.forward(batch)

        # 取原始 logits（未 sigmoid/softmax/log）。coarse_fused_score 是 safe 版融合分数。
        imi = predictions["imi"].float().cpu().numpy()                      # [B, V]  logits
        coarse = predictions["coarse_fused_score"].float().cpu().numpy()    # [B, V]  safe 融合分
        # 取最后一级精排（num_refinement_stage 可能 >1；缓存脚本默认 1，此时 [-1] == [0]）
        refine = predictions['refinement'][-1]
        topk_idx = refine['indices_absolute'].long().cpu().numpy()          # [B, K]
        topk_score = refine['scores'].float().cpu().numpy()                 # [B, K]  精排分（原始）
        # 词表在师生两端同一个 8192.npy；取 coarse 维数即可断言长度对得上
        assert imi.shape[1] == coarse.shape[1] == self._cfg.vocab_size, (
            f"vocab mismatch: imi {imi.shape}, coarse {coarse.shape}, cfg.vocab_size {self._cfg.vocab_size}")
        assert np.isfinite(coarse).all(), (
            "coarse_fused_score 含 inf/nan——教师侧必须 agent.config.opd.safe_fused_score=true，"
            "否则非 safe 分支的 softmax().log() 会下溢成 -inf，训练端 gather 后变 NaN")
        heads = {m: predictions[m].float().cpu().numpy() for m in _METRICS} if self._store_heads else None

        rank = dist.get_rank() if (dist.is_available() and dist.is_initialized()) else 0
        for i, token in enumerate(tokens):
            final_path = os.path.join(self._out_dir, f"{token}.pkl")
            if os.path.exists(final_path):
                continue  # 断点续跑
            payload = {
                'imi': imi[i].astype(np.float32),
                'coarse': coarse[i].astype(np.float32),
                'topk_idx': topk_idx[i].astype(np.int32),
                'topk_score': topk_score[i].astype(np.float32),
                'vocab_sha1': self._vocab_sha1,
                'vocab_size': imi.shape[1],
                'offline_aug_file_sha1': self._offline_aug_sha1,
                # self._cfg 是 cfg.agent.config，里面没有 checkpoint_path（它在 cfg.agent 上），
                # 原来的 getattr(self._cfg,'checkpoint_path','') 永远返回 ''——由 main 显式传入
                'teacher_ckpt': self._teacher_ckpt,
                # 训练端据此断言师生两侧用的是同一条融合分数公式
                'safe_fused_score': bool(self._safe_fused_score),
                'store_heads': bool(self._store_heads),
                'view_idx': int(self._view_idx),
            }
            if self._store_heads:
                # 训练端（`_ops_teacher_to_tensors` / `opd_distill_loss`）按 **8 个顶层 per-head 键**
                # 读取，不是一个堆叠数组——这里必须逐头写 top-level key，否则首个 batch 就 KeyError。
                for m in _METRICS:
                    payload[m] = heads[m][i].astype(np.float16)
            _atomic_dump(payload, final_path, rank)
        return None


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    build_logger(cfg)

    out_dir = os.environ["OPD_TEACHER_SCORE_DIR"]
    # 默认 **开启**：训练端 opd.lambda_head 默认 1.0、opd.store_heads 默认 True，
    # 缓存不含 8 头时那一路会被整体置零（不报错，但少一路监督）。两边默认值必须一致。
    store_heads = os.environ.get("OPD_STORE_HEADS", "1") == "1"
    view_idx = int(os.environ.get("OPD_VIEW_IDX", "0"))

    scene_filter = instantiate(cfg.train_test_split.scene_filter)
    scene_loader = SceneLoader(
        synthetic_sensor_path=None,
        original_sensor_path=None,
        data_path=Path(cfg.navsim_log_path),
        synthetic_scenes_path=Path(cfg.synthetic_scenes_path),
        scene_filter=scene_filter,
        sensor_config=SensorConfig.build_no_sensors(),
    )
    # 教师打分不需要 metric cache；直接用 SceneLoader 的全部 token。
    tokens_to_cache = list(scene_loader.tokens)
    logger.info(f"Caching teacher scores for {len(tokens_to_cache)} tokens -> {out_dir}")

    agent: AbstractAgent = instantiate(cfg.agent)
    agent.initialize()

    scene_loader_inference = SceneLoader(
        synthetic_sensor_path=Path(cfg.synthetic_sensor_path),
        original_sensor_path=Path(cfg.original_sensor_path),
        data_path=Path(cfg.navsim_log_path),
        synthetic_scenes_path=Path(cfg.synthetic_scenes_path),
        scene_filter=scene_filter,
        sensor_config=agent.get_sensor_config(),
    )
    dataset = Dataset(
        scene_loader=scene_loader_inference,
        feature_builders=agent.get_feature_builders(),
        target_builders=agent.get_target_builders(),
        cfg=cfg.agent.config,
        cache_path=None,
        force_cache_computation=False,
        append_token_to_batch=True,
    )
    dataloader = DataLoader(dataset, **cfg.dataloader.params, shuffle=False)

    vocab_sha1 = _sha1_prefix(cfg.agent.config.vocab_path)
    offline_aug_sha1 = _sha1_prefix(cfg.agent.config.ego_perturb.offline_aug_file)
    # 教师侧与训练端必须同一条融合分数公式；非 safe 分支会下溢出 -inf，训练端 gather 后变 NaN。
    safe_fused_score = bool(cfg.agent.config.opd.safe_fused_score)
    assert safe_fused_score, (
        "教师缓存必须带 agent.config.opd.safe_fused_score=true（见 cache_teacher.sh）；"
        "否则落盘的 coarse 含 -inf，训练端 log_softmax 后整 batch NaN")
    logger.info(f"vocab_sha1={vocab_sha1} offline_aug_sha1={offline_aug_sha1} "
                f"store_heads={store_heads} safe_fused_score={safe_fused_score} view_idx={view_idx}")

    module = OPDCaptureModule(cfg=cfg.agent.config, agent=agent, out_dir=out_dir,
                              vocab_sha1=vocab_sha1, offline_aug_sha1=offline_aug_sha1,
                              store_heads=store_heads, safe_fused_score=safe_fused_score,
                              teacher_ckpt=str(cfg.agent.get('checkpoint_path', '') or ''),
                              view_idx=view_idx)

    # 每个 rank 先各建一次目录（exist_ok 吞并发 FileExistsError），再 barrier 对齐后开跑
    os.makedirs(out_dir, exist_ok=True)
    if dist.is_available() and dist.is_initialized():
        dist.barrier()

    trainer = pl.Trainer(**cfg.trainer.params, callbacks=agent.get_training_callbacks())
    trainer.predict(module, dataloader, return_predictions=False)

    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        if dist.get_rank() == 0:
            done = sum(1 for f in os.listdir(out_dir) if f.endswith('.pkl') and '.tmp.' not in f)
            logger.info(f"Teacher cache complete: {done}/{len(tokens_to_cache)} tokens in {out_dir}")
    else:
        done = sum(1 for f in os.listdir(out_dir) if f.endswith('.pkl') and '.tmp.' not in f)
        logger.info(f"Teacher cache complete: {done}/{len(tokens_to_cache)} tokens in {out_dir}")


if __name__ == "__main__":
    main()
