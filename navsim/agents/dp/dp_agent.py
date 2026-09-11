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

import logging
import os
from typing import Any, Union
from typing import Dict
from typing import List

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler, OneCycleLR

from navsim.agents.abstract_agent import AbstractAgent
from navsim.agents.dp.dp_config import DPConfig
from navsim.agents.dp.dp_model import DPModel
from navsim.agents.gtrs_dense.hydra_features import HydraFeatureBuilder, HydraTargetBuilder
from navsim.common.dataclasses import SensorConfig
from navsim.planning.training.abstract_feature_target_builder import (
    AbstractFeatureBuilder,
    AbstractTargetBuilder,
)

logger = logging.getLogger(__name__)


def dp_loss_bev(
        targets: Dict[str, torch.Tensor], predictions: Dict[str, torch.Tensor],
        config: DPConfig, traj_head
):
    # B, 8 (4 secs, 0.5Hz), 3
    target_traj = targets["trajectory"]
    dp_loss, dp_extras = traj_head.get_dp_loss(predictions['env_kv'], target_traj.float())
    bev_semantic_loss = F.cross_entropy(predictions["bev_semantic_map"], targets["bev_semantic_map"].long())
    dp_loss = dp_loss * config.dp_loss_weight
    bev_semantic_loss = bev_semantic_loss * config.bev_loss_weight
    loss = (
            dp_loss +
            bev_semantic_loss
    )
    loss_dict = {
        'dp_loss': dp_loss,
        'bev_semantic_loss': bev_semantic_loss
    }
    # detached FM diagnostics (fm_aux_* / fm_x1_ade / fm_x1_fde); empty dict for DDPM
    loss_dict.update(dp_extras)
    return loss, loss_dict


class DPAgent(AbstractAgent):
    def __init__(
            self,
            config: DPConfig,
            lr: float,
            checkpoint_path: str = None
    ):
        super().__init__(
            trajectory_sampling=config.trajectory_sampling
        )
        self._config = config
        self._lr = lr
        self._checkpoint_path = checkpoint_path
        self.model = DPModel(config)
        self.backbone_wd = config.backbone_wd
        self.scheduler = config.scheduler

    def name(self) -> str:
        """Inherited, see superclass."""
        return self.__class__.__name__

    def _apply_freeze_except_traj_head(self) -> None:
        if not getattr(self._config, "freeze_except_traj_head", False):
            return
        n_train, n_freeze = 0, 0
        for name, param in self.model.named_parameters():
            if name.startswith("_trajectory_head"):
                param.requires_grad = True
                n_train += param.numel()
            else:
                param.requires_grad = False
                n_freeze += param.numel()
        logger.info(
            f"freeze_except_traj_head: trainable={n_train:,} frozen={n_freeze:,} "
            f"flow_matching={getattr(self._config, 'use_flow_matching', False)}"
        )

    def _reinit_trajectory_head(self) -> None:
        """Randomly re-initialize DPHead transformer (FM trains a fresh decoder)."""
        traj_head = self.model._trajectory_head
        transformer = traj_head.transformer_dp
        transformer.apply(transformer._init_weights)
        n_params = sum(p.numel() for p in traj_head.parameters())
        logger.info(f"Re-initialized _trajectory_head ({n_params:,} params) for Flow Matching")

    def initialize(self) -> None:
        """Inherited, see superclass."""
        state_dict: Dict[str, Any] = torch.load(self._checkpoint_path, map_location=torch.device("cpu"))[
            "state_dict"]
        cleaned = {k.replace("agent.", ""): v for k, v in state_dict.items()}
        reinit_traj_head = bool(getattr(self._config, "reinit_traj_head", False))
        if reinit_traj_head:
            cleaned = {k: v for k, v in cleaned.items() if "_trajectory_head" not in k}
            load_result = self.load_state_dict(cleaned, strict=False)
            # strict=False otherwise hides *any* key mismatch: a renamed backbone key would
            # leave the backbone randomly initialized while training starts and loss still
            # "goes down". Assert the missing set is exactly the head we deliberately dropped.
            unexpected = list(load_result.unexpected_keys)
            stray_missing = [k for k in load_result.missing_keys if "_trajectory_head" not in k]
            assert not stray_missing, (
                f"reinit_traj_head=True dropped only _trajectory_head, but these non-head keys "
                f"are also missing from the checkpoint (they would stay randomly initialized): "
                f"{stray_missing[:20]}"
            )
            assert not unexpected, (
                f"checkpoint has keys the model does not define: {unexpected[:20]}"
            )
            logger.info(
                f"Loaded DP checkpoint from {self._checkpoint_path} "
                f"(skipped _trajectory_head; missing={len(load_result.missing_keys)}, "
                f"unexpected={len(load_result.unexpected_keys)})"
            )
            self._reinit_trajectory_head()
        else:
            if getattr(self._config, 'use_flow_matching', False) and any(
                    '_trajectory_head' in k for k in cleaned):
                logger.warning(
                    "use_flow_matching=True but _trajectory_head weights were loaded from an "
                    "(official DDPM epsilon-prediction) checkpoint. An epsilon-prediction is NOT "
                    "an FM velocity field; set reinit_traj_head=True unless this ckpt was itself "
                    "trained with use_flow_matching=True."
                )
            load_result = self.load_state_dict(cleaned, strict=True)
            logger.info(
                f"Loaded DP checkpoint from {self._checkpoint_path} "
                f"(missing={len(load_result.missing_keys)}, unexpected={len(load_result.unexpected_keys)})"
            )
        self._apply_freeze_except_traj_head()

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
        return [HydraTargetBuilder(config=self._config)]

    def get_feature_builders(self) -> List[AbstractFeatureBuilder]:
        return [HydraFeatureBuilder(config=self._config)]

    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return self.model(features)

    def compute_loss(
            self,
            features: Dict[str, torch.Tensor],
            targets: Dict[str, torch.Tensor],
            predictions: Dict[str, torch.Tensor],
            tokens=None
    ):
        return dp_loss_bev(targets, predictions, self._config, self.model._trajectory_head)

    def get_optimizers(self) -> Union[Optimizer, Dict[str, Union[Optimizer, LRScheduler]]]:
        # Resume loads weights via Lightning ckpt without calling initialize(); freezing must
        # happen here so optimizer param groups match the checkpoint (1 group when frozen).
        self._apply_freeze_except_traj_head()

        backbone_params_name = '_backbone.image_encoder'
        reference_transformer_name = '_trajectory_head.reference_transformer'
        ori_transformer_name = '_trajectory_head.ori_transformer'
        img_backbone_params = list(
            filter(lambda kv: backbone_params_name in kv[0] and kv[1].requires_grad, self.model.named_parameters()))
        default_params = list(filter(lambda kv:
                                     kv[1].requires_grad and
                                     backbone_params_name not in kv[0] and
                                     reference_transformer_name not in kv[0] and
                                     ori_transformer_name not in kv[0], self.model.named_parameters()))
        params_lr_dict = [
            {'params': [tmp[1] for tmp in default_params]},
        ]
        if img_backbone_params:
            params_lr_dict.append({
                'params': [tmp[1] for tmp in img_backbone_params],
                'lr': self._lr * self._config.lr_mult_backbone,
                'weight_decay': self.backbone_wd
            })

        if self.scheduler == 'default':
            return torch.optim.Adam(params_lr_dict, lr=self._lr, weight_decay=self._config.weight_decay)
        elif self.scheduler == 'cycle':
            optim = torch.optim.Adam(params_lr_dict, lr=self._lr)
            return {
                "optimizer": optim,
                "lr_scheduler": OneCycleLR(
                    optim,
                    max_lr=0.01,
                    total_steps=100 * 202
                )
            }
        else:
            raise ValueError('Unsupported lr scheduler')

    def get_training_callbacks(self) -> List[pl.Callback]:
        ckpt_callback = ModelCheckpoint(
            save_top_k=100,
            monitor="val/loss_epoch",
            mode="min",
            dirpath=f"{os.environ.get('NAVSIM_EXP_ROOT')}/{self._config.ckpt_path}/",
            filename="{epoch:02d}-{step:04d}",
        )
        # Mid-epoch snapshots so a late-epoch crash does not lose ~hours of work
        # (ported from gtrs_aug agent).
        step_callback = ModelCheckpoint(
            every_n_train_steps=1000,
            save_on_train_epoch_end=False,
            save_top_k=-1,
            dirpath=f"{os.environ.get('NAVSIM_EXP_ROOT')}/{self._config.ckpt_path}/",
            filename="step-{step:06d}",
        )
        return [
            ckpt_callback,
            step_callback,
        ]
