# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from typing import Any, Dict, List, Union

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler, OneCycleLR

from navsim.agents.abstract_agent import AbstractAgent
from navsim.agents.dp.dp_config import DPConfig
from navsim.agents.dp.dp_model import DPModel
from navsim.agents.gtrs_dense.gtrs_agent import GTRSAgent
from navsim.agents.gtrs_dense.hydra_config import HydraConfig
from navsim.agents.gtrs_dense.hydra_features import HydraFeatureBuilder, HydraTargetBuilder
from navsim.common.dataclasses import SensorConfig, Trajectory


class GTRSGuidedDPAgent(AbstractAgent):
    """Scheme A: use a pretrained GTRS-Dense scorer to select an anchor trajectory,
    then condition DP sampling on this anchor and rescore to select final output.

    Notes:
    - This is an inference-first implementation. Training is not modified.
    - Anchor conditioning is implemented in DPModel/DPHead via features['anchor_traj'].
    """

    def __init__(
        self,
        dp_config: DPConfig,
        dense_config: HydraConfig,
        lr: float,
        dp_checkpoint_path: str = None,
        dense_checkpoint_path: str = None,
        pdm_gt_path: str | None = None,
        anchor_topk: int = 1,
        final_topk: int = 1,
        dp_only_inference: bool = True,
    ):
        super().__init__(trajectory_sampling=dp_config.trajectory_sampling)
        self._dp_config = dp_config
        self._dense_config = dense_config
        self._lr = lr
        self._dp_checkpoint_path = dp_checkpoint_path
        self._dense_checkpoint_path = dense_checkpoint_path
        self._pdm_gt_path = pdm_gt_path
        self._anchor_topk = anchor_topk
        self._final_topk = final_topk
        self._dp_only_inference = dp_only_inference

        self.dp_model = DPModel(dp_config)
        # For scoring we reuse the dense agent implementation (HydraModel inside)
        self.dense_agent = GTRSAgent(
            config=dense_config,
            lr=lr,
            checkpoint_path=dense_checkpoint_path,
            pdm_gt_path=pdm_gt_path,
        )

        self.scheduler = dp_config.scheduler
        self.backbone_wd = dp_config.backbone_wd

    def name(self) -> str:
        return self.__class__.__name__

    def initialize(self) -> None:
        # Load DP
        if self._dp_checkpoint_path is None:
            raise ValueError("dp_checkpoint_path is required for GTRSGuidedDPAgent")
        dp_state: Dict[str, Any] = torch.load(self._dp_checkpoint_path, map_location=torch.device("cpu"))["state_dict"]
        self.dp_model.load_state_dict({k.replace("agent.", ""): v for k, v in dp_state.items()}, strict=False)

        # Load Dense scorer
        if self._dense_checkpoint_path is None:
            raise ValueError("dense_checkpoint_path is required for GTRSGuidedDPAgent")
        self.dense_agent.initialize()

    def get_sensor_config(self) -> SensorConfig:
        # Same sensors as DP / Dense
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

    def get_target_builders(self):
        # In inference we can reuse hydra builders (status/camera features). Targets unused.
        return [HydraTargetBuilder(config=self._dp_config)]

    def get_feature_builders(self):
        return [HydraFeatureBuilder(config=self._dp_config)]

    @torch.no_grad()
    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # 1) DP generate proposals (unconditioned)
        dp_out_1 = self.dp_model(features)
        dp_proposals_1 = dp_out_1.get("dp_pred")  # [B, N, 8, 3]
        if dp_proposals_1 is None:
            raise RuntimeError("DP did not output 'dp_pred' in inference mode")

        # 2) Dense scorer evaluates proposals and selects anchor
        score_out_1 = self.dense_agent.evaluate_dp_proposals(
            features,
            dp_proposals_1,
        )
        # HydraTrajHead.eval_dp_proposals returns 'selected_traj' (topk trajectories) in most setups.
        # Fall back to 'trajectory' if necessary.
        anchor_traj = score_out_1.get("selected_traj", None)
        if anchor_traj is None:
            anchor_traj = score_out_1.get("trajectory", None)
        if anchor_traj is None:
            raise RuntimeError("Dense scorer output does not contain 'selected_traj' or 'trajectory'")

        # anchor_traj could be [B, K, T, 3] or [B, T, 3]
        if anchor_traj.dim() == 4:
            # use top-1
            anchor_traj = anchor_traj[:, 0]

        # 3) DP generate proposals conditioned on anchor
        features_guided = dict(features)
        features_guided["anchor_traj"] = anchor_traj
        dp_out_2 = self.dp_model(features_guided)
        dp_proposals_2 = dp_out_2.get("dp_pred")

        # 4) Dense rescore final proposals and pick final trajectory
        score_out_2 = self.dense_agent.evaluate_dp_proposals(
            features,
            dp_proposals_2,
        )

        # Return in the standard format expected by AgentLightningModule: 'trajectory'
        final = {}
        if "trajectory" in score_out_2:
            final["trajectory"] = score_out_2["trajectory"]
        elif "selected_traj" in score_out_2:
            # pick top-1
            final["trajectory"] = score_out_2["selected_traj"][:, 0]
        else:
            # last fallback: take anchor itself
            final["trajectory"] = anchor_traj
        return final

    def compute_loss(self, *args, **kwargs):
        raise NotImplementedError("GTRSGuidedDPAgent is intended for inference-first experiments.")

    def get_optimizers(self) -> Union[Optimizer, Dict[str, Union[Optimizer, LRScheduler]]]:
        # Provide optimizer for compatibility (unused in inference)
        params = list(self.dp_model.parameters())
        if self.scheduler == 'default':
            return torch.optim.Adam(params, lr=self._lr, weight_decay=self._dp_config.weight_decay)
        elif self.scheduler == 'cycle':
            optim = torch.optim.Adam(params, lr=self._lr)
            return {
                "optimizer": optim,
                "lr_scheduler": OneCycleLR(
                    optim,
                    max_lr=0.01,
                    total_steps=100 * 202,
                ),
            }
        else:
            raise ValueError('Unsupported lr scheduler')

    def get_training_callbacks(self) -> List[pl.Callback]:
        ckpt_callback = ModelCheckpoint(
            save_top_k=10,
            monitor="val/loss_epoch",
            mode="min",
            dirpath=f"{os.environ.get('NAVSIM_EXP_ROOT')}/guided_dp_ckpt/",
            filename="{epoch:02d}-{step:04d}",
        )
        return [ckpt_callback]
