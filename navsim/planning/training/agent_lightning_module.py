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

import os
import pickle
import traceback
from typing import Dict, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import StateSE2, TimePoint, StateVector2D
from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from torch import Tensor

from navsim.agents.abstract_agent import AbstractAgent
from navsim.agents.dp.dp_agent import DPAgent
from navsim.agents.gtrs_dense.gtrs_agent import GTRSAgent
from navsim.agents.gtrs_dense.hydra_features import state2traj
from navsim.agents.transfuser.transfuser_agent import TransfuserAgent
from navsim.common.dataclasses import Trajectory
from navsim.evaluate.pdm_score import transform_trajectory, get_trajectory_as_array


class AgentLightningModule(pl.LightningModule):
    """Pytorch lightning wrapper for learnable agent."""

    def __init__(self,
                 agent: AbstractAgent,
                 combined: bool = False,
                 ):
        """
        Initialise the lightning module wrapper.
        :param agent: agent interface in NAVSIM
        """
        super().__init__()
        self.combined = combined
        self.agent = agent
        self.v_params = get_pacifica_parameters()
        try:
            self.dp_preds_2hz = pickle.load(open(os.getenv('DP_PREDS'), 'rb'))
        except Exception:
            traceback.print_exc()
            self.dp_preds_2hz = dict()

    def _step(self, batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]], logging_prefix: str) -> Tensor:
        """
        Propagates the model forward and backwards and computes/logs losses and metrics.
        :param batch: tuple of dictionaries for feature and target tensors (batched)
        :param logging_prefix: prefix where to log step
        :return: scalar loss
        """
        features, targets, tokens = batch
        # DP: during validation Lightning sets the whole model to `eval()`.
        # Unfortunately DPHead.forward() triggers expensive `dp_pred` proposal sampling
        # when `not self.training`, even though the DP loss here only needs `env_kv`.
        #
        # That proposal sampling happens at every epoch-end validation and can explode
        # memory, causing the job to hang/oom. We temporarily keep DPHead in `train()`
        # mode only for the forward() call to suppress dp_pred generation.
        if isinstance(self.agent, DPAgent):
            dp_head = getattr(getattr(self.agent, "model", None), "_trajectory_head", None)
        else:
            dp_head = None

        if (not self.training) and dp_head is not None:
            prev_training = dp_head.training
            dp_head.train(True)  # suppress `if not self.training:` sampling in DPHead.forward()
            try:
                prediction = self.agent.forward(features)
            finally:
                dp_head.train(prev_training)
        else:
            prediction = self.agent.forward(features)

        if isinstance(self.agent, TransfuserAgent):
            loss, loss_dict = self.agent.compute_loss(features, targets, prediction)
        else:
            loss, loss_dict = self.agent.compute_loss(features, targets, prediction, tokens)

        for k, v in loss_dict.items():
            # Lightning forbids logging the same metric name twice with different kwargs.
            # sync_dist=True triggers DDP allreduce; with cache_path=null a slow rank can
            # hang others until NCCL watchdog timeout. Keep metrics local (no sync_dist).
            self.log(
                f"{logging_prefix}/{k}",
                v,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                sync_dist=False,
            )

        self.log(
            f"{logging_prefix}/loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=False,
        )
        return loss

    def training_step(self, batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]], batch_idx: int) -> Tensor:
        """
        Step called on training samples
        :param batch: tuple of dictionaries for feature and target tensors (batched)
        :param batch_idx: index of batch (ignored)
        :return: scalar loss
        """
        return self._step(batch, "train")

    def validation_step(self, batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]], batch_idx: int):
        """
        Step called on validation samples
        :param batch: tuple of dictionaries for feature and target tensors (batched)
        :param batch_idx: index of batch (ignored)
        :return: scalar loss
        """
        return self._step(batch, "val")

    def configure_optimizers(self):
        """Inherited, see superclass."""
        return self.agent.get_optimizers()

    def on_load_checkpoint(self, checkpoint: Dict) -> None:
        """Pad Adam moments when fm_self_conditioning widens input_emb.

        ``SimpleDiffusionTransformer._expand_input_emb_hook`` zero-pads
        ``input_emb.weight`` from (d, D) → (d, 2D) so a non-SC ckpt can resume
        with self-conditioning on. Lightning still restores optimizer_states
        with the old (d, D) shape, and the first Adam step then crashes with
        ``size of tensor a (D) must match ... (2D)``. Expand matching moments
        the same way before PL loads them.
        """
        if not isinstance(self.agent, DPAgent):
            return
        if not bool(getattr(self.agent._config, 'fm_self_conditioning', False)):
            return
        opt_states = checkpoint.get('optimizer_states')
        if not opt_states:
            return

        # Rebuild the same param order configure_optimizers will use.
        opt = self.agent.get_optimizers()
        if isinstance(opt, dict):
            opt = opt['optimizer']
        params = [p for g in opt.param_groups for p in g['params']]

        want = None
        for name, p in self.agent.model.named_parameters():
            if name.endswith('transformer_dp.input_emb.weight'):
                want = p
                break
        if want is None:
            return

        try:
            pidx = next(i for i, p in enumerate(params) if p is want)
        except StopIteration:
            return

        n_fixed = 0
        for ost in opt_states:
            st = ost.get('state', {}).get(pidx)
            if not st:
                continue
            for key in ('exp_avg', 'exp_avg_sq'):
                have = st.get(key)
                if have is None or not hasattr(have, 'shape'):
                    continue
                if tuple(have.shape) == tuple(want.shape):
                    continue
                if (have.dim() == 2 and have.shape[0] == want.shape[0]
                        and have.shape[1] < want.shape[1]):
                    padded = have.new_zeros(want.shape)
                    padded[:, :have.shape[1]] = have
                    st[key] = padded
                    n_fixed += 1
        if n_fixed:
            import logging
            logging.getLogger(__name__).info(
                "on_load_checkpoint: padded input_emb Adam state "
                "(%d tensors) for fm_self_conditioning", n_fixed
            )

    def predict_step(
            self,
            batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]],
            batch_idx: int
    ):
        if self.combined:
            return self.predict_step_combined(batch, batch_idx)
        if isinstance(self.agent, GTRSAgent):
            return self.predict_step_hydra(batch, batch_idx)
        elif isinstance(self.agent, DPAgent):
            return self.predict_step_dp_traj(batch, batch_idx)
        else:
            raise ValueError('unsupported agent')

    def predict_step_dp_traj(
            self,
            batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]],
            batch_idx: int
    ):
        features, targets, tokens = batch
        self.agent.eval()
        with torch.no_grad():
            predictions = self.agent.forward(features)
            # [B, PROPOSAL, HORIZON, 3]
            all_trajs = predictions["dp_pred"]

        # interpolate them to 10Hz
        all_interpolated_proposals = []
        for batch_idx, token in enumerate(tokens):
            ego_state = EgoState.build_from_rear_axle(
                StateSE2(*features['ego_pose'].cpu().numpy()[batch_idx]),
                tire_steering_angle=0.0,
                vehicle_parameters=self.v_params,
                time_point=TimePoint(0),
                rear_axle_velocity_2d=StateVector2D(
                    *features['ego_velocity'].cpu().numpy()[batch_idx]
                ),
                rear_axle_acceleration_2d=StateVector2D(
                    *features['ego_acceleration'].cpu().numpy()[batch_idx]
                ),
            )
            interpolated_proposals = []
            proposals = all_trajs[batch_idx].cpu().numpy()
            for proposal in proposals:
                traj = Trajectory(
                    proposal,
                    TrajectorySampling(time_horizon=4, interval_length=0.5)
                )
                trans_traj = transform_trajectory(
                    traj, ego_state
                )
                interpolated_traj = get_trajectory_as_array(
                    trans_traj,
                    TrajectorySampling(num_poses=40, interval_length=0.1),
                    ego_state.time_point
                )
                final_traj = state2traj(interpolated_traj)
                interpolated_proposals.append(final_traj)
            interpolated_proposals = np.array(interpolated_proposals)
            interpolated_proposals = torch.from_numpy(interpolated_proposals).float()[None]
            all_interpolated_proposals.append(interpolated_proposals)
        # B, 100, 40, 3
        all_interpolated_proposals = torch.cat(all_interpolated_proposals, 0)

        # 单样本推断的聚合方式（第三章）：
        #   first  —— 官方行为：取第 0 条，等价于随机采样一条
        #   medoid —— 取到其余提案距离和最小的那条，压低单样本方差
        agg = os.getenv('DP_PROPOSAL_AGG', 'first').strip().lower()
        if agg not in ('first', 'medoid'):
            raise ValueError(f'Unknown DP_PROPOSAL_AGG={agg!r}; expected first|medoid')

        result = {}
        for idx, (proposals, interp_proposals, token) in enumerate(zip(
                all_trajs.cpu().numpy(),
                all_interpolated_proposals.cpu().numpy(),
                tokens
        )):
            sel = 0
            if agg == 'medoid' and len(proposals) > 1:
                xy = interp_proposals[..., :2].astype(np.float32)       # [P, T, 2]
                d2 = ((xy[:, None] - xy[None]) ** 2).sum(-1).mean(-1)   # [P, P]
                sel = int(d2.sum(1).argmin())
            final_pose = proposals[sel]
            if final_pose.shape[0] == 40:
                interval_length = 0.1
            else:
                interval_length = 0.5

            result[token] = {
                'trajectory': Trajectory(final_pose,
                                         TrajectorySampling(time_horizon=4, interval_length=interval_length)),
                'proposals': proposals,
                'interpolated_proposal': interp_proposals
            }
        return result

    def predict_step_hydra(
            self,
            batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]],
            batch_idx: int
    ):
        features, targets, tokens = batch
        self.agent.eval()
        with torch.no_grad():
            predictions = self.agent.forward(features)
            poses = predictions["trajectory"].cpu().numpy()

            imis = predictions["imi"].softmax(-1).log().cpu().numpy()
            no_at_fault_collisions_all = predictions["no_at_fault_collisions"].sigmoid().log().cpu().numpy()
            drivable_area_compliance_all = predictions["drivable_area_compliance"].sigmoid().log().cpu().numpy()
            time_to_collision_within_bound_all = predictions[
                "time_to_collision_within_bound"].sigmoid().log().cpu().numpy()
            ego_progress_all = predictions["ego_progress"].sigmoid().log().cpu().numpy()
            driving_direction_compliance_all = predictions["driving_direction_compliance"].sigmoid().log().cpu().numpy()
            lane_keeping_all = predictions["lane_keeping"].sigmoid().log().cpu().numpy()
            traffic_light_compliance_all = predictions["traffic_light_compliance"].sigmoid().log().cpu().numpy()

        if poses.shape[1] == 40:
            interval_length = 0.1
        else:
            interval_length = 0.5

        result = {}
        for (pose,
             imi,
             no_at_fault_collisions,
             drivable_area_compliance,
             time_to_collision_within_bound,
             ego_progress,
             driving_direction_compliance,
             lane_keeping,
             traffic_light_compliance,
             token) in \
                zip(poses,
                    imis,
                    no_at_fault_collisions_all,
                    drivable_area_compliance_all,
                    time_to_collision_within_bound_all,
                    ego_progress_all,
                    driving_direction_compliance_all,
                    lane_keeping_all,
                    traffic_light_compliance_all,
                    tokens):
            result[token] = {
                'trajectory': Trajectory(pose, TrajectorySampling(time_horizon=4, interval_length=interval_length)),
                'imi': imi,
                'no_at_fault_collisions': no_at_fault_collisions,
                'drivable_area_compliance': drivable_area_compliance,
                'time_to_collision_within_bound': time_to_collision_within_bound,
                'ego_progress': ego_progress,
                'driving_direction_compliance': driving_direction_compliance,
                'lane_keeping': lane_keeping,
                'traffic_light_compliance': traffic_light_compliance,
            }
        return result

    def predict_step_combined(
            self,
            batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]],
            not_used: int
    ):
        features, targets, tokens = batch
        device = features['ego_pose'].device
        all_interpolated_proposals = []
        for batch_idx, token in enumerate(tokens):
            proposals = self.dp_preds_2hz[token]['proposals']
            ego_state = EgoState.build_from_rear_axle(
                StateSE2(*features['ego_pose'].cpu().numpy()[batch_idx]),
                tire_steering_angle=0.0,
                vehicle_parameters=self.v_params,
                time_point=TimePoint(0),
                rear_axle_velocity_2d=StateVector2D(
                    *features['ego_velocity'].cpu().numpy()[batch_idx]
                ),
                rear_axle_acceleration_2d=StateVector2D(
                    *features['ego_acceleration'].cpu().numpy()[batch_idx]
                ),
            )
            interpolated_proposals = []
            for proposal in proposals:
                traj = Trajectory(
                    proposal,
                    TrajectorySampling(time_horizon=4, interval_length=0.5)
                )
                trans_traj = transform_trajectory(
                    traj, ego_state
                )
                interpolated_traj = get_trajectory_as_array(
                    trans_traj,
                    TrajectorySampling(num_poses=40, interval_length=0.1),
                    ego_state.time_point
                )
                final_traj = state2traj(interpolated_traj)
                interpolated_proposals.append(final_traj)
            interpolated_proposals = np.array(interpolated_proposals)
            interpolated_proposals = torch.from_numpy(interpolated_proposals).float().to(device)[None]
            all_interpolated_proposals.append(interpolated_proposals)
        # B, 100, 40, 3
        all_interpolated_proposals = torch.cat(all_interpolated_proposals, 0)

        self.agent.eval()
        with torch.no_grad():
            predictions = self.agent.evaluate_dp_proposals(features, all_interpolated_proposals)
            poses = predictions["trajectory"].cpu().numpy()
            imis = predictions["imi"].softmax(-1).log().cpu().numpy()
            no_at_fault_collisions_all = predictions["no_at_fault_collisions"].sigmoid().log().cpu().numpy()
            drivable_area_compliance_all = predictions["drivable_area_compliance"].sigmoid().log().cpu().numpy()
            time_to_collision_within_bound_all = predictions[
                "time_to_collision_within_bound"].sigmoid().log().cpu().numpy()
            ego_progress_all = predictions["ego_progress"].sigmoid().log().cpu().numpy()
            driving_direction_compliance_all = predictions[
                "driving_direction_compliance"].sigmoid().log().cpu().numpy()
            lane_keeping_all = predictions["lane_keeping"].sigmoid().log().cpu().numpy()
            traffic_light_compliance_all = predictions["traffic_light_compliance"].sigmoid().log().cpu().numpy()

        if poses.shape[1] == 40:
            interval_length = 0.1
        else:
            interval_length = 0.5

        result = {}
        for (pose,
             interpolated_proposal,
             imi,
             no_at_fault_collisions,
             drivable_area_compliance,
             time_to_collision_within_bound,
             ego_progress,
             driving_direction_compliance,
             lane_keeping,
             traffic_light_compliance,
             token) in zip(poses,
                           all_interpolated_proposals.cpu().numpy(),
                           imis,
                           no_at_fault_collisions_all,
                           drivable_area_compliance_all,
                           time_to_collision_within_bound_all,
                           ego_progress_all,
                           driving_direction_compliance_all,
                           lane_keeping_all,
                           traffic_light_compliance_all,
                           tokens):
            result[token] = {
                'trajectory': Trajectory(pose, TrajectorySampling(time_horizon=4, interval_length=interval_length)),
                'imi': imi,
                'no_at_fault_collisions': no_at_fault_collisions,
                'drivable_area_compliance': drivable_area_compliance,
                'time_to_collision_within_bound': time_to_collision_within_bound,
                'ego_progress': ego_progress,
                'driving_direction_compliance': driving_direction_compliance,
                'lane_keeping': lane_keeping,
                'traffic_light_compliance': traffic_light_compliance,
                'interpolated_proposal': interpolated_proposal,
            }
        return result