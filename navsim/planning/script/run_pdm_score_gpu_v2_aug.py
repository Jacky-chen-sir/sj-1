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
import pickle
import traceback
import uuid
from dataclasses import fields
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Union

import hydra
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch.distributed as dist
from hydra.utils import instantiate
from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.common.geometry.convert import relative_to_absolute_poses
from nuplan.planning.script.builders.logging_builder import build_logger
from nuplan.planning.utils.multithreading.worker_utils import worker_map
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import PDMResults, SensorConfig
from navsim.common.dataloader import MetricCacheLoader, SceneFilter, SceneLoader
from navsim.common.enums import SceneFrameType
from navsim.evaluate.pdm_score import pdm_score
from navsim.planning.script.builders.worker_pool_builder import build_worker
from navsim.planning.script.run_pdm_score import compute_final_scores, calculate_individual_mapping_scores, \
    create_scene_aggregators
from navsim.planning.script.run_pdm_score_gpu_v2 import (
    _finalize_minimal_scores,
    _finalize_one_stage_scores,
    _finalize_two_stage_scores,
)
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
from navsim.planning.training.agent_lightning_module_aug import AgentLightningModuleAug
from navsim.planning.training.dataset_aug import DatasetAug as Dataset
from navsim.traffic_agents_policies.abstract_traffic_agents_policy import AbstractTrafficAgentsPolicy

logger = logging.getLogger(__name__)

CONFIG_PATH = "config/pdm_scoring"
CONFIG_NAME = "default_run_pdm_score_gpu"


def run_pdm_score_wo_inference(args: List[Dict[str, Union[List[str], DictConfig]]]) -> List[pd.DataFrame]:
    """
    Helper function to run PDMS evaluation in.
    :param args: input arguments
    """
    node_id = int(os.environ.get("NODE_RANK", 0))
    thread_id = str(uuid.uuid4())
    logger.info(f"Starting worker in thread_id={thread_id}, node_id={node_id}")

    log_names = [a["log_file"] for a in args]
    tokens = [t for a in args for t in a["tokens"]]
    cfg: DictConfig = args[0]["cfg"]
    model_trajectory = {}
    for a in args:
        model_trajectory.update(a["model_trajectory"])

    simulator: PDMSimulator = instantiate(cfg.simulator)
    scorer: PDMScorer = instantiate(cfg.scorer)
    assert (
            simulator.proposal_sampling == scorer.proposal_sampling
    ), "Simulator and scorer proposal sampling has to be identical"

    metric_cache_loader = MetricCacheLoader(Path(cfg.metric_cache_path))
    scene_filter: SceneFilter = instantiate(cfg.train_test_split.scene_filter)
    scene_filter.log_names = log_names
    scene_filter.tokens = tokens
    scene_loader = SceneLoader(
        synthetic_sensor_path=Path(cfg.synthetic_sensor_path),
        original_sensor_path=Path(cfg.original_sensor_path),
        data_path=Path(cfg.navsim_log_path),
        synthetic_scenes_path=Path(cfg.synthetic_scenes_path),
        scene_filter=scene_filter,
    )

    pdm_results: List[pd.DataFrame] = []

    # first stage

    traffic_agents_policy_stage_one: AbstractTrafficAgentsPolicy = instantiate(
        cfg.traffic_agents_policy.reactive, simulator.proposal_sampling
    )

    scene_loader_tokens_stage_one = scene_loader.tokens_stage_one or []
    metric_cache_tokens = metric_cache_loader.tokens or []

    tokens_to_evaluate_stage_one = list(set(scene_loader_tokens_stage_one) & set(metric_cache_tokens))

    for idx, (token) in enumerate(tokens_to_evaluate_stage_one):
        logger.info(
            f"Processing stage one reactive scenario {idx + 1} / {len(tokens_to_evaluate_stage_one)} in thread_id={thread_id}, node_id={node_id}"
        )
        try:
            metric_cache = metric_cache_loader.get_from_token(token)
            trajectory = model_trajectory[token]['trajectory']

            score_row_stage_one, ego_simulated_states = pdm_score(
                metric_cache=metric_cache,
                model_trajectory=trajectory,
                future_sampling=simulator.proposal_sampling,
                simulator=simulator,
                scorer=scorer,
                traffic_agents_policy=traffic_agents_policy_stage_one,
            )
            score_row_stage_one["valid"] = True
            score_row_stage_one["log_name"] = metric_cache.log_name
            score_row_stage_one["frame_type"] = metric_cache.scene_type
            score_row_stage_one["start_time"] = metric_cache.timepoint.time_s
            end_pose = StateSE2(
                x=trajectory.poses[-1, 0],
                y=trajectory.poses[-1, 1],
                heading=trajectory.poses[-1, 2],
            )
            absolute_endpoint = relative_to_absolute_poses(metric_cache.ego_state.rear_axle, [end_pose])[0]
            score_row_stage_one["endpoint_x"] = absolute_endpoint.x
            score_row_stage_one["endpoint_y"] = absolute_endpoint.y
            score_row_stage_one["start_point_x"] = metric_cache.ego_state.rear_axle.x
            score_row_stage_one["start_point_y"] = metric_cache.ego_state.rear_axle.y
            score_row_stage_one["ego_simulated_states"] = [ego_simulated_states]  # used for two-frames extended comfort

        except Exception:
            logger.warning(f"----------- Agent failed for token {token}:")
            traceback.print_exc()
            score_row_stage_one = pd.DataFrame([PDMResults.get_empty_results()])
            score_row_stage_one["valid"] = False
        score_row_stage_one["token"] = token

        pdm_results.append(score_row_stage_one)

    # second stage

    traffic_agents_policy_stage_two: AbstractTrafficAgentsPolicy = instantiate(
        cfg.traffic_agents_policy.reactive, simulator.proposal_sampling
    )
    scene_loader_tokens_stage_two = scene_loader.reactive_tokens_stage_two or []
    if scene_loader.reactive_tokens_stage_two is None:
        logger.warning("scene_loader.reactive_tokens_stage_two is None, skipping stage two for this worker.")

    tokens_to_evaluate_stage_two = list(set(scene_loader_tokens_stage_two) & set(metric_cache_tokens))
    for idx, (token) in enumerate(tokens_to_evaluate_stage_two):
        logger.info(
            f"Processing stage two reactive scenario {idx + 1} / {len(tokens_to_evaluate_stage_two)} in thread_id={thread_id}, node_id={node_id}"
        )
        try:
            metric_cache = metric_cache_loader.get_from_token(token)
            trajectory = model_trajectory[token]['trajectory']

            score_row_stage_two, ego_simulated_states = pdm_score(
                metric_cache=metric_cache,
                model_trajectory=trajectory,
                future_sampling=simulator.proposal_sampling,
                simulator=simulator,
                scorer=scorer,
                traffic_agents_policy=traffic_agents_policy_stage_two,
            )
            score_row_stage_two["valid"] = True
            score_row_stage_two["log_name"] = metric_cache.log_name
            score_row_stage_two["frame_type"] = metric_cache.scene_type
            score_row_stage_two["start_time"] = metric_cache.timepoint.time_s
            end_pose = StateSE2(
                x=trajectory.poses[-1, 0],
                y=trajectory.poses[-1, 1],
                heading=trajectory.poses[-1, 2],
            )
            absolute_endpoint = relative_to_absolute_poses(metric_cache.ego_state.rear_axle, [end_pose])[0]
            score_row_stage_two["endpoint_x"] = absolute_endpoint.x
            score_row_stage_two["endpoint_y"] = absolute_endpoint.y
            score_row_stage_two["start_point_x"] = metric_cache.ego_state.rear_axle.x
            score_row_stage_two["start_point_y"] = metric_cache.ego_state.rear_axle.y
            score_row_stage_two["ego_simulated_states"] = [ego_simulated_states]  # used for two-frames extended comfort

        except Exception:
            logger.warning(f"----------- Agent failed for token {token}:")
            traceback.print_exc()
            score_row_stage_two = pd.DataFrame([PDMResults.get_empty_results()])
            score_row_stage_two["valid"] = False
        score_row_stage_two["token"] = token

        pdm_results.append(score_row_stage_two)

    return pdm_results


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Main entrypoint for running PDMS evaluation.
    :param cfg: omegaconf dictionary
    """

    build_logger(cfg)
    pkl_path = os.getenv('SUBSCORE_PATH')
    if not pkl_path:
        # Hydra output_dir is always set; dumping to None used to crash after a full
        # inference pass (TypeError: expected str, not NoneType).
        pkl_path = str(Path(cfg.output_dir) / "subscores.pkl")
        logger.info("SUBSCORE_PATH unset; writing proposals to %s", pkl_path)
    skip_infer = os.getenv('SKIP_INFER', '').lower() in ('1', 'true', 'yes')
    if skip_infer and not (pkl_path and os.path.isfile(pkl_path)):
        raise FileNotFoundError(f"SKIP_INFER=1 but pickle not found: {pkl_path}")

    scene_filter = instantiate(cfg.train_test_split.scene_filter)
    scene_loader = SceneLoader(
        synthetic_sensor_path=None,
        original_sensor_path=None,
        data_path=Path(cfg.navsim_log_path),
        synthetic_scenes_path=Path(cfg.synthetic_scenes_path),
        scene_filter=scene_filter,
        sensor_config=SensorConfig.build_no_sensors(),
    )
    metric_cache_loader = MetricCacheLoader(Path(cfg.metric_cache_path))

    tokens_to_evaluate = list(set(scene_loader.tokens) & set(metric_cache_loader.tokens))
    num_missing_metric_cache_tokens = len(set(scene_loader.tokens) - set(metric_cache_loader.tokens))
    num_unused_metric_cache_tokens = len(set(metric_cache_loader.tokens) - set(scene_loader.tokens))
    if num_missing_metric_cache_tokens > 0:
        logger.warning(f"Missing metric cache for {num_missing_metric_cache_tokens} tokens. Skipping these tokens.")
    if num_unused_metric_cache_tokens > 0:
        logger.warning(f"Unused metric cache for {num_unused_metric_cache_tokens} tokens. Skipping these tokens.")
    logger.info(f"Starting pdm scoring of {len(tokens_to_evaluate)} scenarios...")

    if skip_infer:
        logger.info(f"SKIP_INFER=1, loading proposals from {pkl_path}")
        merged_predictions = pickle.load(open(pkl_path, 'rb'))
    else:
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
            append_token_to_batch=True
        )
        dataloader = DataLoader(dataset, **cfg.dataloader.params, shuffle=False)
        if len(dataset) != len(tokens_to_evaluate):
            logger.warning(
                f"Dataloader has {len(dataset)} samples vs {len(tokens_to_evaluate)} metric-cache tokens; "
                "scoring will use the intersection only."
            )

        trainer = pl.Trainer(**cfg.trainer.params, callbacks=agent.get_training_callbacks())
        predictions = trainer.predict(
            AgentLightningModuleAug(
                cfg=cfg.agent.config,
                agent=agent,
            ),
            dataloader,
            return_predictions=True
        )

        # Single-GPU predict does not init a process group; only barrier/gather when DDP is active.
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
            all_predictions = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(all_predictions, predictions)
            if dist.get_rank() != 0:
                return None
        else:
            all_predictions = [predictions]

        merged_predictions = {}
        for proc_prediction in all_predictions:
            for d in proc_prediction:
                merged_predictions.update(d)

        pickle.dump(merged_predictions, open(pkl_path, 'wb'))
        logger.info(f"Wrote proposals/subscores to {pkl_path}")

    scoring_predictions = {
        token: {"trajectory": prediction["trajectory"]}
        for token, prediction in merged_predictions.items()
    }
    del merged_predictions

    data_points = [
        {
            "cfg": cfg,
            "log_file": log_file,
            "tokens": tokens_list,
            "model_trajectory": {
                token: scoring_predictions[token]
                for token in tokens_list
                if token in scoring_predictions
            },
        }
        for log_file, tokens_list in scene_loader.get_tokens_list_per_log().items()
    ]

    worker = build_worker(cfg)
    score_rows: List[pd.DataFrame] = worker_map(worker, run_pdm_score_wo_inference, data_points)

    raw_pdm_score_df = pd.concat(score_rows, ignore_index=True)
    save_path = Path(cfg.output_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    raw_score_path = save_path / "raw_pdm_score.pkl"
    try:
        with open(raw_score_path, "wb") as f:
            pickle.dump(raw_pdm_score_df, f)
        logger.info(f"Raw PDM score rows are stored in: {raw_score_path}.")
    except Exception:
        logger.exception("Failed to save raw PDM score rows; continuing to final aggregation.")

    num_sucessful_scenarios = int(raw_pdm_score_df["valid"].fillna(False).sum())
    num_failed_scenarios = len(raw_pdm_score_df) - num_sucessful_scenarios
    if num_failed_scenarios > 0:
        failed_tokens = raw_pdm_score_df[~raw_pdm_score_df["valid"].fillna(False)]["token"].to_list()
    else:
        failed_tokens = []

    if cfg.train_test_split.get("reactive_all_mapping"):
        try:
            pdm_score_df = _finalize_two_stage_scores(raw_pdm_score_df.copy(), cfg, scene_loader)
        except Exception:
            logger.exception("Two-stage final aggregation failed; writing one-stage fallback CSV instead.")
            try:
                pdm_score_df = _finalize_one_stage_scores(raw_pdm_score_df.copy(), cfg)
            except Exception:
                logger.exception("One-stage fallback aggregation failed; writing minimal raw fallback CSV instead.")
                pdm_score_df = _finalize_minimal_scores(raw_pdm_score_df.copy())
    else:
        logger.warning("No train_test_split.reactive_all_mapping configured; writing one-stage fallback CSV.")
        try:
            pdm_score_df = _finalize_one_stage_scores(raw_pdm_score_df.copy(), cfg)
        except Exception:
            logger.exception("One-stage fallback aggregation failed; writing minimal raw fallback CSV instead.")
            pdm_score_df = _finalize_minimal_scores(raw_pdm_score_df.copy())

    timestamp = datetime.now().strftime("%Y.%m.%d.%H.%M.%S")
    csv_path = save_path / f"{timestamp}.csv"
    pdm_score_df.to_csv(csv_path)

    final_score_rows = pdm_score_df[pdm_score_df["token"] == "extended_pdm_score_combined"]
    final_score = final_score_rows["score"].iloc[0] if not final_score_rows.empty else np.nan
    aggregation_mode = (
        pdm_score_df["aggregation_mode"].iloc[0] if "aggregation_mode" in pdm_score_df.columns else "unknown"
    )

    logger.info(
        f"""
        Finished running evaluation.
            Number of successful scenarios: {num_sucessful_scenarios}.
            Number of failed scenarios: {num_failed_scenarios}.
            Aggregation mode: {aggregation_mode}.
            Final extended pdm score of valid results: {final_score}.
            Raw score rows are stored in: {raw_score_path}.
            Results are stored in: {csv_path}.
        """
    )

    if cfg.verbose:
        logger.info(
            f"""
            Detailed results:
            {pdm_score_df.iloc[-3:].T}
            """
        )
    if num_failed_scenarios > 0:
        logger.info(
            f"""
            List of failed tokens:
            {failed_tokens}
            """
        )


if __name__ == "__main__":
    main()
