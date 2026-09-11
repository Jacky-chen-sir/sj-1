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
from navsim.planning.script.run_pdm_score import create_scene_aggregators, calculate_individual_mapping_scores, \
    compute_final_scores
from navsim.planning.script.run_pdm_score_one_stage import (
    create_scene_aggregators as create_one_stage_scene_aggregators,
    compute_final_scores as compute_one_stage_final_scores,
    infer_start_adjacent_mapping,
)
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
from navsim.planning.training.agent_lightning_module import AgentLightningModule
from navsim.planning.training.dataset import Dataset
from navsim.traffic_agents_policies.abstract_traffic_agents_policy import AbstractTrafficAgentsPolicy

logger = logging.getLogger(__name__)

CONFIG_PATH = "config/pdm_scoring"
CONFIG_NAME = "default_run_pdm_score_gpu"


def run_pdm_score(args: List[Dict[str, Union[List[str], DictConfig]]]) -> List[pd.DataFrame]:
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


def _score_columns(pdm_score_df: pd.DataFrame) -> List[str]:
    return [
        c
        for c in pdm_score_df.columns
        if (
            (any(score.name in c for score in fields(PDMResults)) or c == "two_frame_extended_comfort" or c == "score")
            and c != "pdm_score"
        )
    ]


def _numeric_mean(df: pd.DataFrame, score_cols: List[str]) -> pd.Series:
    if df.empty:
        return pd.Series({col: np.nan for col in score_cols})
    return df[score_cols].apply(pd.to_numeric, errors="coerce").mean(skipna=True)


def _drop_internal_score_columns(pdm_score_df: pd.DataFrame) -> pd.DataFrame:
    return pdm_score_df.drop(
        columns=[
            c
            for c in ["weighted_metrics", "weighted_metrics_array", "multiplicative_metrics_prod", "ego_simulated_states"]
            if c in pdm_score_df.columns
        ],
        errors="ignore",
    )


def _finalize_two_stage_scores(
    pdm_score_df: pd.DataFrame, cfg: DictConfig, scene_loader: SceneLoader
) -> pd.DataFrame:
    raw_mapping = cfg.train_test_split.get("reactive_all_mapping")
    if not raw_mapping:
        raise ValueError("train_test_split.reactive_all_mapping is empty; cannot compute two-stage scores.")

    scene_tokens = set(scene_loader.tokens)
    all_mappings: Dict[Tuple[str, str], List[Tuple[str, str]]] = {}
    for orig_token, prev_token, two_stage_pairs in raw_mapping:
        if prev_token in scene_tokens or orig_token in scene_tokens:
            all_mappings[(orig_token, prev_token)] = [tuple(pair) for pair in two_stage_pairs]

    if not all_mappings:
        raise ValueError("No two-stage mapping matched the loaded scenes.")

    pdm_score_df = create_scene_aggregators(
        all_mappings, pdm_score_df, instantiate(cfg.simulator.proposal_sampling)
    )
    pdm_score_df = compute_final_scores(pdm_score_df)

    score_cols = _score_columns(pdm_score_df)
    pcl_group_score, pcl_stage1_score, pcl_stage2_score = calculate_individual_mapping_scores(
        pdm_score_df[score_cols + ["token", "weight"]], all_mappings
    )

    for col in score_cols:
        stage_one_mask = pdm_score_df["frame_type"] == SceneFrameType.ORIGINAL
        stage_two_mask = pdm_score_df["frame_type"] == SceneFrameType.SYNTHETIC

        pdm_score_df.loc[stage_one_mask, f"{col}_stage_one"] = pdm_score_df.loc[stage_one_mask, col]
        pdm_score_df.loc[stage_two_mask, f"{col}_stage_two"] = pdm_score_df.loc[stage_two_mask, col]

    pdm_score_df.drop(columns=score_cols, inplace=True)
    pdm_score_df["score"] = pdm_score_df["score_stage_one"].combine_first(pdm_score_df["score_stage_two"])
    pdm_score_df.drop(columns=["score_stage_one", "score_stage_two"], inplace=True)

    stage1_cols = [f"{col}_stage_one" for col in score_cols if col != "score"]
    stage2_cols = [f"{col}_stage_two" for col in score_cols if col != "score"]
    score_cols = stage1_cols + stage2_cols + ["score"]

    pdm_score_df = pdm_score_df[["token", "valid"] + score_cols]

    summary_rows = []

    stage1_row = pd.Series(index=pdm_score_df.columns, dtype=object)
    stage1_row["token"] = "extended_pdm_score_stage_one"
    stage1_row["valid"] = True
    stage1_row["score"] = pcl_stage1_score.get("score", np.nan)
    for col in pcl_stage1_score.index:
        if col not in ["token", "valid", "score"]:
            stage1_row[f"{col}_stage_one"] = pcl_stage1_score[col]
    summary_rows.append(stage1_row)

    stage2_row = pd.Series(index=pdm_score_df.columns, dtype=object)
    stage2_row["token"] = "extended_pdm_score_stage_two"
    stage2_row["valid"] = True
    stage2_row["score"] = pcl_stage2_score.get("score", np.nan)
    for col in pcl_stage2_score.index:
        if col not in ["token", "valid", "score"]:
            stage2_row[f"{col}_stage_two"] = pcl_stage2_score[col]
    summary_rows.append(stage2_row)

    combined_row = pd.Series(index=pdm_score_df.columns, dtype=object)
    combined_row["token"] = "extended_pdm_score_combined"
    combined_row["valid"] = True
    combined_row["score"] = pcl_group_score.get("score", np.nan)

    for col in pcl_stage1_score.index:
        if col not in ["token", "valid", "score"]:
            combined_row[f"{col}_stage_one"] = pcl_stage1_score[col]

    for col in pcl_stage2_score.index:
        if col not in ["token", "valid", "score"]:
            combined_row[f"{col}_stage_two"] = pcl_stage2_score[col]
    summary_rows.append(combined_row)

    pdm_score_df = pd.concat([pdm_score_df, pd.DataFrame(summary_rows)], ignore_index=True)
    pdm_score_df["aggregation_mode"] = "two_stage"
    return pdm_score_df


def _finalize_one_stage_scores(pdm_score_df: pd.DataFrame, cfg: DictConfig) -> pd.DataFrame:
    pdm_score_df = pdm_score_df.copy()

    try:
        start_adjacent_mapping = infer_start_adjacent_mapping(pdm_score_df)
        if start_adjacent_mapping:
            pdm_score_df = create_one_stage_scene_aggregators(
                start_adjacent_mapping, pdm_score_df, instantiate(cfg.simulator.proposal_sampling)
            )
            pdm_score_df = compute_one_stage_final_scores(pdm_score_df)
        else:
            logger.warning("No adjacent one-stage mapping found; scoring without two-frame extended comfort.")
            if "score" not in pdm_score_df.columns:
                pdm_score_df["score"] = pdm_score_df["pdm_score"] if "pdm_score" in pdm_score_df.columns else np.nan
            pdm_score_df["two_frame_extended_comfort"] = np.nan
            pdm_score_df = _drop_internal_score_columns(pdm_score_df)

    except Exception:
        logger.exception("Failed to calculate one-stage two-frame comfort; using raw pdm_score fallback.")
        if "score" not in pdm_score_df.columns:
            pdm_score_df["score"] = pdm_score_df["pdm_score"] if "pdm_score" in pdm_score_df.columns else np.nan
        if "two_frame_extended_comfort" not in pdm_score_df.columns:
            pdm_score_df["two_frame_extended_comfort"] = np.nan
        pdm_score_df = _drop_internal_score_columns(pdm_score_df)

    score_cols = list(dict.fromkeys(_score_columns(pdm_score_df)))
    if "score" in pdm_score_df.columns and "score" not in score_cols:
        score_cols.append("score")

    valid_mask = pdm_score_df["valid"].fillna(False).astype(bool)
    summary_source = pdm_score_df.loc[valid_mask, score_cols]
    if summary_source.empty:
        summary_source = pdm_score_df[score_cols]
    average_scores = _numeric_mean(summary_source, score_cols)
    all_scenarios_valid = bool(valid_mask.all())

    pdm_score_df = pdm_score_df[["token", "valid"] + score_cols]
    summary_rows = []

    average_row = pd.Series(index=pdm_score_df.columns, dtype=object)
    average_row.loc[average_scores.index] = average_scores
    average_row["token"] = "average_all_frames"
    average_row["valid"] = all_scenarios_valid
    summary_rows.append(average_row)

    stage1_row = pd.Series(index=pdm_score_df.columns, dtype=object)
    stage1_row.loc[average_scores.index] = average_scores
    stage1_row["token"] = "extended_pdm_score_stage_one"
    stage1_row["valid"] = all_scenarios_valid
    summary_rows.append(stage1_row)

    stage2_row = pd.Series(index=pdm_score_df.columns, dtype=object)
    stage2_row["token"] = "extended_pdm_score_stage_two"
    stage2_row["valid"] = False
    stage2_row["score"] = np.nan
    summary_rows.append(stage2_row)

    combined_row = pd.Series(index=pdm_score_df.columns, dtype=object)
    combined_row.loc[average_scores.index] = average_scores
    combined_row["token"] = "extended_pdm_score_combined"
    combined_row["valid"] = all_scenarios_valid
    summary_rows.append(combined_row)

    pdm_score_df = pd.concat([pdm_score_df, pd.DataFrame(summary_rows)], ignore_index=True)
    pdm_score_df["aggregation_mode"] = "one_stage_fallback"
    return pdm_score_df


def _finalize_minimal_scores(pdm_score_df: pd.DataFrame) -> pd.DataFrame:
    pdm_score_df = pdm_score_df.copy()

    if "score" not in pdm_score_df.columns:
        pdm_score_df["score"] = pdm_score_df["pdm_score"] if "pdm_score" in pdm_score_df.columns else np.nan
    if "two_frame_extended_comfort" not in pdm_score_df.columns:
        pdm_score_df["two_frame_extended_comfort"] = np.nan

    pdm_score_df = _drop_internal_score_columns(pdm_score_df)
    score_cols = list(dict.fromkeys(_score_columns(pdm_score_df)))
    if "score" in pdm_score_df.columns and "score" not in score_cols:
        score_cols.append("score")

    if "token" not in pdm_score_df.columns:
        pdm_score_df["token"] = pdm_score_df.index.astype(str)
    if "valid" not in pdm_score_df.columns:
        pdm_score_df["valid"] = False

    valid_mask = pdm_score_df["valid"].fillna(False).astype(bool)
    summary_source = pdm_score_df.loc[valid_mask, score_cols]
    if summary_source.empty:
        summary_source = pdm_score_df[score_cols]
    average_scores = _numeric_mean(summary_source, score_cols)
    all_scenarios_valid = bool(valid_mask.all())

    pdm_score_df = pdm_score_df[["token", "valid"] + score_cols]
    summary_rows = []

    average_row = pd.Series(index=pdm_score_df.columns, dtype=object)
    average_row.loc[average_scores.index] = average_scores
    average_row["token"] = "average_all_frames"
    average_row["valid"] = all_scenarios_valid
    summary_rows.append(average_row)

    stage1_row = pd.Series(index=pdm_score_df.columns, dtype=object)
    stage1_row.loc[average_scores.index] = average_scores
    stage1_row["token"] = "extended_pdm_score_stage_one"
    stage1_row["valid"] = all_scenarios_valid
    summary_rows.append(stage1_row)

    stage2_row = pd.Series(index=pdm_score_df.columns, dtype=object)
    stage2_row["token"] = "extended_pdm_score_stage_two"
    stage2_row["valid"] = False
    stage2_row["score"] = np.nan
    summary_rows.append(stage2_row)

    combined_row = pd.Series(index=pdm_score_df.columns, dtype=object)
    combined_row.loc[average_scores.index] = average_scores
    combined_row["token"] = "extended_pdm_score_combined"
    combined_row["valid"] = all_scenarios_valid
    summary_rows.append(combined_row)

    pdm_score_df = pd.concat([pdm_score_df, pd.DataFrame(summary_rows)], ignore_index=True)
    pdm_score_df["aggregation_mode"] = "minimal_raw_fallback"
    return pdm_score_df


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Main entrypoint for running PDMS evaluation.
    :param cfg: omegaconf dictionary
    """

    build_logger(cfg)
    combined = cfg.get('combined_inference', False)

    print(f'Combined inference: {combined}')
    dump_path = os.getenv('SUBSCORE_PATH')
    print(f'Subscore/Trajectories saved to {dump_path}')
    skip_infer = os.getenv('SKIP_INFER', '').lower() in ('1', 'true', 'yes')
    if skip_infer and not (dump_path and os.path.isfile(dump_path)):
        raise FileNotFoundError(f"SKIP_INFER=1 but pickle not found: {dump_path}")

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
        logger.info(f"SKIP_INFER=1, loading proposals from {dump_path}")
        merged_predictions = pickle.load(open(dump_path, 'rb'))
    else:
        # gpu inference
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
            cache_path=None,
            force_cache_computation=False,
            append_token_to_batch=True,
            is_training=False
        )
        dataloader = DataLoader(dataset, **cfg.dataloader.params, shuffle=False)

        trainer = pl.Trainer(**cfg.trainer.params, callbacks=agent.get_training_callbacks())
        predictions = trainer.predict(
            AgentLightningModule(
                agent=agent,
                combined=combined
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

        pickle.dump(merged_predictions, open(dump_path, 'wb'))
        logger.info(f"Wrote proposals/subscores to {dump_path}")

    # Only ship trajectories to Ray workers (full prediction dict is multi-GB).
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
    score_rows: List[pd.DataFrame] = worker_map(worker, run_pdm_score, data_points)

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
