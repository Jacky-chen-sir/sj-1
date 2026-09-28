# SPDX-License-Identifier: Apache-2.0
"""抽取逐场景的属性表，供挑战子集分解（#9）与分桶 bootstrap（#10）使用。

只用 metric cache，不需要模型前向，也不需要 GPU。跑一次落盘 `scene_attributes.csv`，
之后所有分桶分析都复用它。

属性定义（都取自 metric cache，可复现、无模型依赖）：

  n_objects / n_vehicle / n_pedestrian / n_bicycle
      场景中被他车占据的 track 数，按 `TrackedObjectType` 分类（PDMObservation.unique_objects）。
  has_vru
      n_pedestrian + n_bicycle > 0。
  is_intersection
      route 中包含 lane connector（`route_lane_ids` 里任一 id 在图上查得到 LANE_CONNECTOR）。
      这是"路口"的标准定义：路口在 nuPlan 图里就是 connector。
  ego_speed
      当前帧自车速度 [m/s]。
  lat_max / lat_end / is_lane_change
      人类轨迹在自车初始坐标系下的横向偏移（最大 / 末端）。`is_lane_change = lat_max > 3.5 m`
      （一个标准车道宽）。这是"换道"的几何定义，不依赖地图。
  route_len
      route 的车道数，作为路径复杂度代理。

用法：

    python navsim/planning/script/run_scene_attributes.py \
        train_test_split=navtest \
        ++attrs.out=$NAVSIM_EXP_ROOT/scene_attributes.csv \
        ++attrs.workers=16
"""

import logging
import lzma
import os
import pickle
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import hydra
import numpy as np
import pandas as pd
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.common.geometry.convert import convert_absolute_to_relative_se2_array
from nuplan.common.maps.maps_datatypes import SemanticMapLayer
from nuplan.common.maps.nuplan_map.map_factory import get_maps_api
from nuplan.planning.script.builders.logging_builder import build_logger
from omegaconf import DictConfig

from navsim.common.dataloader import MetricCacheLoader

logger = logging.getLogger(__name__)

CONFIG_PATH = "config/pdm_scoring"
CONFIG_NAME = "default_run_pdm_score_gpu"

LANE_CHANGE_LAT_M = 3.5   # 一个标准车道宽 [m]

_MAP_API_CACHE: Dict[str, object] = {}


def _map_api(map_root: str, map_version: str, map_name: str):
    """每个 worker 进程内按 map_name 缓存一份地图句柄（地图很重，不能每 token 重建）。"""
    key = f"{map_root}|{map_version}|{map_name}"
    if key not in _MAP_API_CACHE:
        _MAP_API_CACHE[key] = get_maps_api(map_root, map_version, map_name)
    return _MAP_API_CACHE[key]


def _is_intersection(route_lane_ids: List[str], map_root: str, map_version: str,
                     map_name: str) -> Optional[bool]:
    if not route_lane_ids:
        return None
    try:
        api = _map_api(map_root, map_version, map_name)
    except Exception:
        logger.warning("地图 %s 加载失败，is_intersection 记为 NA", map_name)
        return None
    for lane_id in route_lane_ids:
        try:
            if api.get_map_object(str(lane_id), SemanticMapLayer.LANE_CONNECTOR) is not None:
                return True
        except Exception:
            continue
    return False


def _ego_speed(cache) -> Optional[float]:
    try:
        return float(cache.ego_state.dynamic_car_state.speed)
    except Exception:
        pass
    try:
        return float(cache.ego_state.rear_axle_velocity_2d.magnitude())
    except Exception:
        return None


def _relative_poses(cache) -> Optional[np.ndarray]:
    """人类轨迹在自车初始坐标系下的位姿 [T,3]。"""
    ht = getattr(cache, "human_trajectory", None)
    poses = getattr(ht, "poses", None) if ht is not None else None
    if poses is not None:
        return np.asarray(poses, dtype=np.float64)
    try:
        states = cache.trajectory.get_sampled_trajectory()
        abs_poses = np.array([[s.rear_axle.x, s.rear_axle.y, s.rear_axle.heading] for s in states])
        return convert_absolute_to_relative_se2_array(cache.ego_state.rear_axle, abs_poses)
    except Exception:
        return None


def _object_counts(cache) -> Dict[str, int]:
    counts = {"n_vehicle": 0, "n_pedestrian": 0, "n_bicycle": 0, "n_generic": 0}
    try:
        objects = cache.observation.unique_objects
    except Exception:
        return counts
    for obj in objects.values():
        t = obj.tracked_object_type
        if t == TrackedObjectType.VEHICLE:
            counts["n_vehicle"] += 1
        elif t == TrackedObjectType.PEDESTRIAN:
            counts["n_pedestrian"] += 1
        elif t == TrackedObjectType.BICYCLE:
            counts["n_bicycle"] += 1
        else:
            counts["n_generic"] += 1
    return counts


def _extract(task: Tuple[str, str]) -> Optional[Dict]:
    token, path = task
    try:
        with lzma.open(path, "rb") as f:
            cache = pickle.load(f)
    except Exception:
        logger.warning("metric cache 读取失败: %s", token)
        return None

    row: Dict = {"token": token}
    try:
        row["log_name"] = cache.log_name
        mp = cache.map_parameters
        row["map_name"] = mp.map_name
        row["n_objects_route"] = len(cache.route_lane_ids or [])
        row["is_intersection"] = _is_intersection(
            list(cache.route_lane_ids or []), mp.map_root, mp.map_version, mp.map_name)

        row.update(_object_counts(cache))
        row["n_objects"] = row["n_vehicle"] + row["n_pedestrian"] + row["n_bicycle"] + row["n_generic"]
        row["has_vru"] = (row["n_pedestrian"] + row["n_bicycle"]) > 0

        row["ego_speed"] = _ego_speed(cache)

        poses = _relative_poses(cache)
        if poses is not None and len(poses) > 0:
            lat = poses[:, 1]
            row["lat_max"] = float(np.max(np.abs(lat)))
            row["lat_end"] = float(lat[-1])
            row["traj_len"] = int(len(poses))
            row["is_lane_change"] = bool(row["lat_max"] > LANE_CHANGE_LAT_M)
        else:
            row["lat_max"] = row["lat_end"] = row["traj_len"] = None
            row["is_lane_change"] = None
    except Exception:
        logger.exception("属性抽取失败: %s", token)
        return None
    return row


def _worker_init():
    """子进程里静音 nuplan 的冗长日志。"""
    logging.getLogger("nuplan").setLevel(logging.ERROR)


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    build_logger(cfg)

    attrs_cfg = cfg.get("attrs", {})
    out_path = Path(str(attrs_cfg.get("out", "scene_attributes.csv")))
    workers = int(attrs_cfg.get("workers", max(1, (os.cpu_count() or 4) // 2)))
    limit = attrs_cfg.get("limit", None)

    loader = MetricCacheLoader(Path(cfg.metric_cache_path))
    tasks = list(loader.metric_cache_paths.items())
    if limit:
        tasks = tasks[: int(limit)]
    logger.info(f"抽取 {len(tasks)} 个场景的属性 → {out_path}（{workers} 进程）")

    rows = []
    with ProcessPoolExecutor(max_workers=workers, initializer=_worker_init) as ex:
        for i, row in enumerate(ex.map(_extract, tasks, chunksize=32)):
            if row is not None:
                rows.append(row)
            if (i + 1) % 2000 == 0:
                logger.info(f"  {i + 1}/{len(tasks)} ...")

    df = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    n_ok = len(df)
    logger.info(
        f"""
        场景属性抽取完成: {n_ok}/{len(tasks)} 成功 → {out_path}
          has_vru        : {int(df['has_vru'].sum()) if 'has_vru' in df else 'NA'} 个场景
          is_intersection: {int(df['is_intersection'].fillna(False).sum()) if 'is_intersection' in df else 'NA'} 个场景
          is_lane_change : {int(df['is_lane_change'].fillna(False).sum()) if 'is_lane_change' in df else 'NA'} 个场景
          n_objects 分位 : {df['n_objects'].quantile([0.25, 0.5, 0.75]).round(1).to_dict() if 'n_objects' in df else 'NA'}
        """
    )


if __name__ == "__main__":
    main()
