import glob
import os
import pickle
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from failure_prob.conf import Config

from .utils import Rollout, process_tensor_idx_rel, set_task_min_step, split_rollouts_by_seen_unseen


SPLIT_NAMES = ("train", "val_seen", "val_unseen")
ROLLOUT_RE = re.compile(r"task(\d+)--ep(\d+)--succ(\d+)\.pkl")


def _as_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    return list(value)


def _extract_info_from_path(filename: str) -> tuple[int, int, bool]:
    match = ROLLOUT_RE.match(filename)
    if match is None:
        raise ValueError(f"Filename format is incorrect: {filename}")
    task_id = int(match.group(1))
    episode_id = int(match.group(2))
    success = bool(int(match.group(3)))
    return task_id, episode_id, success


def _load_pkl_paths(data_path: Any) -> list[str]:
    pkl_paths = []
    for path in _as_list(data_path):
        if os.path.isdir(path):
            pkl_paths.extend(glob.glob(os.path.join(path, "**", "*.pkl"), recursive=True))
        else:
            pkl_paths.extend(glob.glob(path))
    return sorted(pkl_paths)


def _infer_split_name(path: str) -> str | None:
    parts = Path(path).parts
    for split_name in SPLIT_NAMES:
        if split_name in parts:
            return split_name
    return None


def _stack_hidden_states(hidden_states: Any) -> torch.Tensor:
    if isinstance(hidden_states, list):
        hidden_states = [
            h if isinstance(h, torch.Tensor) else torch.as_tensor(h)
            for h in hidden_states
        ]
        hidden_states = torch.stack(hidden_states, dim=0)
    elif not isinstance(hidden_states, torch.Tensor):
        hidden_states = torch.as_tensor(hidden_states)

    hidden_states = hidden_states.float()
    if hidden_states.ndim not in (2, 3, 4):
        raise ValueError(f"Expected hidden_states rank 2, 3, or 4, got {hidden_states.shape}")
    return hidden_states


def _pool_hidden_states(hidden_states: torch.Tensor, cfg: Config) -> torch.Tensor:
    if hidden_states.ndim == 4:
        # [T, K, H, D] -> [T, K, D] -> [T, D]
        hidden_states = process_tensor_idx_rel(hidden_states, cfg.dataset.horizon_idx_rel)
        hidden_states = process_tensor_idx_rel(hidden_states, cfg.dataset.diff_idx_rel)
    elif hidden_states.ndim == 3:
        # [T, N, D] -> [T, D]
        hidden_states = process_tensor_idx_rel(hidden_states, cfg.dataset.horizon_idx_rel)
    elif hidden_states.ndim != 2:
        raise ValueError(f"Expected hidden_states rank 2, 3, or 4, got {hidden_states.shape}")

    if hidden_states.ndim != 2:
        raise ValueError(f"Expected pooled hidden_states rank 2, got {hidden_states.shape}")
    return hidden_states.float()


def _first_step_vector(value: Any) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim == 0:
        return arr.reshape(1)
    if arr.ndim == 1:
        return arr
    if arr.ndim == 2:
        return arr[0].reshape(-1)
    return arr[0, 0].reshape(-1)


def _action_vector_from_actions(
    actions: list[dict[str, Any]] | None,
    action_keys: list[str],
) -> torch.Tensor | None:
    if not actions:
        return None

    vectors = []
    for action in actions:
        pieces = []
        normalized = {
            key[len("action."):] if key.startswith("action.") else key: value
            for key, value in action.items()
        }
        for key in action_keys:
            value = action.get(key)
            if value is None:
                local_key = key[len("action."):] if key.startswith("action.") else key
                value = normalized.get(local_key)
            if value is not None:
                pieces.append(_first_step_vector(value))
        if not pieces:
            return None
        vectors.append(np.concatenate(pieces, axis=0))

    return torch.tensor(np.stack(vectors, axis=0), dtype=torch.float32)


def _load_action_vectors(payload: dict[str, Any], cfg: Config) -> torch.Tensor | None:
    if "action_vectors" in payload:
        return torch.tensor(payload["action_vectors"], dtype=torch.float32)
    return _action_vector_from_actions(
        payload.get("actions"),
        list(cfg.dataset.action_keys),
    )


def _set_dynamic_dataset_dims(cfg: Config, payload: dict[str, Any], hidden_states: torch.Tensor, action_vectors: torch.Tensor | None) -> None:
    cfg.dataset.dim_features = hidden_states.shape[-1]
    if action_vectors is not None:
        cfg.dataset.dim_action = action_vectors.shape[-1]

    valid_horizon = payload.get("valid_action_horizon")
    if valid_horizon is not None:
        cfg.dataset.pred_horizon = int(valid_horizon)
        cfg.dataset.exec_horizon = int(valid_horizon)


def load_rollouts(cfg: Config) -> list[Rollout]:
    pkl_paths = _load_pkl_paths(cfg.dataset.data_path)
    all_rollouts = []

    for pkl_path in tqdm(pkl_paths, desc="Loading GR00T N1.6 rollouts"):
        with open(pkl_path, "rb") as f:
            payload = pickle.load(f)

        hidden_states = _stack_hidden_states(payload["hidden_states"])
        hidden_states = _pool_hidden_states(hidden_states, cfg)

        action_vectors = _load_action_vectors(payload, cfg)
        _set_dynamic_dataset_dims(cfg, payload, hidden_states, action_vectors)

        task_id, episode_id, success = _extract_info_from_path(os.path.basename(pkl_path))
        rollout = Rollout(
            hidden_states=hidden_states,
            task_suite_name=payload.get("task_suite_name", "groot_n16_robocasa"),
            task_id=int(payload.get("task_id", task_id)),
            task_description=payload.get("task_description", f"Task {task_id}"),
            episode_idx=int(payload.get("episode_idx", episode_id)),
            episode_success=int(payload.get("episode_success", success)),
            mp4_path=pkl_path.replace(".pkl", ".mp4"),
            logs=None,
            exec_horizon=cfg.dataset.exec_horizon,
            action_vectors=action_vectors,
        )
        rollout.split_name = _infer_split_name(pkl_path)
        all_rollouts.append(rollout)

    print(f"Loaded {len(all_rollouts)} GR00T N1.6 rollouts")
    return set_task_min_step(all_rollouts)


def _split_rollouts_from_physical_dirs(all_rollouts: list[Rollout]) -> dict[str, list[Rollout]] | None:
    if not all(hasattr(r, "split_name") and r.split_name in SPLIT_NAMES for r in all_rollouts):
        return None

    rollouts_by_split_name = {split_name: [] for split_name in SPLIT_NAMES}
    for rollout in all_rollouts:
        rollouts_by_split_name[rollout.split_name].append(rollout)

    for split_name in list(rollouts_by_split_name):
        if len(rollouts_by_split_name[split_name]) == 0:
            del rollouts_by_split_name[split_name]
            continue
        rollouts_by_split_name[split_name] = sorted(
            rollouts_by_split_name[split_name],
            key=lambda r: (r.task_id, r.episode_idx),
        )

    return rollouts_by_split_name


def _print_split_summary(rollouts_by_split_name: dict[str, list[Rollout]]) -> None:
    for split, rollouts in rollouts_by_split_name.items():
        n_success = sum([r.episode_success for r in rollouts])
        n_fail = len(rollouts) - n_success
        print(f"{split}: {len(rollouts)} rollouts, {n_success} success, {n_fail} fail")

        task_ids = sorted({r.task_id for r in rollouts})
        for task_id in task_ids:
            task_rollouts = [r for r in rollouts if r.task_id == task_id]
            task_success = sum([r.episode_success for r in task_rollouts])
            task_fail = len(task_rollouts) - task_success
            print(f"  task{task_id}: {len(task_rollouts)} rollouts, {task_success} success, {task_fail} fail")


def split_rollouts(cfg: Config, all_rollouts: list[Rollout]) -> dict[str, list[Rollout]]:
    rollouts_by_split_name = _split_rollouts_from_physical_dirs(all_rollouts)
    if rollouts_by_split_name is not None:
        print("Using physical GR00T N1.6 split directories")
        _print_split_summary(rollouts_by_split_name)
        return rollouts_by_split_name

    task_ids = list(set([r.task_id for r in all_rollouts]))
    n_unseen = round(cfg.dataset.unseen_task_ratio * len(task_ids))
    n_seen = len(task_ids) - n_unseen

    np.random.shuffle(task_ids)
    seen_task_ids = task_ids[:n_seen]
    unseen_task_ids = task_ids[n_seen:]

    return split_rollouts_by_seen_unseen(
        cfg, all_rollouts, seen_task_ids, unseen_task_ids
    )
