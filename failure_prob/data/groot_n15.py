import glob
import os
import pickle
import re
from typing import Any

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from failure_prob.conf import Config

from .utils import Rollout, process_tensor_idx_rel, set_task_min_step, split_rollouts_by_seen_unseen


SAFE_ACTION_COLUMNS = [
    "action/dx",
    "action/dy",
    "action/dz",
    "action/droll",
    "action/dpitch",
    "action/dyaw",
    "action/dgripper",
]


def _as_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    return list(value)


def _extract_info_from_path(filename: str) -> tuple[int, int, bool]:
    match = re.match(r"task(\d+)--ep(\d+)--succ(\d+)\.pkl", filename)
    if match is None:
        raise ValueError(f"Filename format is incorrect: {filename}")
    task_id = int(match.group(1))
    episode_id = int(match.group(2))
    success = bool(int(match.group(3)))
    return task_id, episode_id, success


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
    if hidden_states.ndim not in (2, 3):
        raise ValueError(f"Expected hidden_states rank 2 or 3, got {hidden_states.shape}")
    return hidden_states


def _stack_attention_masks(attention_masks: Any) -> torch.Tensor | None:
    if attention_masks is None:
        return None
    if isinstance(attention_masks, list):
        if len(attention_masks) == 0:
            return None
        masks = [
            m if isinstance(m, torch.Tensor) else torch.as_tensor(m)
            for m in attention_masks
        ]
        attention_masks = torch.stack(masks, dim=0)
    elif not isinstance(attention_masks, torch.Tensor):
        attention_masks = torch.as_tensor(attention_masks)

    if attention_masks.ndim == 1:
        attention_masks = attention_masks.unsqueeze(0)
    return attention_masks.float()


def _maybe_float(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return value
    return value


def _masked_mean(hidden_states: torch.Tensor, attention_masks: torch.Tensor | None) -> torch.Tensor:
    if hidden_states.ndim == 2:
        return hidden_states
    if attention_masks is None:
        return hidden_states.mean(dim=-2)
    mask = attention_masks.to(hidden_states.device, dtype=hidden_states.dtype)
    denom = mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
    return (hidden_states * mask.unsqueeze(-1)).sum(dim=-2) / denom


def _last_valid_token(hidden_states: torch.Tensor, attention_masks: torch.Tensor | None) -> torch.Tensor:
    if hidden_states.ndim == 2:
        return hidden_states
    if attention_masks is None:
        return hidden_states[:, -1]
    masks = attention_masks.to(hidden_states.device)
    indices = masks.sum(dim=-1).long().clamp_min(1) - 1
    return hidden_states[torch.arange(hidden_states.shape[0]), indices]


def _pool_hidden_states(
    hidden_states: torch.Tensor,
    attention_masks: torch.Tensor | None,
    feature_pool: float | str,
) -> torch.Tensor:
    command = _maybe_float(feature_pool)
    if command == "masked_mean":
        return _masked_mean(hidden_states, attention_masks)
    if command == "last_valid":
        return _last_valid_token(hidden_states, attention_masks)
    return process_tensor_idx_rel(hidden_states, command)


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


def _action_vector_from_csv(csv_path: str) -> torch.Tensor | None:
    if not os.path.exists(csv_path):
        return None
    df = pd.read_csv(csv_path)
    if not all(col in df.columns for col in SAFE_ACTION_COLUMNS):
        return None
    return torch.tensor(df[SAFE_ACTION_COLUMNS].values, dtype=torch.float32)


def _load_pkl_paths(data_path: Any) -> list[str]:
    pkl_paths = []
    for path in _as_list(data_path):
        if os.path.isdir(path):
            pkl_paths.extend(glob.glob(os.path.join(path, "**", "*.pkl"), recursive=True))
        else:
            pkl_paths.extend(glob.glob(path))
    return sorted(pkl_paths)


def load_rollouts(cfg: Config) -> list[Rollout]:
    pkl_paths = _load_pkl_paths(cfg.dataset.data_path)
    all_rollouts = []

    for pkl_path in tqdm(pkl_paths, desc="Loading GR00T N1.5 rollouts"):
        with open(pkl_path, "rb") as f:
            payload = pickle.load(f)

        hidden_states = _stack_hidden_states(payload["hidden_states"])
        attention_masks = _stack_attention_masks(payload.get("attention_masks"))
        hidden_states = _pool_hidden_states(
            hidden_states,
            attention_masks,
            cfg.dataset.feature_pool,
        )

        if "action_vectors" in payload:
            action_vectors = torch.tensor(payload["action_vectors"], dtype=torch.float32)
        else:
            action_vectors = _action_vector_from_actions(
                payload.get("actions"),
                list(cfg.dataset.action_keys),
            )
            if action_vectors is None:
                action_vectors = _action_vector_from_csv(pkl_path.replace(".pkl", ".csv"))

        if action_vectors is not None:
            cfg.dataset.dim_action = action_vectors.shape[-1]
        cfg.dataset.dim_features = hidden_states.shape[-1]

        task_id, episode_id, success = _extract_info_from_path(os.path.basename(pkl_path))
        rollout = Rollout(
            hidden_states=hidden_states,
            task_suite_name=payload.get("task_suite_name", "groot_n15_robocasa"),
            task_id=int(payload.get("task_id", task_id)),
            task_description=payload.get("task_description", f"Task {task_id}"),
            episode_idx=int(payload.get("episode_idx", episode_id)),
            episode_success=int(payload.get("episode_success", success)),
            mp4_path=pkl_path.replace(".pkl", ".mp4"),
            logs=None,
            exec_horizon=cfg.dataset.exec_horizon,
            action_vectors=action_vectors,
        )
        all_rollouts.append(rollout)

    print(f"Loaded {len(all_rollouts)} GR00T N1.5 rollouts")
    return set_task_min_step(all_rollouts)


def split_rollouts(cfg: Config, all_rollouts: list[Rollout]) -> dict[str, list[Rollout]]:
    task_ids = list(set([r.task_id for r in all_rollouts]))
    n_unseen = round(cfg.dataset.unseen_task_ratio * len(task_ids))
    n_seen = len(task_ids) - n_unseen

    np.random.shuffle(task_ids)
    seen_task_ids = task_ids[:n_seen]
    unseen_task_ids = task_ids[n_seen:]

    return split_rollouts_by_seen_unseen(
        cfg, all_rollouts, seen_task_ids, unseen_task_ids
    )
