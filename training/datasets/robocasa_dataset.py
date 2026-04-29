"""
robocasa_dataset.py

PyTorch Dataset and Collator for RoboCasa HDF5 demonstrations, compatible with CogACT's
VLA action prediction training pipeline.

RoboCasa stores data in robomimic-compatible HDF5 files. Each file contains multiple
demonstrations, each with image observations and 7-DoF robot actions.

Expected HDF5 structure (per demo):
    data/demo_N/
        actions          [T, 7]  — delta EEF (x,y,z, roll,pitch,yaw, gripper)
        obs/
            robot0_agentview_left_image   [T, H, W, 3]
            robot0_agentview_right_image  [T, H, W, 3]
            robot0_eye_in_hand_image      [T, H, W, 3]
            robot0_eef_pos    [T, 3]
            robot0_eef_quat   [T, 4]
            robot0_gripper_qpos [T, 2]
        attrs:
            ep_meta  (JSON string with 'lang' key in RoboCasa v0.1+)
"""

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from transformers import LlamaTokenizerFast

IGNORE_INDEX = -100


class RoboCasaDataset(Dataset):
    """
    Map-style Dataset over RoboCasa HDF5 demonstration files.

    Each sample is a (image_t, tokenized_instruction, action_chunk_{t..t+H}) triple.
    The batch format matches CogACT's VLA forward() signature exactly.
    """

    # Possible suffixes for camera observation keys in robomimic HDF5 files
    _CAMERA_KEY_SUFFIXES = ("_image", "")

    def __init__(
        self,
        data_root: str,
        image_transform: Callable,
        tokenizer,
        prompt_builder_fn: Callable,
        future_action_window_size: int = 15,
        past_action_window_size: int = 0,
        camera_name: str = "robot0_agentview_left",
        action_norm_stats: Optional[Dict] = None,
    ) -> None:
        """
        Args:
            data_root: Path to a single .hdf5 file or a directory tree containing .hdf5 files.
            image_transform: Vision backbone's image transform (from get_image_transform()).
            tokenizer: LLM tokenizer.
            prompt_builder_fn: Callable that returns a fresh PromptBuilder instance.
            future_action_window_size: Number of future steps to predict (default 15).
            past_action_window_size: Number of past steps as context (default 0 = disabled).
            camera_name: Base camera name to use for observations.
            action_norm_stats: Pre-computed {"q01", "q99", "mask"} dict. If None, computed
                               from the dataset on initialisation (requires loading all actions).
        """
        self.image_transform = image_transform
        self.tokenizer = tokenizer
        self.prompt_builder_fn = prompt_builder_fn
        self.future_window = future_action_window_size
        self.past_window = past_action_window_size
        self.camera_name = camera_name

        data_root = Path(data_root)
        if data_root.is_dir():
            hdf5_paths = sorted(data_root.glob("**/*.hdf5"))
        else:
            hdf5_paths = [data_root]
        assert hdf5_paths, f"No .hdf5 files found at {data_root}"

        # Build flat index: (hdf5_path, demo_key, start_timestep)
        self.samples: List[Tuple[str, str, int]] = []
        all_actions_for_stats: List[np.ndarray] = []

        for hdf5_path in hdf5_paths:
            with h5py.File(str(hdf5_path), "r") as f:
                demos = sorted(f["data"].keys(), key=lambda k: int(k.split("_")[-1]))
                for demo_key in demos:
                    demo = f["data"][demo_key]
                    T = demo["actions"].shape[0]
                    if action_norm_stats is None:
                        all_actions_for_stats.append(demo["actions"][:])
                    # Need at least future_window + 1 steps starting from t
                    for t in range(T - self.future_window):
                        self.samples.append((str(hdf5_path), demo_key, t))

        if action_norm_stats is None:
            all_actions = np.concatenate(all_actions_for_stats, axis=0)
            q01 = np.quantile(all_actions, 0.01, axis=0)
            q99 = np.quantile(all_actions, 0.99, axis=0)
            self.action_norm_stats: Dict = {
                "q01": q01.tolist(),
                "q99": q99.tolist(),
                "mask": [True] * all_actions.shape[1],
            }
        else:
            self.action_norm_stats = action_norm_stats

    @property
    def dataset_statistics(self) -> Dict:
        """Returns statistics in the format expected by CogACT's dataset_statistics.json."""
        return {"robocasa": {"action": self.action_norm_stats}}

    def __len__(self) -> int:
        return len(self.samples)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_task_instruction(self, demo_grp: h5py.Group) -> str:
        """Extract a natural-language instruction, falling back gracefully."""
        # RoboCasa v0.1+: ep_meta JSON attr on each demo
        if "ep_meta" in demo_grp.attrs:
            try:
                ep_meta = json.loads(demo_grp.attrs["ep_meta"])
                if "lang" in ep_meta:
                    return ep_meta["lang"]
            except (json.JSONDecodeError, TypeError):
                pass

        # Direct string attrs (various naming conventions)
        for key in ("lang", "language_instruction", "task_description", "annotation"):
            if key in demo_grp.attrs:
                val = demo_grp.attrs[key]
                return val.decode() if isinstance(val, (bytes, np.bytes_)) else str(val)

        # File-level env name → human-readable fallback
        f = demo_grp.file
        env_name = f["data"].attrs.get("env", "")
        if env_name:
            return str(env_name).replace("_", " ").lower()

        return "perform the manipulation task"

    def _get_camera_key(self, obs_grp: h5py.Group) -> str:
        """Resolve the HDF5 key for the requested camera, tolerating suffix differences."""
        for suffix in self._CAMERA_KEY_SUFFIXES:
            key = f"{self.camera_name}{suffix}"
            if key in obs_grp:
                return key
        available = list(obs_grp.keys())
        raise KeyError(
            f"Camera '{self.camera_name}' not in obs group. "
            f"Available keys: {available}"
        )

    def _normalize(self, actions: np.ndarray) -> np.ndarray:
        q01 = np.array(self.action_norm_stats["q01"])
        q99 = np.array(self.action_norm_stats["q99"])
        mask = np.array(self.action_norm_stats.get("mask", [True] * actions.shape[-1]), dtype=bool)
        # Linear map [q01, q99] -> [-1, 1] for masked dimensions
        normalized = np.where(
            mask,
            2.0 * (actions - q01) / (q99 - q01 + 1e-8) - 1.0,
            actions,
        )
        return np.clip(normalized, -1.0, 1.0).astype(np.float32)

    # ------------------------------------------------------------------
    # Dataset access
    # ------------------------------------------------------------------

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        hdf5_path, demo_key, t = self.samples[idx]

        with h5py.File(hdf5_path, "r") as f:
            demo = f["data"][demo_key]
            obs = demo["obs"]

            # Image at timestep t
            cam_key = self._get_camera_key(obs)
            img_np: np.ndarray = obs[cam_key][t]          # [H, W, 3] uint8
            instruction: str = self._get_task_instruction(demo)

            # Action chunk: t → t + future_window (inclusive), shape [W+1, action_dim]
            window = self.future_window + 1
            raw_future = demo["actions"][t: t + window].astype(np.float32)
            if raw_future.shape[0] < window:
                pad = np.zeros((window - raw_future.shape[0], raw_future.shape[1]), dtype=np.float32)
                raw_future = np.concatenate([raw_future, pad], axis=0)

            # Past action context (zero-padded at episode start)
            if self.past_window > 0:
                t_start = max(0, t - self.past_window)
                raw_past = demo["actions"][t_start:t].astype(np.float32)
                if raw_past.shape[0] < self.past_window:
                    pad = np.zeros(
                        (self.past_window - raw_past.shape[0], raw_past.shape[1]), dtype=np.float32
                    )
                    raw_past = np.concatenate([pad, raw_past], axis=0)
                norm_past = self._normalize(raw_past)
            else:
                norm_past = np.zeros((0, raw_future.shape[1]), dtype=np.float32)

        # Normalize and concatenate [past | future]
        norm_future = self._normalize(raw_future)
        all_actions = np.concatenate([norm_past, norm_future], axis=0)  # [past+W+1, D]

        # Image → tensor via vision backbone transform
        image = Image.fromarray(img_np)
        pixel_values = self.image_transform(image)

        # Tokenize the instruction prompt (same format as CogACT pretraining)
        prompt_builder = self.prompt_builder_fn()
        prompt_builder.add_turn(
            role="human",
            message=f"What action should the robot take to {instruction.lower()}?",
        )
        prompt_text = prompt_builder.get_prompt()
        input_ids = self.tokenizer(
            prompt_text, truncation=True, return_tensors="pt"
        ).input_ids.squeeze(0)

        # Llama tokenizer: append '' (29871) + EOS (2) to match training-time format
        if isinstance(self.tokenizer, LlamaTokenizerFast):
            input_ids = torch.cat(
                [input_ids, torch.tensor([29871, 2], dtype=torch.long)]
            )

        # Labels: IGNORE everything; only the last (EOS/cognition) token is a target
        labels = torch.full_like(input_ids, IGNORE_INDEX)
        labels[-1] = input_ids[-1]

        attention_mask = torch.ones_like(input_ids, dtype=torch.bool)

        return {
            "input_ids": input_ids,                               # [seq_len]
            "attention_mask": attention_mask,                      # [seq_len]
            "labels": labels,                                      # [seq_len]
            "pixel_values": pixel_values,                          # [C, H, W] or dict
            "actions": torch.from_numpy(all_actions),              # [past+W+1, action_dim]
            "action_masks": torch.ones(window, dtype=torch.bool),  # [W+1]
        }


class RoboCasaCollator:
    """
    Batch collator for RoboCasaDataset.

    Pads variable-length token sequences and stacks fixed-size tensors to produce
    batches with the keys expected by CogACT.forward().
    """

    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = pad_token_id

    def __call__(self, instances: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_ids = [inst["input_ids"] for inst in instances]
        labels = [inst["labels"] for inst in instances]
        attn_masks = [inst["attention_mask"] for inst in instances]

        max_len = max(ids.shape[0] for ids in input_ids)

        padded_ids = torch.stack([
            F.pad(ids, (0, max_len - ids.shape[0]), value=self.pad_token_id)
            for ids in input_ids
        ])
        padded_labels = torch.stack([
            F.pad(lb, (0, max_len - lb.shape[0]), value=IGNORE_INDEX)
            for lb in labels
        ])
        padded_masks = torch.stack([
            F.pad(m.long(), (0, max_len - m.shape[0]), value=0).bool()
            for m in attn_masks
        ])

        # pixel_values can be a plain tensor or a dict of tensors (dual-encoder backbones)
        pv0 = instances[0]["pixel_values"]
        if isinstance(pv0, torch.Tensor):
            pixel_values = torch.stack([inst["pixel_values"] for inst in instances])
        elif isinstance(pv0, dict):
            pixel_values = {
                k: torch.stack([inst["pixel_values"][k] for inst in instances])
                for k in pv0
            }
        else:
            raise ValueError(f"Unsupported pixel_values type: {type(pv0)}")

        return {
            "input_ids": padded_ids,
            "attention_mask": padded_masks,
            "labels": padded_labels,
            "pixel_values": pixel_values,
            "actions": torch.stack([inst["actions"] for inst in instances]),
            "action_masks": torch.stack([inst["action_masks"] for inst in instances]),
        }
