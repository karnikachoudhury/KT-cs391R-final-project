from dataclasses import dataclass
from collections import deque
from typing import Deque, Tuple, Optional, Dict, Any

from icm import ICM

import numpy as np
import gymnasium as gym
import torch


@dataclass
class TransitionBatch:
    obs: torch.Tensor
    action: torch.Tensor
    next_obs: torch.Tensor


class ICMIntegration(gym.Wrapper):
    def __init__(
        self,
        env: gym.Env,
        icm: ICM,
        icm_optimizer: torch.optim.Optimizer,
        *,
        lam: float = 0.1,
        use_intrinsic_reward: bool = False,
        device: str = "cpu",
        buffer_size: int = 50_000,
        icm_batch_size: int = 256,
        icm_batches_per_call: int = 2,
        grad_clip_norm: float = 0.5,
        chunk_size: int = 2000,
        # NEW: how often (in env steps) to run an ICM update inline
        icm_train_every: int = 200,
        # NEW: how many gradient steps per inline ICM update
        icm_steps_per_update: int = 10,
    ):
        super().__init__(env)
        self.icm = icm.to(device)
        self.icm_optimizer = icm_optimizer
        self.lambda_icm = float(lam)
        self.use_intrinsic_reward = use_intrinsic_reward
        self.device = device
        self.buffer: Deque[Tuple[np.ndarray, np.ndarray, np.ndarray]] = deque(maxlen=buffer_size)
        self.batch_size = int(icm_batch_size)
        self.icm_updates_per_call = int(icm_batches_per_call)
        self.grad_clip_norm = float(grad_clip_norm)
        self.previous_observation: Optional[np.ndarray] = None
        self.chunk_size = int(chunk_size)

        # Inline ICM training schedule
        self._step_counter = 0
        self._icm_train_every = int(icm_train_every)
        self._icm_steps_per_update = int(icm_steps_per_update)

        # whether success has occurred at any point in the current episode
        self.success = 0.0

    def _unwrap_to_success_env(self):
        """
        Walk through wrapper layers until we find an env that defines _check_success().
        """
        env = self.env
        visited = set()

        while env is not None and id(env) not in visited:
            visited.add(id(env))

            if hasattr(env, "_check_success") and callable(getattr(env, "_check_success")):
                return env

            env = getattr(env, "env", None)

        return None

    def _query_success(self) -> float:
        """
        Safely query robosuite task success from the underlying env.
        Returns 1.0 if successful, else 0.0.
        """
        base_env = self._unwrap_to_success_env()
        if base_env is None:
            return 0.0

        try:
            return float(base_env._check_success())
        except Exception:
            return 0.0

    @torch.no_grad()
    def compute_intrinsic_single(self, obs: np.ndarray, action: np.ndarray, next_obs: np.ndarray) -> float:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        nxt_t = torch.as_tensor(next_obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        act_t = torch.as_tensor(action, dtype=torch.float32, device=self.device).unsqueeze(0)

        r_int = self.icm.intrinsic_reward(obs_t, nxt_t, act_t)
        return float(r_int.item())

    def _train_icm_inline(self) -> None:
        """
        Run a fixed number of ICM gradient steps using random batches from
        the replay buffer. Called automatically inside step() every
        `icm_train_every` environment steps so that the ICM is always
        up-to-date by the time PPO uses the intrinsic rewards.
        """
        self.icm.train()
        for _ in range(self._icm_steps_per_update):
            batch = self.sample_batch()
            if batch is None:
                break
            self.icm_optimizer.zero_grad(set_to_none=True)
            out = self.icm.forward(batch.obs, batch.next_obs, batch.action)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(self.icm.parameters(), self.grad_clip_norm)
            self.icm_optimizer.step()
        self.icm.eval()

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        observation = observation.astype(np.float32)
        self.previous_observation = observation
        self.success = 0.0
        return observation, info

    def step(self, action: np.ndarray):
        action_np = np.asarray(action, dtype=np.float32)

        next_observation, r_ext, terminated, truncated, info = self.env.step(action_np)
        next_observation = next_observation.astype(np.float32)

        # Store transition before computing reward so the buffer is always fresh
        self.buffer.append((self.previous_observation, action_np, next_observation))
        self._step_counter += 1

        # FIX: Train ICM inline so it is up-to-date before PPO uses rewards.
        # We do this BEFORE computing r_int so the reward benefits immediately.
        if (
            self.use_intrinsic_reward
            and self._step_counter % self._icm_train_every == 0
            and len(self.buffer) >= self.batch_size
        ):
            self._train_icm_inline()

        # Compute intrinsic reward with the (freshly updated) ICM
        r_int = self.compute_intrinsic_single(self.previous_observation, action_np, next_observation)

        r_total = r_ext + self.lambda_icm * r_int if self.use_intrinsic_reward else r_ext

        info = dict(info)
        info["reward_extrinsic"] = float(r_ext)
        info["reward_intrinsic"] = float(self.lambda_icm * r_int)
        info["reward_total"] = float(r_total)
        info["icm_buffer_size"] = len(self.buffer)

        # Query success directly from the real underlying environment
        current_success = self._query_success()
        self.success = max(self.success, current_success)

        # step-level and episode-level success signals
        info["success"] = float(current_success)
        info["episode_success_so_far"] = float(self.success)

        if terminated or truncated:
            info["episode_success"] = float(self.success)

        self.previous_observation = next_observation
        return next_observation, r_total, terminated, truncated, info

    def sample_batch(self) -> Optional[TransitionBatch]:
        if len(self.buffer) < self.batch_size:
            return None

        index = np.random.randint(0, len(self.buffer), size=self.batch_size)
        obs_b = np.stack([self.buffer[i][0] for i in index], axis=0)
        act_b = np.stack([self.buffer[i][1] for i in index], axis=0)
        nxt_b = np.stack([self.buffer[i][2] for i in index], axis=0)

        return TransitionBatch(
            obs=torch.as_tensor(obs_b, dtype=torch.float32, device=self.device),
            action=torch.as_tensor(act_b, dtype=torch.float32, device=self.device),
            next_obs=torch.as_tensor(nxt_b, dtype=torch.float32, device=self.device),
        )

    def train_icm(self) -> Dict[str, Any]:
        """
        Kept for backwards compatibility but no longer called from the main
        training loop. Inline training via _train_icm_inline() inside step()
        replaces this. Calling it is now a no-op that returns empty logs.
        """
        return {}