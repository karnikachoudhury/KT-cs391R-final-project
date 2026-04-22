from dataclasses import dataclass
from collections import deque
from typing import Deque, Tuple, Optional

from icm import ICM

import numpy as np
import gymnasium as gym
import torch
import math


@dataclass
class TransitionBatch:
    obs: torch.Tensor
    action: torch.Tensor
    next_obs: torch.Tensor


def unwrap_fully(env):
    """Unwrap all wrapper layers until we reach the base env."""
    while hasattr(env, "env"):
        env = env.env
    return env


def get_decay_func(decay_type, init_lambda, half_life=200_000):
    expfunc = lambda step: init_lambda * math.pow(2, -step / half_life)
    linfunc = lambda step: max(0.0, -step / 750_000 + 1)
    sinfunc = lambda step: 0.005 * math.sin(step / 10_000) + 0.005

    if decay_type == "linear":
        return lambda step: init_lambda * linfunc(step)
    elif decay_type == "exp":
        return expfunc
    elif decay_type == "sin":
        return lambda step: linfunc(step) * sinfunc(step)
    else:
        return None


class ICMIntegration(gym.Wrapper):
    """
    Gym wrapper that adds ICM-based intrinsic reward to any environment.

    Stack order (outermost -> innermost):
        VecNormalize -> DummyVecEnv -> ICMIntegration -> Monitor -> base env
    """

    def __init__(
        self,
        env: gym.Env,
        icm: ICM,
        icm_optimizer: torch.optim.Optimizer,
        *,
        lam: float = 0.001,
        use_intrinsic_reward: bool = True,
        device: str = "cpu",
        buffer_size: int = 50_000,
        icm_batch_size: int = 256,
        grad_clip_norm: float = 0.5,
        icm_train_every: int = 100,
        icm_steps_per_update: int = 5,
        train_icm: bool = True,
        r_int_clip: float = 0.05,
        delay: int = 0,
        decay_type=None,
        n_envs: int = 4,
    ):
        super().__init__(env)

        self.icm = icm.to(device)
        self.icm_optimizer = icm_optimizer
        self.lambda_icm = float(lam)
        self.use_intrinsic_reward = bool(use_intrinsic_reward)
        self.device = device
        self.train_icm = bool(train_icm)
        self.r_int_clip = float(r_int_clip)

        self.buffer: Deque[Tuple[np.ndarray, np.ndarray, np.ndarray]] = deque(
            maxlen=buffer_size
        )
        self.batch_size = int(icm_batch_size)
        self.grad_clip_norm = float(grad_clip_norm)

        self.previous_observation: Optional[np.ndarray] = None

        self._step_counter = 0
        self.delay = int(delay)
        self._icm_train_every = int(icm_train_every)
        self._icm_steps_per_update = int(icm_steps_per_update)

        # Welford online normalization for intrinsic reward
        self._r_int_n: int = 0
        self._r_int_mean: float = 0.0
        self._r_int_M2: float = 0.0

        self.success = 0.0
        self.decay = get_decay_func(decay_type, self.lambda_icm)
        self.n_envs = n_envs

    # ------------------------------------------------------------------
    # Success detection
    # ------------------------------------------------------------------

    def _query_success_from_info(self, info) -> float:
        if info is None:
            return 0.0
        if "success" in info:
            return float(info["success"])
        if "is_success" in info:
            return float(info["is_success"])
        if "episode_success" in info:
            return float(info["episode_success"])
        if "episode_success_so_far" in info:
            return float(info["episode_success_so_far"])
        return 0.0

    def _query_success_fallback(self) -> float:
        """
        Fallback for envs that expose an internal success checker.
        Works with Robosuite-style environments; harmless for Meta-World.
        """
        try:
            base = unwrap_fully(self.env)
            if hasattr(base, "_check_success"):
                return float(base._check_success())
        except Exception:
            pass
        return 0.0

    # ------------------------------------------------------------------
    # ICM internals
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _compute_intrinsic_single(
        self, obs: np.ndarray, action: np.ndarray, next_obs: np.ndarray
    ) -> float:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        nxt_t = torch.as_tensor(next_obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        act_t = torch.as_tensor(action, dtype=torch.float32, device=self.device).unsqueeze(0)
        return float(self.icm.intrinsic_reward(obs_t, nxt_t, act_t).item())

    def _normalize_r_int(self, r_int: float) -> float:
        self._r_int_n += 1
        delta = r_int - self._r_int_mean
        self._r_int_mean += delta / self._r_int_n
        delta2 = r_int - self._r_int_mean
        self._r_int_M2 += delta * delta2

        variance = self._r_int_M2 / max(self._r_int_n - 1, 1)
        std = max(variance ** 0.5, 1e-8)
        return r_int / std

    def _sample_batch(self) -> Optional[TransitionBatch]:
        if len(self.buffer) < self.batch_size:
            return None

        idx = np.random.randint(0, len(self.buffer), size=self.batch_size)
        obs_b = np.stack([self.buffer[i][0] for i in idx])
        act_b = np.stack([self.buffer[i][1] for i in idx])
        nxt_b = np.stack([self.buffer[i][2] for i in idx])

        return TransitionBatch(
            obs=torch.as_tensor(obs_b, dtype=torch.float32, device=self.device),
            action=torch.as_tensor(act_b, dtype=torch.float32, device=self.device),
            next_obs=torch.as_tensor(nxt_b, dtype=torch.float32, device=self.device),
        )

    def _train_icm_inline(self) -> None:
        self.icm.train()

        for _ in range(self._icm_steps_per_update):
            batch = self._sample_batch()
            if batch is None:
                break

            self.icm_optimizer.zero_grad(set_to_none=True)
            out = self.icm.forward(batch.obs, batch.next_obs, batch.action)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(self.icm.parameters(), self.grad_clip_norm)
            self.icm_optimizer.step()

        self.icm.eval()

    # ------------------------------------------------------------------
    # Gym interface
    # ------------------------------------------------------------------

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        observation = np.asarray(observation, dtype=np.float32)
        self.previous_observation = observation
        self.success = 0.0
        return observation, info

    def step(self, action: np.ndarray):
        action_np = np.asarray(action, dtype=np.float32)
        next_observation, r_ext, terminated, truncated, info = self.env.step(action_np)
        next_observation = np.asarray(next_observation, dtype=np.float32)
        info = dict(info)

        current_success = self._query_success_from_info(info)
        if current_success == 0.0:
            current_success = self._query_success_fallback()

        # Baseline path
        if (not self.use_intrinsic_reward) or (self.lambda_icm == 0.0):
            self.success = max(self.success, current_success)

            info["reward_extrinsic"] = float(r_ext)
            info["reward_intrinsic"] = 0.0
            info["reward_total"] = float(r_ext)
            info["reward_intrinsic_raw"] = 0.0
            info["reward_intrinsic_normalized"] = 0.0
            info["icm_buffer_size"] = len(self.buffer)
            info["success"] = float(current_success)
            info["episode_success_so_far"] = float(self.success)

            if terminated or truncated:
                info["episode_success"] = float(self.success)

            self.previous_observation = next_observation
            return next_observation, float(r_ext), terminated, truncated, info

        # ICM path
        self.buffer.append((self.previous_observation, action_np, next_observation))
        self._step_counter += 1

        if (
            self.train_icm
            and self._step_counter % self._icm_train_every == 0
            and len(self.buffer) >= self.batch_size
        ):
            self._train_icm_inline()

        r_int_raw = self._compute_intrinsic_single(
            self.previous_observation, action_np, next_observation
        )
        r_int_norm = self._normalize_r_int(r_int_raw)

        ext_scale = 1.0
        if self.decay is not None:
            self.lambda_icm = self.decay(self._step_counter * self.n_envs)
            ext_scale -= self.lambda_icm

        r_intrinsic_scaled = self.lambda_icm * float(r_int_norm)
        r_intrinsic_scaled = float(
            np.clip(r_intrinsic_scaled, -self.r_int_clip, self.r_int_clip)
        )

        if self._step_counter >= self.delay:
            r_total = ext_scale * float(r_ext) + r_intrinsic_scaled
        else:
            r_total = float(r_ext)

        self.success = max(self.success, current_success)

        info["reward_extrinsic"] = float(r_ext)
        info["reward_intrinsic"] = float(r_intrinsic_scaled)
        info["reward_total"] = float(r_total)
        info["reward_intrinsic_raw"] = float(r_int_raw)
        info["reward_intrinsic_normalized"] = float(r_int_norm)
        info["icm_buffer_size"] = len(self.buffer)
        info["lambda"] = self.lambda_icm

        info["success"] = float(current_success)
        info["episode_success_so_far"] = float(self.success)

        if terminated or truncated:
            info["episode_success"] = float(self.success)

        self.previous_observation = next_observation
        return next_observation, float(r_total), terminated, truncated, info