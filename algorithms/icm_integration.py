from dataclasses import dataclass

from icm import ICM
from typing import Deque, Tuple, Optional, Dict, Any

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
        lam: float = 0.1, # scales intrinsic reward
        use_intrinsic_reward: bool = False,  
        device: str = "cpu",
        buffer_size: int = 50_000,
        icm_batch_size: int = 256, # how many samples to use when updating ICM parameters
        icm_batches_per_call: int = 2, # how many batches to train ICM on per train_icm call       
        grad_clip_norm: float = 0.5,
        chunk_size: int = 2000,  # number of steps to run PPO before updating ICM
    ):
        super().__init__(env)
        self.icm = icm.to(device)
        self.icm_optimizer = icm_optimizer
        self.lambda_icm = float(lam)
        self.use_intrinsic_reward = use_intrinsic_reward
        self.device = device
        self.buffer: Deque[Tuple[np.ndarray, np.ndarray, np.ndarray]] = Deque(maxlen=buffer_size)
        self.batch_size = int(icm_batch_size)
        self.icm_updates_per_call = int(icm_batches_per_call)
        self.grad_clip_norm = float(grad_clip_norm)
        self.previous_observation: Optional[np.ndarray] = None
        self.chunk_size = int(chunk_size)
        self.success = -1 # have we had success in current episode? -1 if not set

    @torch.no_grad()
    def compute_instrinsic_single(self, obs: np.ndarray, action: np.ndarray, next_obs: np.ndarray) -> float:

        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)      
        nxt_t = torch.as_tensor(next_obs, dtype=torch.float32, device=self.device).unsqueeze(0)  
        act_t = torch.as_tensor(action, dtype=torch.float32, device=self.device).unsqueeze(0)   

        r_int = self.icm.intrinsic_reward(obs_t, nxt_t, act_t)  
        return float(r_int.item())
    
    # resets environment variables and defined types
    def reset(self, **kwargs):
        observation, info  = self.env.reset(**kwargs)
        observation = observation.astype(np.float32)
        self.previous_observation = observation
        if self.success == -1:
            self.success = 0
        elif self.success == 1:
            print("Success")
            self.success = 0
        else:
            print("Failure")
        return observation, info
    
    # step function that computes intrinsic reward and stores transitions in buffer
    def step(self, action: np.ndarray):
        action_np = np.asarray(action, dtype=np.float32)

        next_observation, r_ext, terminated, truncated, info = self.env.step(action_np)
        next_observation = next_observation.astype(np.float32)

        # calculate intrinsic reward and store it in buffer
        r_int = self.compute_instrinsic_single(self.previous_observation, action_np, next_observation)
        self.buffer.append((self.previous_observation, action_np, next_observation))

        # define PPO reward function !!!!! will be changing the weighting here and stuff
        r_total = r_ext + self.lambda_icm * r_int if self.use_intrinsic_reward else r_ext

        # logging for tensor board
        info = dict(info)
        info["reward_extrinsic"] = float(r_ext)
        info["reward_intrinsic"] = float(self.lambda_icm * r_int)
        info["reward_total"] = float(r_ext + self.lambda_icm * r_int)  # total even if PPO isn't using it yet
        info["icm_buffer_size"] = len(self.buffer)
        info["success"] = float(self.env.env.env._check_success()) 
        if info["success"]:
            self.success = 1

        self.previous_observation = next_observation
        return next_observation, r_total, terminated, truncated, info
    
    # get random batch from buffer and update ICM parameters
    def sample_batch(self) -> Optional[TransitionBatch]:
        if(len(self.buffer) < self.batch_size):
            return None
        
        index = np.random.randint(0, len(self.buffer), size=self.batch_size)
        obs_b = np.stack([self.buffer[i][0] for i in index], axis=0)      
        act_b = np.stack([self.buffer[i][1] for i in index], axis=0)      
        nxt_b = np.stack([self.buffer[i][2] for i in index], axis=0)  

        return TransitionBatch(
            obs = torch.as_tensor(obs_b, dtype=torch.float32, device=self.device),
            action = torch.as_tensor(act_b, dtype=torch.float32, device=self.device),
            next_obs = torch.as_tensor(nxt_b, dtype=torch.float32, device=self.device),
        )
    
    def train_icm(self) -> Dict[str, Any]:
        print(f"len buffer: {len(self.buffer)}")
        logs: Dict[str, Any] = {}
        if not self.use_intrinsic_reward:
            logs["icm_train_skipped"] = 1.0
            return logs

        batch = self.sample_batch()
        if batch is None:
            logs["icm_loss"] = 1.0
            return logs
        self.icm.train()

        # keep logging as we go through a few steps
        icm_loss_vals = []
        inverse_loss_vals = []
        forward_loss_vals = []
        r_int_mean_vals = []

        for _ in range(self.icm_updates_per_call):
            batch = self.sample_batch()
            if batch is None:
                break
            self.icm_optimizer.zero_grad(set_to_none=True)
            out = self.icm.forward(batch.obs, batch.next_obs, batch.action)

            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(self.icm.parameters(), self.grad_clip_norm)
            self.icm_optimizer.step()

            icm_loss_vals.append(float(out.info["icm_loss"].item()))
            inverse_loss_vals.append(float(out.info["inv_loss"].item()))
            forward_loss_vals.append(float(out.info["fwd_loss"].item()))
            r_int_mean_vals.append(float(out.info["r_int_mean"].item()))

        self.icm.eval()

        if icm_loss_vals:
            logs["icm_loss"] = float(np.mean(icm_loss_vals))
            logs["inv_loss"] = float(np.mean(inverse_loss_vals))
            logs["fwd_loss"] = float(np.mean(forward_loss_vals))
            logs["r_int_mean"] = float(np.mean(r_int_mean_vals))
            logs["icm_train_skipped"] = 0.0
        else:
            logs["icm_train_skipped"] = 1.0
        

        return logs