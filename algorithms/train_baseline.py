import os
import numpy as np
import torch
import random

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize, sync_envs_normalization
from stable_baselines3.common.callbacks import BaseCallback
from make_env import make_env

LOG_DIR = "./ppo_lift_tensorboard"
os.makedirs(LOG_DIR, exist_ok=True)

# create vectorized training environments
n_envs = 8
vec_env = make_vec_env(
    make_env,
    n_envs=n_envs,
    env_kwargs={"horizon": 200, "dense_reward": True},
    seed=0,
)
vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False, clip_obs=10.0)

# separate evaluation environment
eval_env = make_vec_env(
    make_env,
    n_envs=1,
    env_kwargs={"horizon": 200, "dense_reward": True},
    seed=42,
)
eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0)

# important: do not update normalization stats during evaluation
eval_env.training = False
eval_env.norm_reward = False

# larger policy network to give PPO more capacity
policy_kwargs = dict(net_arch=dict(pi=[256, 256], vf=[256, 256]))

model = PPO(
    "MlpPolicy",
    vec_env,
    verbose=1,
    n_steps=2048,
    batch_size=64,
    gamma=0.99,
    gae_lambda=0.95,
    n_epochs=10,
    learning_rate=3e-4,
    clip_range=0.2,
    tensorboard_log=LOG_DIR,
    policy_kwargs=policy_kwargs,
    seed=0,
)


class SuccessEvalCallback(BaseCallback):
    """Evaluate the agent and save best model by robosuite success rate."""

    def __init__(
        self,
        eval_env,
        eval_freq: int = 50_000,
        n_eval_episodes: int = 10,
        save_path: str = "./logs/best_model",
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.best_success = -1.0
        self.save_path = save_path
        self.last_eval_timestep = 0

    def _get_base_env(self, env):
        """
        Unwrap Monitor / Gym wrappers until we reach the underlying robosuite env.
        """
        while hasattr(env, "env"):
            env = env.env
        return env

    def _get_success_from_base_env(self):
        """
        Safely query robosuite task success from the underlying env.
        """
        try:
            # eval_env is VecNormalize -> DummyVecEnv -> wrapped env
            base_env = self._get_base_env(self.eval_env.venv.envs[0])
            return float(base_env._check_success())
        except Exception:
            return 0.0

    def _on_step(self) -> bool:
        # use actual timesteps, not callback calls
        if self.num_timesteps - self.last_eval_timestep < self.eval_freq:
            return True

        self.last_eval_timestep = self.num_timesteps

        # copy normalization stats from training env to eval env
        sync_envs_normalization(self.training_env, self.eval_env)

        successes = 0
        episode_rewards = []

        for _ in range(self.n_eval_episodes):
            obs = self.eval_env.reset()
            done = False
            ep_reward = 0.0
            ep_success = 0.0

            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, rewards, dones, infos = self.eval_env.step(action)

                ep_reward += float(rewards[0])
                done = bool(dones[0])

                # safer version:
                # keep checking robosuite success during the episode
                ep_success = max(ep_success, self._get_success_from_base_env())

            successes += int(ep_success > 0.5)
            episode_rewards.append(ep_reward)
            print("EPISODE SUCCESS FROM ENV:", ep_success)

        success_rate = successes / float(self.n_eval_episodes)
        mean_reward = float(np.mean(episode_rewards))

        print(
            f"[Eval @ {self.num_timesteps}] "
            f"success_rate={success_rate:.3f}, "
            f"mean_reward={mean_reward:.3f}, "
            f"best_success={self.best_success:.3f}"
        )

        # log to TensorBoard
        self.logger.record("eval/success_rate", success_rate)
        self.logger.record("eval/mean_reward", mean_reward)
        self.logger.dump(self.num_timesteps)

        if success_rate > self.best_success:
            self.best_success = success_rate
            os.makedirs(self.save_path, exist_ok=True)
            model_path = os.path.join(self.save_path, "best_model")
            self.model.save(model_path)
            # save normalization stats too
            self.training_env.save(os.path.join(self.save_path, "vecnormalize.pkl"))
            print(f"[Eval] Saved new best model to {model_path}")

        return True


eval_callback = SuccessEvalCallback(
    eval_env,
    eval_freq=50_000,
    n_eval_episodes=10,
    save_path="./logs/best_model",
)

# Train longer to allow reaching success more often
seed = 0
np.random.seed(seed)
torch.manual_seed(seed)
random.seed(seed)

model.learn(
    total_timesteps=1_000_000,
    callback=eval_callback,
    tb_log_name="baseline_vec",
)
model.save("ppo_lift_baseline")

vec_env.close()
eval_env.close()