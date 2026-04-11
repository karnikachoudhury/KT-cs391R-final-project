import os
import numpy as np
import torch
import random

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize, sync_envs_normalization
from stable_baselines3.common.callbacks import BaseCallback

from make_env import make_env
from icm import ICM
from icm_integration import ICMIntegration


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def make_icm_env(
    horizon: int,
    dense_reward: bool,
    lam: float,
    use_icm: bool,
    device: str,
    shared_icm: ICM,
    shared_icm_optimizer: torch.optim.Optimizer,
    icm_train_every: int = 50,
    icm_steps_per_update: int = 10,
    icm_batch_size: int = 256,
    train_icm: bool = True,
    r_int_clip: float = 0.5,
):
    def _make():
        base_env = make_env(horizon=horizon, dense_reward=dense_reward)

        return ICMIntegration(
            env=base_env,
            icm=shared_icm,
            icm_optimizer=shared_icm_optimizer,
            lam=lam,
            use_intrinsic_reward=use_icm,
            device=device,
            icm_batch_size=icm_batch_size,
            icm_train_every=icm_train_every,
            icm_steps_per_update=icm_steps_per_update,
            train_icm=train_icm,
            r_int_clip=r_int_clip,
        )

    return _make


# ---------------------------------------------------------------------------
# Callback
# ---------------------------------------------------------------------------

class EvalCallback(BaseCallback):
    def __init__(
        self,
        eval_env,
        eval_freq: int = 20_000,
        n_eval_episodes: int = 10,
        save_path: str = "./logs/best_model",
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = int(eval_freq)
        self.n_eval_episodes = int(n_eval_episodes)
        self.save_path = save_path
        self.best_success = -1.0
        self.last_eval_timestep = 0

    @staticmethod
    def _extract_success_from_info(info) -> float:
        if info is None:
            return 0.0
        if "episode_success" in info:
            return float(info["episode_success"])
        if "episode_success_so_far" in info:
            return float(info["episode_success_so_far"])
        if "success" in info:
            return float(info["success"])
        if "is_success" in info:
            return float(info["is_success"])
        return 0.0

    def _on_step(self) -> bool:
        if self.num_timesteps - self.last_eval_timestep < self.eval_freq:
            return True
        self.last_eval_timestep = self.num_timesteps

        sync_envs_normalization(self.training_env, self.eval_env)

        successes = 0
        episode_rewards_total = []
        episode_rewards_ext = []
        episode_rewards_int = []

        for _ in range(self.n_eval_episodes):
            obs = self.eval_env.reset()
            done = False

            ep_reward_total = 0.0
            ep_reward_ext = 0.0
            ep_reward_int = 0.0
            ep_success = 0.0

            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, rewards, dones, infos = self.eval_env.step(action)

                reward = float(rewards[0])
                done = bool(dones[0])
                info = infos[0] if isinstance(infos, (list, tuple)) else infos

                ep_reward_total += reward
                ep_reward_ext += float(info.get("reward_extrinsic", reward))
                ep_reward_int += float(info.get("reward_intrinsic", 0.0))

                step_success = self._extract_success_from_info(info)
                ep_success = max(ep_success, step_success)

            successes += int(ep_success > 0.5)
            episode_rewards_total.append(ep_reward_total)
            episode_rewards_ext.append(ep_reward_ext)
            episode_rewards_int.append(ep_reward_int)

        success_rate = successes / float(self.n_eval_episodes)
        mean_reward_total = float(np.mean(episode_rewards_total)) if episode_rewards_total else 0.0
        mean_reward_ext = float(np.mean(episode_rewards_ext)) if episode_rewards_ext else 0.0
        mean_reward_int = float(np.mean(episode_rewards_int)) if episode_rewards_int else 0.0

        if self.verbose:
            print(
                f"[Eval @ {self.num_timesteps}] "
                f"success_rate={success_rate:.3f}, "
                f"mean_total={mean_reward_total:.3f}, "
                f"mean_ext={mean_reward_ext:.3f}, "
                f"mean_int={mean_reward_int:.3f}, "
                f"best_success={self.best_success:.3f}"
            )

        self.logger.record("eval/success_rate", success_rate)
        self.logger.record("eval/mean_reward_total", mean_reward_total)
        self.logger.record("eval/mean_reward_extrinsic", mean_reward_ext)
        self.logger.record("eval/mean_reward_intrinsic", mean_reward_int)
        self.logger.dump(self.num_timesteps)

        if success_rate > self.best_success:
            self.best_success = success_rate
            os.makedirs(self.save_path, exist_ok=True)
            self.model.save(os.path.join(self.save_path, "best_model"))
            self.training_env.save(os.path.join(self.save_path, "vecnormalize.pkl"))
            if self.verbose:
                print(f"[Eval] Saved new best model (success={success_rate:.3f})")

        return True


# ---------------------------------------------------------------------------
# Training function
# ---------------------------------------------------------------------------

def train_icm(
    env_horizon: int = 1000,
    env_dense_reward: bool = True,
    icm_beta: float = 0.2,
    icm_lr: float = 3e-4,
    env_lambda: float = 1e-4,
    use_icm: bool = True,
    total_timesteps: int = 1_000_000,
    n_envs: int = 2,
    feature_dim: int = 64,
    device: str = "cpu",
    seed: int = 0,
    output_dir: str = None,
    eval_freq: int = 20_000,
    n_eval_episodes: int = 10,
    # ICM update frequency: 50 steps between updates, 10 gradient steps each.
    # With n_envs=2 and n_steps=2048, one PPO rollout = ~4096 env steps, so the
    # ICM gets ~800 gradient steps per rollout — enough to stay in sync with policy.
    icm_train_every: int = 50,
    icm_steps_per_update: int = 10,
    icm_batch_size: int = 256,
    r_int_clip: float = 0.5,
):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    run_name = "icm" if use_icm and env_lambda > 0.0 else "baseline"

    if output_dir is None:
        output_dir = os.path.join(
            "outputs_kc_icm_pickplace",
            f"{run_name}_dense_{env_dense_reward}_lam_{env_lambda}_horizon_{env_horizon}",
        )
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output dir: {output_dir}")

    probe_env = make_env(horizon=env_horizon, dense_reward=env_dense_reward)
    obs_dim = probe_env.observation_space.shape[0]
    action_dim = probe_env.action_space.shape[0]
    probe_env.close()
    print(f"obs_dim={obs_dim}, action_dim={action_dim}")

    # -----------------------------------------------------------------------
    # Shared ICM + shared optimizer across all envs
    # -----------------------------------------------------------------------
    shared_icm = ICM(
        obs_dim=obs_dim,
        action_dim=action_dim,
        feature_dim=feature_dim,
        beta=icm_beta,
    ).to(device)

    shared_icm_optimizer = torch.optim.Adam(shared_icm.parameters(), lr=icm_lr)

    # Training envs: train_icm=True (default) so ICM weights are updated here.
    env_factory = make_icm_env(
        horizon=env_horizon,
        dense_reward=env_dense_reward,
        lam=env_lambda,
        use_icm=use_icm,
        device=device,
        shared_icm=shared_icm,
        shared_icm_optimizer=shared_icm_optimizer,
        icm_train_every=icm_train_every,
        icm_steps_per_update=icm_steps_per_update,
        icm_batch_size=icm_batch_size,
        train_icm=True,
        r_int_clip=r_int_clip,
    )

    vec_env = make_vec_env(env_factory, n_envs=n_envs, seed=seed)
    # norm_reward=True: VecNormalize normalizes the PPO reward signal, which matters
    # because the total reward (r_ext + r_int) has a shifting scale during training.
    # Previously this was False, leaving the policy to deal with raw unbounded rewards.
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    # Eval env: train_icm=False so evaluation runs never mutate the shared ICM weights
    # or optimizer state. Without this, eval episodes would silently continue training
    # the ICM, contaminating both the metrics and the learned reward signal.
    eval_env_factory = make_icm_env(
        horizon=env_horizon,
        dense_reward=env_dense_reward,
        lam=env_lambda,
        use_icm=use_icm,
        device=device,
        shared_icm=shared_icm,
        shared_icm_optimizer=shared_icm_optimizer,
        icm_train_every=icm_train_every,
        icm_steps_per_update=icm_steps_per_update,
        icm_batch_size=icm_batch_size,
        train_icm=False,   # <-- eval env never trains the ICM
        r_int_clip=r_int_clip,
    )

    eval_vec_env = make_vec_env(eval_env_factory, n_envs=1, seed=seed + 100)
    eval_vec_env = VecNormalize(
        eval_vec_env, norm_obs=True, norm_reward=False, clip_obs=10.0
    )
    eval_vec_env.training = False
    eval_vec_env.norm_reward = False

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
        ent_coef=0.0,
        tensorboard_log=output_dir,
        policy_kwargs=policy_kwargs,
        seed=seed,
    )

    callback = EvalCallback(
        eval_env=eval_vec_env,
        eval_freq=eval_freq,
        n_eval_episodes=n_eval_episodes,
        save_path=os.path.join(output_dir, "best_model"),
        verbose=1,
    )

    model.learn(
        total_timesteps=total_timesteps,
        callback=callback,
        tb_log_name=run_name,
    )

    model.save(os.path.join(output_dir, "final_model"))
    vec_env.save(os.path.join(output_dir, "vecnormalize_final.pkl"))

    vec_env.close()
    eval_vec_env.close()

    return model


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    train_icm(
        env_horizon=500,
        env_dense_reward=True,
        icm_beta=0.2,
        icm_lr=3e-4,
        env_lambda=0.005,
        use_icm=True,
        total_timesteps=1_000_000,
        n_envs=4,
        eval_freq=20_000,
        n_eval_episodes=10,
        icm_train_every=50,
        icm_steps_per_update=10,
        icm_batch_size=256,
        r_int_clip=0.5,
    )