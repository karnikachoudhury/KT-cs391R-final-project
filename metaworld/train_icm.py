import os
import random
import numpy as np
import torch
import faulthandler

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize, sync_envs_normalization
from stable_baselines3.common.callbacks import BaseCallback

from make_env_meta import make_env
from icm import ICM
from icm_integration_meta import ICMIntegration


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
    env_name: str = "drawer-open-v3",
    seed: int = 0,
    icm_train_every: int = 100,
    icm_steps_per_update: int = 5,
    icm_batch_size: int = 256,
    train_icm: bool = True,
    r_int_clip: float = 0.05,
    delay: int = 0,
    decay: str = None,
    n_envs: int = 4,
):
    def _make():
        base_env = make_env(
            horizon=horizon,
            seed=seed,
            env_name=env_name,
        )

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
            delay=delay,
            decay_type=decay,
            n_envs=n_envs,
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
        n_eval_episodes: int = 20,
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
        print(f"\nEval @ timesteps = {self.num_timesteps}")

        sync_envs_normalization(self.training_env, self.eval_env)

        successes = 0
        episode_rewards_total = []
        episode_rewards_ext = []
        episode_rewards_int = []
        episode_rewards_int_norm = []
        lambda_icm = -1.0

        for _ in range(self.n_eval_episodes):
            obs = self.eval_env.reset()
            done = False

            ep_reward_total = 0.0
            ep_reward_ext = 0.0
            ep_reward_int = 0.0
            ep_reward_int_norm = 0.0
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
                ep_reward_int_norm += float(info.get("reward_intrinsic_normalized", 0.0))
                lambda_icm = float(info.get("lambda", -1.0))
                step_success = self._extract_success_from_info(info)
                ep_success = max(ep_success, step_success)

            successes += int(ep_success > 0.5)
            episode_rewards_total.append(ep_reward_total)
            episode_rewards_ext.append(ep_reward_ext)
            episode_rewards_int.append(ep_reward_int)
            episode_rewards_int_norm.append(ep_reward_int_norm)

        success_rate = successes / float(self.n_eval_episodes)
        mean_reward_total = float(np.mean(episode_rewards_total)) if episode_rewards_total else 0.0
        mean_reward_ext = float(np.mean(episode_rewards_ext)) if episode_rewards_ext else 0.0
        mean_reward_int = float(np.mean(episode_rewards_int)) if episode_rewards_int else 0.0
        mean_reward_int_norm = float(np.mean(episode_rewards_int_norm)) if episode_rewards_int_norm else 0.0

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
        self.logger.record("eval/mean_reward_intrinsic_normalized", mean_reward_int_norm)
        if lambda_icm >= 0.0:
            self.logger.record("eval/lambda", lambda_icm)
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
    env_name: str = "drawer-open-v3",
    env_horizon: int = 150,
    env_dense_reward: bool = True,
    icm_beta: float = 0.2,
    icm_lr: float = 3e-4,
    env_lambda: float = 1e-3,
    use_icm: bool = True,
    total_timesteps: int = 1_000_000,
    n_envs: int = 8,
    feature_dim: int = 64,
    device: str = "cpu",
    seed: int = 0,
    output_dir: str = None,
    eval_freq: int = 20_000,
    n_eval_episodes: int = 20,
    icm_train_every: int = 100,
    icm_steps_per_update: int = 5,
    icm_batch_size: int = 256,
    r_int_clip: float = 0.05,
    delay: int = 0,
    entropy: float = 0.0,
    decay: str = None,
):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    run_name = "icm" if use_icm and env_lambda > 0.0 else "baseline"

    if output_dir is None:
        output_dir = os.path.join(
            "outputs_metaworld_icm_vs_entropy",
            (
                f"{env_name}_{run_name}"
                f"_dense_{env_dense_reward}"
                f"_lam_{env_lambda}"
                f"_horizon_{env_horizon}"
                f"{'_delay_' + str(delay) if delay else ''}"
                f"{'_entropy_' + str(entropy) if entropy > 0.0 else ''}"
                f"{'_decay_' + decay if decay else ''}"
            ),
        )
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output dir: {output_dir}")

    probe_env = make_env(
        horizon=env_horizon,
        seed=seed,
        env_name=env_name,
    )
    obs_dim = probe_env.observation_space.shape[0]
    action_dim = probe_env.action_space.shape[0]
    probe_env.close()
    print(f"obs_dim={obs_dim}, action_dim={action_dim}")

    shared_icm = ICM(
        obs_dim=obs_dim,
        action_dim=action_dim,
        feature_dim=feature_dim,
        beta=icm_beta,
    ).to(device)

    shared_icm_optimizer = torch.optim.Adam(shared_icm.parameters(), lr=icm_lr)

    env_factory = make_icm_env(
        horizon=env_horizon,
        dense_reward=env_dense_reward,
        lam=env_lambda,
        use_icm=use_icm,
        device=device,
        shared_icm=shared_icm,
        shared_icm_optimizer=shared_icm_optimizer,
        env_name=env_name,
        seed=seed,
        icm_train_every=icm_train_every,
        icm_steps_per_update=icm_steps_per_update,
        icm_batch_size=icm_batch_size,
        train_icm=True,
        r_int_clip=r_int_clip,
        delay=delay // max(n_envs, 1),
        decay=decay,
        n_envs=n_envs,
    )

    vec_env = make_vec_env(env_factory, n_envs=n_envs, seed=seed)
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    eval_env_factory = make_icm_env(
        horizon=env_horizon,
        dense_reward=env_dense_reward,
        lam=env_lambda,
        use_icm=use_icm,
        device=device,
        shared_icm=shared_icm,
        shared_icm_optimizer=shared_icm_optimizer,
        env_name=env_name,
        seed=seed + 1000,
        icm_train_every=icm_train_every,
        icm_steps_per_update=icm_steps_per_update,
        icm_batch_size=icm_batch_size,
        train_icm=False,
        r_int_clip=r_int_clip,
        delay=delay // max(n_envs, 1),
        decay=decay,
        n_envs=n_envs,
    )

    eval_vec_env = make_vec_env(eval_env_factory, n_envs=1, seed=seed + 1000)
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
        n_steps=512,
        batch_size=256,
        gamma=0.99,
        gae_lambda=0.95,
        n_epochs=10,
        learning_rate=3e-4,
        clip_range=0.2,
        ent_coef=entropy,
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
    faulthandler.enable()
    print("starting")

    # PPO + ICM run
    train_icm(
        env_name="drawer-open-v3",
        env_horizon=150,
        env_dense_reward=False,
        icm_beta=0.2,
        icm_lr=3e-4,
        env_lambda=0.05,
        use_icm=True,
        total_timesteps=1_000_000,
        n_envs=8,
        eval_freq=20_000,
        n_eval_episodes=20,
        icm_train_every=100,
        icm_steps_per_update=5,
        icm_batch_size=256,
        r_int_clip=0.05,
        delay=0,
        entropy=0.0,
        #decay="exp",
    )

    # For PPO + entropy baseline, change the call above to:
    
    # train_icm(
    #     env_name="drawer-open-v3",
    #     env_horizon=150,
    #     env_dense_reward=False,
    #     env_lambda=0.0,
    #     use_icm=False,
    #     total_timesteps=1_000_000,
    #     n_envs=8,
    #     eval_freq=20_000,
    #     n_eval_episodes=20,
    #     entropy=0.001,
    # )