import os
import numpy as np
import torch
import random

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize, sync_envs_normalization
from stable_baselines3.common.callbacks import BaseCallback
import gc
from make_env import make_env
from icm import ICM
from icm_integration import ICMIntegration
import faulthandler

# create ICM environment and return instance of integration wrapper
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
    delay: int = 0,
    decay: str = None,
    n_envs: int = 4,
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
            delay=delay,
            decay_type=decay,
            n_envs=n_envs,
        )

    return _make



# evaluation subclass that evaluates the current policy and saves best model
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

    # get success from info dict to be logged
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

    # what information we need on each step to decide when to evaluate and to compute metrics
    def _on_step(self) -> bool:
        if self.num_timesteps - self.last_eval_timestep < self.eval_freq:
            return True
        self.last_eval_timestep = self.num_timesteps
        print(f"\neval: \n\tnum_timesteps = {self.num_timesteps}")
        lambda_icm = -1.0
        sync_envs_normalization(self.training_env, self.eval_env)

        successes = 0
        episode_rewards_total = []
        episode_rewards_ext = []
        episode_rewards_int = []
        episode_rewards_int_norm = []

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
            print(f"eval: \n\tlast lambda = {lambda_icm:.4f}")
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
        if info.get("lambda") is not None:
            self.logger.record("eval/lambda", info.get("lambda", 0.0))
        self.logger.dump(self.num_timesteps)

        if success_rate > self.best_success:
            self.best_success = success_rate
            os.makedirs(self.save_path, exist_ok=True)
            self.model.save(os.path.join(self.save_path, "best_model"))
            self.training_env.save(os.path.join(self.save_path, "vecnormalize.pkl"))
            if self.verbose:
                print(f"[Eval] Saved new best model (success={success_rate:.3f})")

        return True

# function to train ICM
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
    icm_train_every: int = 50,
    icm_steps_per_update: int = 10,
    icm_batch_size: int = 256,
    r_int_clip: float = 0.5,
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
            "outputs_kc_icm_beat_entropy_try",
            f"{run_name}_dense_{env_dense_reward}_lam_{env_lambda}_horizon_{env_horizon}{"_delay_" + str(delay) if delay else ""}{"_entropy_" + str(entropy) if entropy > 0.0 else ""}{("_decay_" + decay if decay else "")}",
        )
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output dir: {output_dir}")

    probe_env = make_env(horizon=env_horizon, dense_reward=env_dense_reward)
    obs_dim = probe_env.observation_space.shape[0]
    action_dim = probe_env.action_space.shape[0]
    probe_env.close()
    print(f"obs_dim={obs_dim}, action_dim={action_dim}")

    # we share ICM and optimizers across the multiple environments
    shared_icm = ICM(
        obs_dim=obs_dim,
        action_dim=action_dim,
        feature_dim=feature_dim,
        beta=icm_beta,
    ).to(device)

    shared_icm_optimizer = torch.optim.Adam(shared_icm.parameters(), lr=icm_lr)

    # training env creation function, called when we need to train the multiple envs
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
        delay = delay // n_envs,
        decay = decay,
        n_envs = n_envs,
    )

    # wrap in VecEnv and normalize
    vec_env = make_vec_env(env_factory, n_envs=n_envs, seed=seed)


    # evaluation env creation function
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
        delay = delay // n_envs,
        decay = decay,
        n_envs = n_envs
    )
    # running as often so may be ending decay differently
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


if __name__ == "__main__":
    faulthandler.enable()
    print("starting")
    # super large lambda, 1000 delay
    train_icm(
        env_horizon=500,
        env_dense_reward=False,
        icm_beta=0.2,
        icm_lr=3e-4,
        env_lambda=0.005,
        use_icm=True,
        total_timesteps=1_000_000,
        n_envs=4,
        eval_freq=20_000,
        n_eval_episodes=20,
        icm_train_every=100,
        icm_steps_per_update=10,
        icm_batch_size=256,
        r_int_clip=0.1,
        delay=0,
        entropy=0.0,
        #decay = 'exp'
    )
