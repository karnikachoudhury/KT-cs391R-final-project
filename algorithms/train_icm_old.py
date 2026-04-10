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
    obs_dim: int,
    action_dim: int,
    feature_dim: int,
    icm_beta: float,
    icm_lr: float,
    lam: float,
    use_icm: bool,
    device: str,
):
    def _make():
        base_env = make_env(horizon=horizon, dense_reward=dense_reward)

        icm_module = ICM(
            obs_dim=obs_dim,
            action_dim=action_dim,
            feature_dim=feature_dim,
            beta=icm_beta,
        )
        icm_optimizer = torch.optim.Adam(icm_module.parameters(), lr=icm_lr)

        return ICMIntegration(
            env=base_env,
            icm=icm_module,
            icm_optimizer=icm_optimizer,
            lam=lam,
            use_intrinsic_reward=use_icm,
            device=device,
            icm_batch_size=256,
            icm_train_every=200,
            icm_steps_per_update=10,
        )

    return _make


# ---------------------------------------------------------------------------
# Callback
# ---------------------------------------------------------------------------

class EvalCallback(BaseCallback):
    def __init__(
        self,
        eval_env,
        master_icm: ICM,
        train_vec_env,
        use_icm: bool,
        eval_freq: int = 20_000,
        n_eval_episodes: int = 10,
        save_path: str = "./logs/best_model",
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.master_icm = master_icm
        self.train_vec_env = train_vec_env
        self.use_icm = bool(use_icm)
        self.eval_freq = int(eval_freq)
        self.n_eval_episodes = int(n_eval_episodes)
        self.save_path = save_path
        self.best_success = -1.0
        self.last_eval_timestep = 0

    def _get_icm_wrappers(self, vec_env):
        wrappers = []
        envs = vec_env.venv.envs if hasattr(vec_env, "venv") else vec_env.envs

        for env in envs:
            cur = env
            while cur is not None:
                if isinstance(cur, ICMIntegration):
                    wrappers.append(cur)
                    break
                cur = getattr(cur, "env", None)

        return wrappers

    def _sync_icm_weights(self):
        # Keep this only for actual ICM runs
        if not self.use_icm or self.master_icm is None:
            return

        train_wrappers = self._get_icm_wrappers(self.train_vec_env)
        if not train_wrappers:
            return

        state_dicts = [w.icm.state_dict() for w in train_wrappers]
        avg_state = {}

        for key in state_dicts[0]:
            avg_state[key] = torch.stack(
                [sd[key].float() for sd in state_dicts], dim=0
            ).mean(dim=0)

        self.master_icm.load_state_dict(avg_state)

        for wrapper in train_wrappers:
            wrapper.icm.load_state_dict(avg_state)

        for wrapper in self._get_icm_wrappers(self.eval_env):
            wrapper.icm.load_state_dict(avg_state)

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
        self._sync_icm_weights()

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
    env_lambda: float = 0.001,
    use_icm: bool = True,
    total_timesteps: int = 1_000_000,
    n_envs: int = 2,
    feature_dim: int = 64,
    device: str = "cpu",
    seed: int = 0,
    output_dir: str = None,
    eval_freq: int = 20_000,
    n_eval_episodes: int = 10,
):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    run_name = "icm" if use_icm and env_lambda > 0.0 else "baseline"

    if output_dir is None:
        output_dir = os.path.join(
            "outputs_kc_icm",
            f"{run_name}_dense_{env_dense_reward}_lam_{env_lambda}_horizon_{env_horizon}",
        )
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output dir: {output_dir}")

    probe_env = make_env(horizon=env_horizon, dense_reward=env_dense_reward)
    obs_dim = probe_env.observation_space.shape[0]
    action_dim = probe_env.action_space.shape[0]
    probe_env.close()
    print(f"obs_dim={obs_dim}, action_dim={action_dim}")

    env_factory = make_icm_env(
        horizon=env_horizon,
        dense_reward=env_dense_reward,
        obs_dim=obs_dim,
        action_dim=action_dim,
        feature_dim=feature_dim,
        icm_beta=icm_beta,
        icm_lr=icm_lr,
        lam=env_lambda,
        use_icm=use_icm,
        device=device,
    )

    vec_env = make_vec_env(env_factory, n_envs=n_envs, seed=seed)
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False, clip_obs=10.0)

    eval_vec_env = make_vec_env(env_factory, n_envs=1, seed=seed + 100)
    eval_vec_env = VecNormalize(
        eval_vec_env, norm_obs=True, norm_reward=False, clip_obs=10.0
    )
    eval_vec_env.training = False
    eval_vec_env.norm_reward = False

    master_icm = None
    if use_icm and env_lambda > 0.0:
        master_icm = ICM(
            obs_dim=obs_dim,
            action_dim=action_dim,
            feature_dim=feature_dim,
            beta=icm_beta,
        ).to(device)

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
        master_icm=master_icm,
        train_vec_env=vec_env,
        use_icm=(use_icm and env_lambda > 0.0),
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
    # Baseline run:
    train_icm(
        env_horizon=1000,
        env_dense_reward=True,
        icm_beta=0.2,
        icm_lr=3e-4,
        env_lambda=0.0,
        use_icm=False,
        total_timesteps=1_000_000,
        n_envs=2,
        eval_freq=20_000,
        n_eval_episodes=10,
    )

    # For an ICM run, use:
    # train_icm(
    #     env_horizon=1000,
    #     env_dense_reward=True,
    #     icm_beta=0.2,
    #     icm_lr=3e-4,
    #     env_lambda=0.001,
    #     use_icm=True,
    #     total_timesteps=1_000_000,
    #     n_envs=2,
    #     eval_freq=20_000,
    #     n_eval_episodes=10,
    # )