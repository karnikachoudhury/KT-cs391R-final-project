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
from icm_integration import ICMIntegration, unwrap_fully


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

class ICMEvalCallback(BaseCallback):
    def __init__(
        self,
        eval_env,
        master_icm: ICM,
        train_vec_env,
        eval_freq: int = 50_000,
        n_eval_episodes: int = 10,
        save_path: str = "./logs/best_model_icm",
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.master_icm = master_icm
        self.train_vec_env = train_vec_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
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
        train_wrappers = self._get_icm_wrappers(self.train_vec_env)
        if not train_wrappers:
            return
        avg_state = {}
        for key in train_wrappers[0].icm.state_dict():
            avg_state[key] = torch.stack(
                [w.icm.state_dict()[key].float() for w in train_wrappers]
            ).mean(dim=0)
        self.master_icm.load_state_dict(avg_state)
        for wrapper in train_wrappers:
            wrapper.icm.load_state_dict(avg_state)
        for wrapper in self._get_icm_wrappers(self.eval_env):
            wrapper.icm.load_state_dict(avg_state)

    def _get_success_from_eval_env(self):
        # FIX: unwrap_fully goes through GymWrapper all the way to the raw
        # robosuite env, so _check_success() reflects actual task state.
        try:
            env = self.eval_env.venv.envs[0]
            base = unwrap_fully(env)
            return float(base._check_success())
        except Exception:
            return 0.0

    def _on_step(self) -> bool:
        if self.num_timesteps - self.last_eval_timestep < self.eval_freq:
            return True
        self.last_eval_timestep = self.num_timesteps

        sync_envs_normalization(self.training_env, self.eval_env)
        self._sync_icm_weights()

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
                ep_success = max(ep_success, self._get_success_from_eval_env())

            successes += int(ep_success > 0.5)
            episode_rewards.append(ep_reward)

        success_rate = successes / float(self.n_eval_episodes)
        mean_reward = float(np.mean(episode_rewards))

        if self.verbose:
            print(
                f"[Eval @ {self.num_timesteps}] "
                f"success_rate={success_rate:.3f}, "
                f"mean_reward={mean_reward:.3f}, "
                f"best_success={self.best_success:.3f}"
            )

        self.logger.record("eval/success_rate", success_rate)
        self.logger.record("eval/mean_reward", mean_reward)
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
    env_horizon: int = 500,
    env_dense_reward: bool = True,
    icm_beta: float = 0.2,
    icm_lr: float = 3e-4,
    env_lambda: float = 0.05,
    use_icm: bool = True,
    total_timesteps: int = 1_000_000,
    n_envs: int = 8,
    feature_dim: int = 64,
    device: str = "cpu",
    seed: int = 0,
    output_dir: str = None,
    eval_freq: int = 50_000,
    n_eval_episodes: int = 10,
):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    if output_dir is None:
        output_dir = os.path.join(
            "outputs_kc_icm",
            f"icm_{use_icm}_dense_{env_dense_reward}_lam_{env_lambda}_horizon_{env_horizon}",
        )
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output dir: {output_dir}")

    _probe = make_env(horizon=env_horizon, dense_reward=env_dense_reward)
    obs_dim = _probe.observation_space.shape[0]
    action_dim = _probe.action_space.shape[0]
    _probe.close()
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
        ent_coef=0.01,
        tensorboard_log=output_dir,
        policy_kwargs=policy_kwargs,
        seed=seed,
    )

    callback = ICMEvalCallback(
        eval_env=eval_vec_env,
        master_icm=master_icm,
        train_vec_env=vec_env,
        eval_freq=eval_freq,
        n_eval_episodes=n_eval_episodes,
        save_path=os.path.join(output_dir, "best_model"),
    )

    model.learn(
        total_timesteps=total_timesteps,
        callback=callback,
        tb_log_name="icm" if use_icm else "baseline",
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
        env_lambda=0.05,
        use_icm=True,
        total_timesteps=1_000_000,
        n_envs=8,
    )