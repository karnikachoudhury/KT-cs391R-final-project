import os
import torch
from stable_baselines3 import PPO

from make_env import make_env
from icm import ICM
from icm_integration import ICMIntegration


def _unwrap_to_success_env(env):
    """
    Walk through wrapper layers until we find an env with _check_success().
    """
    cur = env
    visited = set()

    while cur is not None and id(cur) not in visited:
        visited.add(id(cur))

        if hasattr(cur, "_check_success") and callable(getattr(cur, "_check_success")):
            return cur

        cur = getattr(cur, "env", None)

    return None


def _query_env_success(env) -> float:
    """
    Query success directly from the underlying robosuite env.
    """
    base_env = _unwrap_to_success_env(env)
    if base_env is None:
        return 0.0

    try:
        return float(base_env._check_success())
    except Exception:
        return 0.0


def evaluate_success(model, eval_env, eval_episodes=10):
    """
    Deterministic evaluation using the environment's _check_success() directly.

    Returns:
        success_rate: success at terminal state
        had_success_rate: success at any point during episode
        mean_reward: average total episode reward
    """
    eval_env.icm.eval()

    end_successes = 0
    had_successes = 0
    episode_rewards = []

    with torch.no_grad():
        for _ in range(eval_episodes):
            obs, info = eval_env.reset()
            done = False
            ep_reward = 0.0
            ep_had_success = False

            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = eval_env.step(action)
                done = terminated or truncated
                ep_reward += float(reward)

                if _query_env_success(eval_env):
                    ep_had_success = True

            terminal_success = bool(_query_env_success(eval_env))

            if ep_had_success:
                had_successes += 1
            if terminal_success:
                end_successes += 1

            episode_rewards.append(ep_reward)

    success_rate = end_successes / eval_episodes
    had_success_rate = had_successes / eval_episodes
    mean_reward = float(sum(episode_rewards) / len(episode_rewards))

    return success_rate, had_success_rate, mean_reward


def test_model(model, eval_env, output_dir, run_name, eval_episodes=100):
    output_path = os.path.join(output_dir, f"test_{run_name}.txt")
    with open(output_path, "w") as output_file:
        output_file.write("\nTesting model\n")
        eval_env.icm.eval()

        with torch.no_grad():
            successes = 0
            had_success = 0

            for i in range(eval_episodes):
                obs, info = eval_env.reset()
                done = False
                steps = 0
                ep_had_success = False

                last_r_ext = None
                last_r_int = None
                last_r_total = None

                while not done:
                    action, _ = model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, info = eval_env.step(action)
                    done = terminated or truncated
                    steps += 1

                    if _query_env_success(eval_env):
                        ep_had_success = True

                    last_r_ext = info.get("reward_extrinsic")
                    last_r_int = info.get("reward_intrinsic")
                    last_r_total = info.get("reward_total")

                terminal_success = bool(_query_env_success(eval_env))

                if ep_had_success:
                    had_success += 1
                if terminal_success:
                    successes += 1

                output_file.write(
                    f"Episode {i}:\n"
                    f"  Steps: {steps}\n"
                    f"  Had Success: {ep_had_success}\n"
                    f"  End Success: {terminal_success}\n"
                    f"  R_e: {last_r_ext}\n"
                    f"  R_i: {last_r_int}\n"
                    f"  R_total: {last_r_total}\n"
                )

            success_rate = 100.0 * successes / eval_episodes
            had_success_rate = 100.0 * had_success / eval_episodes

            output_file.write(f"End success rate: {success_rate:.2f}%\n")
            output_file.write(f"Had success rate: {had_success_rate:.2f}%\n")

        return success_rate, had_success_rate


def train_icm(
    env_horizon=200,
    env_dense_reward=True,
    icm_beta=0.2,
    icm_lr=3e-4,           # FIX: reduced from 1e-3 to 3e-4 for more stable ICM learning
    env_lambda=0.01,       # FIX: reduced from 0.1 so intrinsic reward doesn't drown extrinsic
    use_icm=True,
    eps_per_update=10,
    updates_per_epoch=5,
    epochs=1,
    output_dir=None,
    icm_train_every=200,   # NEW: how often (steps) to update ICM inline
    icm_steps_per_update=10,  # NEW: gradient steps per inline ICM update
):
    if output_dir is None:
        output_dir = os.path.join(
            "outputs_kc_finalish",
            f"work_icm_{use_icm}_epochs_{epochs}_horizon_{env_horizon}_eps_{eps_per_update}_updates_{updates_per_epoch}_dense_{env_dense_reward}_beta_{icm_beta}_lr_{icm_lr}_lambda_{env_lambda}/",
        )
        os.makedirs(output_dir, exist_ok=True)
    else:
        output_dir = os.path.join("outputs_kc_finalish", output_dir)
        os.makedirs(output_dir, exist_ok=True)

    print(f"Output dir: {output_dir}")

    n_steps = eps_per_update * env_horizon
    chunk = n_steps * updates_per_epoch
    total_steps = chunk * epochs

    # Training env
    train_base_environment = make_env(env_horizon, env_dense_reward)
    train_obs_dim = train_base_environment.observation_space.shape[0]
    train_action_dim = train_base_environment.action_space.shape[0]

    print(f"Train obs_dim: {train_obs_dim}, action_dim: {train_action_dim}")

    train_icm_module = ICM(
        obs_dim=train_obs_dim,
        action_dim=train_action_dim,
        feature_dim=64,       # FIX: reduced from 128; less likely to overfit on small buffer
        beta=icm_beta,
        forward_scale=0.5,
    )
    train_icm_optimizer = torch.optim.Adam(train_icm_module.parameters(), lr=icm_lr)

    train_env = ICMIntegration(
        env=train_base_environment,
        icm=train_icm_module,
        icm_optimizer=train_icm_optimizer,
        lam=env_lambda,
        use_intrinsic_reward=use_icm,
        device="cpu",
        icm_batch_size=64,
        icm_batches_per_call=100,  # kept for compat, no longer used in main loop
        chunk_size=chunk,
        icm_train_every=icm_train_every,        # NEW
        icm_steps_per_update=icm_steps_per_update,  # NEW
    )

    # Separate eval env
    eval_base_environment = make_env(env_horizon, env_dense_reward)
    eval_obs_dim = eval_base_environment.observation_space.shape[0]
    eval_action_dim = eval_base_environment.action_space.shape[0]

    print(f"Eval obs_dim: {eval_obs_dim}, action_dim: {eval_action_dim}")

    eval_icm_module = ICM(
        obs_dim=eval_obs_dim,
        action_dim=eval_action_dim,
        feature_dim=64,       # FIX: must match train ICM for weight sync to work
        beta=icm_beta,
        forward_scale=0.5,
    )
    eval_icm_optimizer = torch.optim.Adam(eval_icm_module.parameters(), lr=icm_lr)

    eval_env = ICMIntegration(
        env=eval_base_environment,
        icm=eval_icm_module,
        icm_optimizer=eval_icm_optimizer,
        lam=env_lambda,
        use_intrinsic_reward=use_icm,
        device="cpu",
        icm_batch_size=64,
        icm_batches_per_call=100,
        chunk_size=chunk,
        icm_train_every=icm_train_every,
        icm_steps_per_update=icm_steps_per_update,
    )

    model = PPO(
        "MlpPolicy",
        train_env,
        verbose=1,
        n_steps=n_steps,
        batch_size=64,
        tensorboard_log=output_dir,
    )

    for i in range(total_steps // chunk):
        print(f"\nStep {i + 1} / {total_steps // chunk}\n")
        model = model.learn(
            total_timesteps=chunk,
            reset_num_timesteps=False,
            tb_log_name="icm",
        )

        # FIX: sync ICM weights from train env to eval env before evaluation.
        # Previously the eval ICM was a permanently random untrained network.
        eval_env.icm.load_state_dict(train_env.icm.state_dict())

        success_rate, had_success_rate, mean_reward = evaluate_success(
            model,
            eval_env,
            eval_episodes=10,
        )

        print(
            f"[Eval @ {model.num_timesteps}] "
            f"success_rate={success_rate:.3f}, "
            f"had_success_rate={had_success_rate:.3f}, "
            f"mean_reward={mean_reward:.3f}"
        )

        model.logger.record("eval/success_rate", success_rate)
        model.logger.record("eval/had_success_rate", had_success_rate)
        model.logger.record("eval/mean_reward", mean_reward)

        # FIX: train_env.train_icm() removed from here. ICM is now trained
        # inline inside step() via _train_icm_inline(), so it is always
        # current by the time PPO computes policy gradients.

        model.logger.dump(model.num_timesteps)

    test_model(model, eval_env, output_dir, "final")

    train_env.close()
    eval_env.close()


def copy_outputs(src_file):
    output_file = os.path.join("outputs", src_file)
    with open(output_file, "r") as f:
        cur_dir = None
        lines = []
        for line in f:
            if line.startswith("Output dir:"):
                if cur_dir is not None:
                    output = os.path.join(cur_dir, "output.txt")
                    with open(output, "w") as out_f:
                        out_f.writelines(lines)

                cur_dir = line.split("Output dir: ")[1].strip()
                lines = []
            else:
                lines.append(line)

        if cur_dir is not None:
            output = os.path.join(cur_dir, "output.txt")
            with open(output, "w") as out_f:
                out_f.writelines(lines)


if __name__ == "__main__":
    # Dense reward baseline (no ICM) — short episodes
    train_icm(
        env_horizon=400,
        env_dense_reward=True,
        icm_beta=0.2,
        icm_lr=3e-4,
        env_lambda=0.01,
        use_icm=False,
        eps_per_update=5,
        updates_per_epoch=5,
        epochs=100,
    )

    # Dense reward baseline (no ICM) — longer rollouts
    train_icm(
        env_horizon=400,
        env_dense_reward=True,
        icm_beta=0.2,
        icm_lr=3e-4,
        env_lambda=0.01,
        use_icm=False,
        eps_per_update=10,
        updates_per_epoch=1,
        epochs=250,
    )

    # Dense reward + ICM
    train_icm(
        env_horizon=400,
        env_dense_reward=True,
        icm_beta=0.2,
        icm_lr=3e-4,
        env_lambda=0.01,   # FIX: was 0.01 originally, kept — good value for dense
        use_icm=True,
        eps_per_update=10,
        updates_per_epoch=1,
        epochs=250,
    )

    # Sparse reward + ICM — hard task, needs longer horizon and more epochs
    train_icm(
        env_horizon=500,   # FIX: increased from 200, Door needs more steps
        env_dense_reward=False,
        icm_beta=0.2,
        icm_lr=3e-4,
        env_lambda=0.01,
        use_icm=True,
        eps_per_update=10,
        updates_per_epoch=1,
        epochs=500,
    )

    # Sparse reward baseline (no ICM)
    train_icm(
        env_horizon=500,   # FIX: increased from 200 to match ICM run above
        env_dense_reward=False,
        icm_beta=0.2,
        icm_lr=3e-4,
        env_lambda=0.01,
        use_icm=False,
        eps_per_update=10,
        updates_per_epoch=1,
        epochs=500,
    )

    copy_outputs("results.txt")