import torch

from stable_baselines3 import PPO

from make_env import make_env
from icm import ICM
from icm_integration import ICMIntegration


def _unwrap_to_success_env(env):
    """
    Walk through wrapper layers until we find an env with _check_success().
    Prints every layer so you can verify what it is actually hitting.
    """
    cur = env
    visited = set()
    depth = 0

    print("\n--- UNWRAP START ---")
    while cur is not None and id(cur) not in visited:
        visited.add(id(cur))
        print(f"[unwrap depth {depth}] type = {type(cur)}")

        if hasattr(cur, "_check_success") and callable(getattr(cur, "_check_success")):
            print(f"[unwrap] FOUND _check_success on: {type(cur)}")
            print("--- UNWRAP END ---\n")
            return cur

        cur = getattr(cur, "env", None)
        depth += 1

    print("[unwrap] NO ENV WITH _check_success FOUND")
    print("--- UNWRAP END ---\n")
    return None


def _query_env_success(env) -> float:
    """
    Query success directly from the underlying environment and print the result.
    """
    base_env = _unwrap_to_success_env(env)
    if base_env is None:
        print("[success query] returning 0.0 because no success env was found")
        return 0.0

    try:
        val = float(base_env._check_success())
        print(f"[success query] _check_success() returned: {val}")
        return val
    except Exception as e:
        print(f"[success query] ERROR calling _check_success(): {repr(e)}")
        return 0.0


def run_debug_episode(model, env, episode_idx=0, max_steps=None):
    """
    Run one deterministic episode and print:
    - wrapper chain
    - success value before stepping
    - success value during the episode
    - whether success was ever seen
    - terminal success
    """
    print(f"\n================ DEBUG EPISODE {episode_idx} ================\n")

    obs, info = env.reset()
    print("[reset] got initial observation")
    print("[reset] querying success right after reset...")
    initial_success = _query_env_success(env)
    print(f"[reset] initial success = {initial_success}\n")

    done = False
    step_count = 0
    had_success = False

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        step_count += 1

        current_success = _query_env_success(env)
        if current_success > 0.5:
            had_success = True
            print(f"[step {step_count}] SUCCESS HIT")

        print(
            f"[step {step_count}] "
            f"reward={float(reward):.6f}, "
            f"terminated={terminated}, truncated={truncated}, "
            f"info_success={info.get('success', 'missing')}, "
            f"episode_success={info.get('episode_success', 'missing')}, "
            f"episode_success_so_far={info.get('episode_success_so_far', 'missing')}, "
            f"queried_success={current_success}"
        )

        if max_steps is not None and step_count >= max_steps:
            print(f"[debug] stopping early at max_steps={max_steps}")
            break

    terminal_success = _query_env_success(env)

    print("\n---------------- EPISODE SUMMARY ----------------")
    print(f"steps taken          : {step_count}")
    print(f"had success anytime  : {had_success}")
    print(f"terminal success     : {terminal_success}")
    print(f"final info dict keys : {list(info.keys())}")
    print("-------------------------------------------------\n")

    return had_success, terminal_success


def main():
    # Match your training setup as closely as possible
    env_horizon = 200
    env_dense_reward = True
    use_icm = False

    base_env = make_env(env_horizon, env_dense_reward)

    icm = ICM(obs_dim=60, action_dim=7, feature_dim=128, beta=0.2, forward_scale=0.5)
    icm_optimizer = torch.optim.Adam(icm.parameters(), lr=1e-3)

    env = ICMIntegration(
        env=base_env,
        icm=icm,
        icm_optimizer=icm_optimizer,
        lam=0.1,
        use_intrinsic_reward=use_icm,
        device="cpu",
        icm_batch_size=64,
        icm_batches_per_call=2,
        chunk_size=2000,
    )

    # Option A: load a trained model if you have one
    # model = PPO.load("your_saved_model_path", env=env)

    # Option B: create an untrained model just to test the success plumbing
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        n_steps=256,
        batch_size=64,
    )

    # Run a couple of episodes and inspect the output
    total_had_success = 0
    total_terminal_success = 0
    num_episodes = 2

    for ep in range(num_episodes):
        had_success, terminal_success = run_debug_episode(
            model,
            env,
            episode_idx=ep,
            max_steps=None,   # set to e.g. 20 if you want shorter debug output
        )
        total_had_success += int(had_success)
        total_terminal_success += int(terminal_success > 0.5)

    print("============== FINAL DEBUG SUMMARY ==============")
    print(f"episodes run              : {num_episodes}")
    print(f"episodes with any success : {total_had_success}")
    print(f"episodes with terminal success : {total_terminal_success}")
    print("=================================================")

    env.close()


if __name__ == "__main__":
    main()