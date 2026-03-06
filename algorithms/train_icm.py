import torch
from stable_baselines3 import PPO
from make_env import make_env
from icm import ICM
from icm_integration import ICMIntegration

base_environment = make_env()
icm = ICM(obs_dim=60, action_dim=7, feature_dim=128, beta = 0.4, forward_scale=0.5)
icm_optimizer = torch.optim.Adam(icm.parameters(), lr=1e-3)

env = ICMIntegration(
    env=base_environment,
    icm=icm,
    icm_optimizer=icm_optimizer,
    lam=0.01,
    use_intrinsic_reward=True,
    device="cpu",
    batch_size=256,
    icm_updates_per_call=1000,
)

model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    n_steps=2048,
    batch_size=64,
    tensorboard_log="./ppo_lift_tensorboard/thom_test_more_icm/",
)

total_steps = 100000
chunk = 2000 # each run is 200 steps

for i in range(total_steps // chunk):
    print(f"Step {i} / {total_steps // chunk}")
    model = model.learn(total_timesteps=chunk, reset_num_timesteps=False)

    icm_logs = env.train_icm()
    if icm_logs.get("icm_loss", 0.0) != 0.0:
        print("ICM Logs:")
        for key, value in icm_logs.items():
            print(f"  {key}: {value}")

def test_model(model, env, eval_episodes=100, do_print=False):
# test model afterwards
    env.icm.eval()
    with torch.no_grad():
        eval_episodes = 100
        successes = 0
        had_success = 0
        for i in range(eval_episodes):
        #    print(f"Eval Ep {i} / {eval_episodes}")
            tmp_success = False
            obs, info = env.reset()
            done = False
            steps = 0
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                if info.get("success", False):
                    had_success += 1
                    tmp_success = True
                steps += 1
            print(f"Episode {i}:\n  Steps: {steps}\n  Had Success: {tmp_success}\n  End Success: {info.get('success', "not found")}\n  R_e: {info.get('reward_extrinsic')}\n  R_i: {info.get('reward_intrinsic')}\n  R_total: {info.get('reward_total')}")
            if info.get("success", False):
                successes += 1

        success_rate = 100 * successes / eval_episodes
        print(f"End success rate: {success_rate:.2%}%")
        print(f"Had success rate: {100 * had_success / eval_episodes:.2f}%")

test_model(model, env, do_print=True)