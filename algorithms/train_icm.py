import torch
from stable_baselines3 import PPO
from make_env import make_env
from icm import ICM
from icm_integration import ICMIntegration

base_environment = make_env()
icm = ICM(obs_dim=60, action_dim=7, feature_dim=128, beta = 0.2, forward_scale=0.5)
icm_optimizer = torch.optim.Adam(icm.parameters(), lr=1e-3)
chunk = 10000
n_steps = 2000
env = ICMIntegration(
    env=base_environment,
    icm=icm,
    icm_optimizer=icm_optimizer,
    lam=0.1,
    use_intrinsic_reward=True,
    device="cpu",
    icm_batch_size=64,
    icm_batches_per_call=100,
    chunk_size=chunk
)

model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    n_steps=n_steps,
    batch_size=64,
    tensorboard_log=".karnika_messing/",
)

total_steps = 1000000

for i in range(total_steps // chunk):
    print(f"\nStep {i + 1} / {total_steps // chunk}")
    model = model.learn(total_timesteps=chunk, reset_num_timesteps=False, tb_log_name="icm")

    icm_logs = env.train_icm()
    if icm_logs.get("icm_loss", 0.0) != 0.0:
        print("ICM Logs:")
        for key, value in icm_logs.items():
            print(f"  {key}: {value}")
        model.logger.record("icm/icm_loss", icm_logs["icm_loss"])
        model.logger.record("icm/inv_loss", icm_logs["inv_loss"])
        model.logger.record("icm/fwd_loss", icm_logs["fwd_loss"])
        model.logger.record("icm/r_int_mean", icm_logs["r_int_mean"])

        # write them to TensorBoard at the current timestep
        model.logger.dump(model.num_timesteps)

def test_model(model, env, eval_episodes=300000, do_print=False):
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
                    tmp_success = True
                steps += 1
            if tmp_success:
                had_success += 1
            # need to take avgs of these stats, just reporting last of episode rn
            print(f"Episode {i}:\n  Steps: {steps}\n  Had Success: {tmp_success}\n  End Success: {info.get('success', "not found")}\n  R_e: {info.get('reward_extrinsic')}\n  R_i: {info.get('reward_intrinsic')}\n  R_total: {info.get('reward_total')}")
            if info.get("success", False):
                successes += 1
        # TODO: report mean reward
        success_rate = 100 * successes / eval_episodes
        had_success_rate = 100 * had_success / eval_episodes
        #print(f"End success rate: {success_rate:.2}%")
        print(f"Had success rate: {had_success_rate:.2}%")
        return (success_rate, had_success_rate)

test_model(model, env, do_print=True)