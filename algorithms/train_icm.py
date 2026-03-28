import os
import torch
from stable_baselines3 import PPO
from make_env import make_env
from icm import ICM
from icm_integration import ICMIntegration



def test_model(model, env, output_file, eval_episodes=100):
# test model afterwards
    output_file.write("\nTesting model\n")
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
            output_file.write(f"Episode {i}:\n  Steps: {steps}\n  Had Success: {tmp_success}\n  End Success: {info.get('success', "not found")}\n  R_e: {info.get('reward_extrinsic')}\n  R_i: {info.get('reward_intrinsic')}\n  R_total: {info.get('reward_total')}\n")
            if info.get("success", False):
                successes += 1
        # TODO: report mean reward
        success_rate = 100 * successes / eval_episodes
        had_success_rate = 100 * had_success / eval_episodes
        #print(f"End success rate: {success_rate:.2}%")
        output_file.write(f"Had success rate: {had_success_rate:.2}%\n")
        return (success_rate, had_success_rate)
    


def train_icm(
        env_horizon=200, 
        env_dense_reward=True, 
        icm_beta=0.2, 
        icm_lr=1e-3, 
        env_lambda=0.1, 
        use_icm=True, 
        eps_per_epoch=10, 
        epochs=300
    ):
    output_dir = os.path.join("outputs", f"icm_{use_icm}_epochs_{epochs}_horizon_{env_horizon}_eps_{eps_per_epoch}_dense_{env_dense_reward}_beta_{icm_beta}_lr_{icm_lr}_lambda_{env_lambda}/")
    os.mkdir(output_dir)
    output_file = open(os.path.join(output_dir, "output.txt"), "w")
    total_steps = eps_per_epoch * env_horizon * epochs
    chunk = eps_per_epoch * env_horizon
    
    base_environment = make_env(horizon=env_horizon, dense_reward=env_dense_reward)
    icm = ICM(obs_dim=60, action_dim=7, feature_dim=64, beta = icm_beta, forward_scale=0.5)
    icm_optimizer = torch.optim.Adam(icm.parameters(), lr=icm_lr)

    env = ICMIntegration(
        env=base_environment,
        icm=icm,
        icm_optimizer=icm_optimizer,
        output_file=output_file,
        lam=env_lambda,
        use_intrinsic_reward=use_icm,
        device="cpu",
        icm_batch_size=64,
        icm_batches_per_call=100,
        chunk_size=chunk
    )


    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        n_steps=chunk, # is this the right value? should maybe be horizon
        batch_size=64,
        tensorboard_log=output_dir,
    )

    for i in range(total_steps // chunk):
        output_file.write(f"\nEpoch {i + 1} / {total_steps // chunk}\n")
        model = model.learn(total_timesteps=chunk, reset_num_timesteps=False)

        icm_logs = env.train_icm()
        if icm_logs.get("icm_loss", 0.0) != 0.0:
            output_file.write("ICM Logs:\n")
            for key, value in icm_logs.items():
                output_file.write(f"  {key}: {value}\n")

    test_model(model, env, output_file=output_file)

if __name__ == "__main__":
    train_icm(
        env_horizon=200,
        env_dense_reward=True,
        icm_beta=0.2,
        icm_lr=1e-3,
        env_lambda=0.1,
        use_icm=True,
        eps_per_epoch=10,
        epochs=3
    )