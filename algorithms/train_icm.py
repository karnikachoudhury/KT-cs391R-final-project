import torch
from stable_baselines3 import PPO
from make_env import make_env
from icm import ICM
from icm_integration import ICMIntegration
import os

def test_model(model, env, output_dir, run_name, eval_episodes=10):
# test model afterwards
    output_file = open(os.path.join(output_dir, f"test_{run_name}.txt"), "w")
    output_file.write("\nTesting model\n")
    env.icm.eval()
    with torch.no_grad():
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
        output_file.close()
        return (success_rate, had_success_rate)

def train_icm(
        env_horizon=200, 
        env_dense_reward=True, 
        icm_beta=0.2, 
        icm_lr=1e-3, 
        env_lambda=0.1, 
        use_icm=True, 
        eps_per_update=10, # n_steps = this * horizon
        updates_per_epoch=5, # chunks = this * n_steps
        epochs=1,
        output_dir=None,
    ): 
    if output_dir == None:
        # may not be the ebst way to label things but we can change later, just want to be able to easily find different runs
        output_dir = os.path.join("outputs", f"icm_{use_icm}_epochs_{epochs}_horizon_{env_horizon}_eps_{eps_per_update}_updates_{updates_per_epoch}_dense_{env_dense_reward}_beta_{icm_beta}_lr_{icm_lr}_lambda_{env_lambda}/")
        os.makedirs(output_dir, exist_ok=True)
    else:
        output_dir = os.path.join("outputs", output_dir)
        os.makedirs(output_dir, exist_ok=True)
    print(f"Output dir: {output_dir}")

    base_environment = make_env(env_horizon, env_dense_reward)
    icm = ICM(obs_dim=60, action_dim=7, feature_dim=128, beta = icm_beta, forward_scale=0.5)
    icm_optimizer = torch.optim.Adam(icm.parameters(), lr=icm_lr)
    n_steps = eps_per_update * env_horizon
    chunk = n_steps * updates_per_epoch
    total_steps = chunk * epochs

    env = ICMIntegration(
        env=base_environment,
        icm=icm,
        icm_optimizer=icm_optimizer,
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
        n_steps=n_steps,
        batch_size=64,
        tensorboard_log=output_dir,
    )


    for i in range(total_steps // chunk):
        print(f"\nStep {i + 1} / {total_steps // chunk}\n")
        model = model.learn(total_timesteps=chunk, reset_num_timesteps=False, tb_log_name="icm")

        icm_logs = env.train_icm()
        if icm_logs.get("icm_loss", 0.0) != 0.0:
            print("ICM Logs:\n")
            for key, value in icm_logs.items():
                print(f"  {key}: {value}\n")
            model.logger.record("icm/icm_loss", icm_logs["icm_loss"])
            model.logger.record("icm/inv_loss", icm_logs["inv_loss"])
            model.logger.record("icm/fwd_loss", icm_logs["fwd_loss"])
            model.logger.record("icm/r_int_mean", icm_logs["r_int_mean"])

            # write them to TensorBoard at the current timestep
            model.logger.dump(model.num_timesteps)


    test_model(model, env, output_dir, "final")



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
        output = os.path.join(cur_dir, "output.txt")
        with open(output, "w") as out_f:
            out_f.writelines(lines)

if __name__ == "__main__":
    # put configurations for the run here. redirect output to "outputs/results.txt" and 
    # outputs for each run will get copied over to an output.txt file in the respective dir
    # can also specify output dir as a parameter
    
    train_icm(        
        env_horizon=200, 
        env_dense_reward=True, 
        icm_beta=0.2, 
        icm_lr=1e-3, 
        env_lambda=0.1, 
        use_icm=True, 
        eps_per_update=10, # n_steps = this * horizon
        updates_per_epoch=5, # chunks = this * n_steps
        epochs=100,

    )
    train_icm(        
        env_horizon=400, 
        env_dense_reward=True, 
        icm_beta=0.2, 
        icm_lr=1e-3, 
        env_lambda=0.1, 
        use_icm=False, 
        eps_per_update=10,
        updates_per_epoch=5,
        epochs=100,
    )

    copy_outputs("results.txt")