import torch
from stable_baselines3 import PPO
from make_env import make_env
from icm import ICM
from icm_integration import ICMIntegration

base_environment = make_env()

icm = ICM(obs_dim=60, action_dim=7, feature_dim=128, beta = 0.2, forward_scale=0.5)
icm_optimizer = torch.optim.Adam(icm.parameters(), lr=1e-3)

env = ICMIntegration(
    env=base_environment,
    icm=icm,
    icm_optimizer=icm_optimizer,
    lam=0.1,
    use_intrinsic_reward=False, 
    device="cpu",
    batch_size=256,
    icm_updates_per_call=2,
)

model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    n_steps=2048,
    batch_size=64,
    tensorboard_log="./ppo_lift_tensorboard/",
)

total_steps = 300000
chunk = 2048

for _ in range(total_steps // chunk):
    model.learn(total_timesteps=chunk, reset_num_timesteps=False)

    icm_logs = env.train_icm()
    if icm_logs.get("icm_loss", 0.0) == 0.0:
        print("ICM logs:", icm_logs)
