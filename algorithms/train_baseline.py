from stable_baselines3 import PPO
from make_env import make_env
# uhh this is just vanilla PPO testing
env = make_env()

model = PPO("MlpPolicy", 
            env, 
            verbose = 1,
            n_steps = 2048,
            batch_size = 64,
            gamma = 0.99,
            gae_lambda = 0.95,
            n_epochs = 10,
            learning_rate = 3e-4,
            clip_range = 0.2,
            tensorboard_log="./ppo_lift_tensorboard/",
)
model.learn(total_timesteps=300_000,
             tb_log_name="baseline")
model.save("ppo_lift_baseline")
env.close()