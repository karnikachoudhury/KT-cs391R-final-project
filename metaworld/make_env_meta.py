import gymnasium as gym
import metaworld  
from stable_baselines3.common.monitor import Monitor
import random

def make_env(
    horizon: int = 150,
    dense_reward: bool = True,
    reward_scale=None,           
    seed: int = 0,
    env_name: str = "drawer-open-v3",
):
    env = gym.make(
        "Meta-World/MT1",
        env_name=env_name,
        seed=seed,
        max_episode_steps=horizon,
    )

    env = Monitor(env)
    return env
