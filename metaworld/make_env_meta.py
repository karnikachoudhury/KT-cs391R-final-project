import gymnasium as gym
import metaworld  # needed so the envs register
from stable_baselines3.common.monitor import Monitor
import random

def make_env(
    horizon: int = 150,
    dense_reward: bool = True,   # kept for compatibility with your old code
    reward_scale=None,           # unused, kept for compatibility
    seed: int = 0,
    env_name: str = "drawer-open-v3",
):
    """
    Create a single Meta-World MT1 environment.

    Notes:
    - Meta-World uses Gymnasium API.
    - We keep dense_reward/reward_scale args so the rest of your code
      does not need a major rewrite.
    - Depending on your installed Meta-World version, there may not be a
      simple public gym.make flag for dense vs sparse reward. So this code
      uses the environment's default reward behavior.
    """

    env = gym.make(
        "Meta-World/MT1",
        env_name=env_name,
        seed=seed,
        max_episode_steps=horizon,
    )

    env = Monitor(env)
    return env
# class MetaWorldRandomizedEnv(gym.Wrapper):
#     def __init__(self, env, tasks):
#         super().__init__(env)
#         self.tasks = tasks

#     def reset(self, **kwargs):
#         # pick a new random task every episode
#         task = random.choice(self.tasks)
#         self.env.set_task(task)
#         return self.env.reset(**kwargs)

# def make_env(
#     horizon=150,
#     seed=0,
#     env_name="drawer-open-v3",
# ):
#     ml1 = metaworld.MT1(env_name, seed=seed)

#     env = ml1.train_classes[env_name]()
#     env._max_episode_steps = horizon

#     # wrap with randomization
#     env = MetaWorldRandomizedEnv(env, ml1.train_tasks)

#     env = Monitor(env)
#     return env