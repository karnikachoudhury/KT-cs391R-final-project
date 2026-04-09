import robosuite as suite
from robosuite.wrappers import GymWrapper
from stable_baselines3.common.monitor import Monitor

def make_env(horizon=200, dense_reward=True, reward_scale=None):
    # allow optional reward scaling to amplify sparse/dense rewards for curriculum
    make_kwargs = dict(
        env_name = "Door",
        robots = "Panda",
        has_renderer = False,
        use_camera_obs = False,
        reward_shaping = dense_reward,
        control_freq = 20,
        horizon = horizon,
    )
    if reward_scale is not None:
        make_kwargs["reward_scale"] = reward_scale

    env = suite.make(**make_kwargs)
    env = GymWrapper(env)
    env = Monitor(env)
    return env