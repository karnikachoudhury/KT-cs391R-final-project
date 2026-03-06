import robosuite as suite
from robosuite.wrappers import GymWrapper
from stable_baselines3.common.monitor import Monitor

def make_env():
    env = suite.make(
        env_name = "Lift",
        robots = "Panda",
        has_renderer = False,
        use_camera_obs = False,
        reward_shaping = True,
        control_freq = 20,
        horizon = 200,
    )
    env = GymWrapper(env)
    env = Monitor(env)
    return env