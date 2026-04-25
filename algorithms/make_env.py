import robosuite as suite
from robosuite.wrappers import GymWrapper
from stable_baselines3.common.monitor import Monitor


def make_env(horizon=200, dense_reward=True, reward_scale=None):
    make_kwargs = dict(
        env_name="PickPlace",
        robots="Panda",
        has_renderer=False,
        use_camera_obs=False,
        reward_shaping=dense_reward,
        control_freq=20,
        horizon=horizon,
        # simplest object mode for faster training - env is already quite complex with the sparse reward
        single_object_mode=2,
        object_type="can",
    )

    if reward_scale is not None:
        make_kwargs["reward_scale"] = reward_scale

    env = suite.make(**make_kwargs)
    env = GymWrapper(env)
    env = Monitor(env)
    return env