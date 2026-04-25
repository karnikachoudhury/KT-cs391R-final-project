import robosuite as suite
from robosuite.wrappers import GymWrapper
from stable_baselines3.common.monitor import Monitor


def make_env(horizon=200, dense_reward=True, reward_scale=None, render=False, task='door'):
    if task == 'door':
        env_name = "Door"
    elif task == 'pickplace':
        env_name = "PickPlace"
    make_kwargs = dict(
        env_name=env_name,
        robots="Panda",
        has_renderer=False,
        has_offscreen_renderer=render,
        use_camera_obs=False,
        reward_shaping=dense_reward,
        control_freq=20,
        horizon=horizon,
    )
    
    print(make_kwargs)
    if env_name == "PickPlace":
        # make the task easier / more consistent at first
        make_kwargs["object_type"] = "can"
        make_kwargs["single_object_mode"] = 2
    if reward_scale is not None:
        make_kwargs["reward_scale"] = reward_scale
    if render: 
        make_kwargs["camera_names"] = "frontview"
        make_kwargs["camera_heights"] = 512
        make_kwargs["camera_widths"] = 512
    
    env = suite.make(**make_kwargs)
    env = GymWrapper(env)
    env = Monitor(env)
    return env
