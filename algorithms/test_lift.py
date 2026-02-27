import robosuite as suite
from robosuite.wrappers import GymWrapper

env = suite.make(
    env_name="Lift",
    robots="Panda",
    has_renderer=False,
    use_camera_obs=False,
)

env = GymWrapper(env)

obs, info = env.reset()
print(type(obs))
print("this is the object shape ", obs.shape)
print("obs dtype:", obs.dtype)
print("action space:", env.action_space)
print("obs space:", env.observation_space)

for _ in range(10):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print("this is the reward checking that it is not null", reward)

env.close()