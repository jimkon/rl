import gym


env = gym.make("MountainCar-v0")

state_low, state_high = env.observation_space.low, env.observation_space.high
actions_num = env.action_space.n

print(state_low, state_high, actions_num)


print("Mountain car standards imported", dir())
