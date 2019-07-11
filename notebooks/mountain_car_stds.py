import gym
import matplotlib.pyplot as plt
import numpy as np


class MountainCarRewardWrapper(gym.RewardWrapper):

    def __init__(self, env):
        super().__init__(env)
        self.state = None
        self.low = self.observation_space.low
        self.high = self.observation_space.high
        self.center = np.array([-.45, .0])
        self.normalize_scaler = np.array([1.05, .07])
        self.done = False

    def reset(self, **kwargs):
        self.state = self.env.reset(**kwargs)
        return self.state

    def step(self, action):
        self.state, reward, self.done, info = self.env.step(action)
        return self.state, self.reward(reward), self.done, info

    def reward(self, reward):
        self.state = np.array([-.45, -.07])
        res = np.linalg.norm((self.state-self.center)/self.normalize_scaler)
        res += 300 if self.done else .0
        return res


unwrapped_env = gym.make("MountainCar-v0")
env = MountainCarRewardWrapper(unwrapped_env)

state_low, state_high = env.observation_space.low, env.observation_space.high
actions_num = env.action_space.n

print(state_low, state_high, actions_num)


print("Mountain car standards imported", dir())




def uniform_state_grid(points_per_axis=100):
    s1, s2 = np.linspace(state_low[0], state_high[0], points_per_axis), np.linspace(state_low[1],
                                                                                          state_high[1],
                                                                                          points_per_axis)
    return np.array([np.array([x, y]) for x in s1 for y in s2])


def plot(xys, v):
    plt.scatter(xys[:, 0], xys[:, 1], c=v, s=10)
    plt.grid(True)
    plt.colorbar()


def plot_Q(qlearning_agent, a=None):
    xys = uniform_state_grid()
    actions = range(actions_num) if a is None else a
    plt.figure(figsize=(15, 5))
    for action in actions:
        plt.subplot(1, len(actions), action + 1)
        plt.title("Q(s in S, action = {})".format(action))
        Qs = np.array([qlearning_agent.Q(xy, action) for xy in xys])
        plot(xys, Qs)

    plt.show()
