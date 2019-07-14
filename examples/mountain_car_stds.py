import numpy as np
import time
import gym
import pandas as pd

from rl_lib.utils.utils import running_average
from rl_lib.agents.q_learning import QLearningAgent

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
plt.style.use("bmh")


class MountainCarRewardWrapper(gym.RewardWrapper):

    def __init__(self, env):
        super().__init__(env)
        self.state = None
        self.low = self.observation_space.low
        self.high = self.observation_space.high
        self.center = np.array([-.45, .0])
        self.normalize_scaler = np.array([1.05, .07])

    def reset(self, **kwargs):
        self.state = self.env.reset(**kwargs)
        return self.state

    def step(self, action):
        self.state, reward, done, info = self.env.step(action)
        return self.state, self.reward(reward), done, info

    def reward(self, reward):
        won = self.state[0]>.5
        res = np.linalg.norm((self.state-self.center)/self.normalize_scaler)
        res += -1.4142# + (1 if won else .0)
        return res


def uniform_state_grid(points_per_axis=31):
    s1, s2 = np.linspace(state_low[0], state_high[0], points_per_axis),\
             np.linspace(state_low[1], state_high[1], points_per_axis)
    return np.array([np.array([x, y]) for x in s1 for y in s2])


def run(agent, episodes=1000, verbose=2):
    run_start_time = time.time()
    df = pd.DataFrame()
    states, actions, rewards, states_, dones = [], [], [], [], []

    for episode in range(episodes):
        episode_start_time = time.time()

        state = env.reset()
        episode_reward = 0
        step_count = 0
        done = False
        while not done:

            action = agent.act(state)
            state_, reward, done, _ = env.step(action)

            agent.observe(state, action, reward, state_, episode=episode, step=step_count)

            episode_reward += reward

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            states_.append(state_)
            dones.append(done)

            state = state_

            step_count+= 1

        if verbose >= 2:
            time_took = 1e3*(time.time()-episode_start_time)
            print('Episode {} finished after {} steps with total reward {:.1f} in  {:.1f} ms ({:.2f} per step)'.format(episode,
                                                                                   step_count,
                                                                                   episode_reward,
                                                                                   time_took,
                                                                                   time_took/step_count))

    df = pd.concat([df, pd.DataFrame(np.array(states), columns=['state1', 'state2'])], axis=1)
    df = pd.concat([df, pd.DataFrame(np.array(actions), columns=['action'])], axis=1)
    df = pd.concat([df, pd.DataFrame(np.array(rewards), columns=['reward'])], axis=1)
    df = pd.concat([df, pd.DataFrame(np.array(states_), columns=['state1_', 'state2_'])], axis=1)
    df = pd.concat([df, pd.DataFrame(np.array(dones), columns=['dones'])], axis=1)
    df['episode'] = df['dones'].cumsum()-df['dones'] # number of episode
    run_time = (time.time()-run_start_time)
    if verbose >= 1:
        print("Run {} episodes in {:.02f} seconds. Average reward {}".format(episodes, run_time, average_reward(df)))
    return df


def best_episode(df):
    rewards = df.groupby(['episode']).agg({
                                                  'episode': 'first',
                                                  'reward' : 'sum'
                                                  })
    episode = int(rewards.loc[rewards['reward'].idxmax()]['episode'])
    return episode


def episode_rewards(df):
    return df.groupby(['episode']).agg({'reward' : 'sum'})


def average_reward(df):
    rewards = episode_rewards(df)
    return rewards['reward'].mean()


def plot(xys, v):
    plt.scatter(xys[:, 0], xys[:, 1], c=v, s=80, marker='s')
    plt.grid(True)


def show_Q(qlearning_agent):
    xys = uniform_state_grid()
    Q = np.array([np.array(qlearning_agent.Q(xy)) for xy in xys])
    plt.figure(figsize=(15, 5))
    for action in range(Q.shape[1]):
        plt.subplot(1, Q.shape[1], action + 1)
        plt.title("Q(s in S, action = {})".format(action))
        Qs = Q[:, action]
        plot(xys, Qs)
        plt.colorbar(orientation='horizontal')
    plt.tight_layout()
    plt.show()


def plot_state_path(df_ep, episode=0):
    plt.plot(df_ep['state1'], df_ep['state2'], linewidth=.5, label='episode {}'.format(episode))
    plt.scatter([df_ep['state1'][0]], [df_ep['state2'][0]], c='g', marker='^')
    plt.scatter([df_ep['state1_'][len(df_ep['state1_'])-1]], [df_ep['state2_'][len(df_ep['state2_'])-1]], c='r',
                marker='v')
    plt.xlabel('pos')
    plt.ylabel('vel')
    plt.title('States')
    plt.legend()

def plot_reward(df_ep, episode=0):
    plt.plot(df_ep['reward'], label='total(ep={})={},'.format(episode, df_ep['reward'].sum()))
    plt.xlabel('steps')
    plt.ylabel('reward')
    plt.legend()

def plot_policy(agent):
    xys = uniform_state_grid()

    epsilon_enabled = agent.epsilon_enabled()
    if isinstance(agent, QLearningAgent) and epsilon_enabled:
        agent.disable_epsilon()
    actions = np.array([agent.act(xy) for xy in xys])
    if isinstance(agent, QLearningAgent) and epsilon_enabled:
        agent.enable_epsilon()

    plot(xys, actions)
    plt.colorbar(ticks=[0, 1, 2])
    plt.xlabel('pos')
    plt.ylabel('vel')
    plt.title("Policy")

def plot_rewards(df):
    rewards = episode_rewards(df)
    steps = df.groupby(['episode']).agg({'reward': 'count'})

    plt.plot(rewards, label='avg reward {:.02f}'.format(average_reward(df)))
    plt.plot(running_average(rewards), label='running avg')
    plt.plot(running_average(steps), label='steps')
    plt.xlabel('episode')
    plt.ylabel('reward')
    plt.title('rewards')
    plt.legend()

def plot_state_usage(df):
    x, y = df['state1'], df['state2']
    plt.hist2d(x, y, bins=40, norm=LogNorm())
    plt.xlim(state_low[0], state_high[0])
    plt.ylim(state_low[1], state_high[1])
    plt.colorbar()
    plt.xlabel('pos')
    plt.ylabel('vel')
    plt.title("exploration")

def plot_action_usage(df):
    actions = df['action']
    plt.hist(actions)
    plt.colorbar()
    plt.xlabel('actions')
    plt.ylabel('%')
    plt.title("usage")

def show_episode(df, episode=-1):
    if episode<0:
        episode = best_episode(df)

    df_ep = df[df['episode'] == episode].reset_index()

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plot_state_path(df_ep, episode)

    plt.subplot(1, 2, 2)
    plot_reward(df_ep, episode)

    plt.tight_layout()
    plt.show()

def show_progress(df, agent):
    plt.figure(figsize=(15, 10))
    plt.subplot(2, 2, 1)
    plot_rewards(df)

    plt.subplot(2, 2, 3)
    plot_state_usage(df)

    plt.subplot(2, 2, 4)
    plot_policy(agent)

    plt.subplot(2, 2, 2)
    plot_action_usage(df)

    plt.tight_layout()
    plt.show()

################### VARIABLES ###################
unwrapped_env = gym.make("MountainCar-v0")
env = MountainCarRewardWrapper(unwrapped_env)

state_low, state_high = np.array(env.observation_space.low, np.float64),\
                        np.array(env.observation_space.high, np.float64)
actions_num = env.action_space.n