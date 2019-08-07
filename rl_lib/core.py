import numpy as np
import pandas as pd
import time

from rl_lib.utils.utils import average_reward


class Agent:

    def __init__(self, state_dims, actions_num):
        self.state_dims = state_dims
        self.actions_num = actions_num

        self.state_shape = tuple([self.state_dims])
        assert state_dims > 0
        assert actions_num > 0

    def act(self, state):
        """
        :param list state: state vector
        :return list: action vector
        """

        assert isinstance(state, np.ndarray)
        assert state.shape == self.state_shape

    def observe(self, state, action, reward, state_, episode=-1, step=-1):
        assert isinstance(state, np.ndarray)
        assert state.shape == self.state_shape

        assert 0 <= action < self.actions_num, 'input action {}'.format(action)

        assert isinstance(state_, np.ndarray)
        assert state_.shape == self.state_shape
        pass


def run(env, agent, episodes=1000, verbose=2):
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

            step_count += 1

        if verbose >= 2:
            time_took = 1e3 * (time.time() - episode_start_time)
            print('Episode {} finished after {} steps with total reward {:.1f} in  {:.1f} ms ({:.2f} per step)'.format(
                episode, step_count, episode_reward, time_took, time_took / step_count))

    df = pd.concat([df, pd.DataFrame(np.array(states), columns=['state1', 'state2'])], axis=1)
    df = pd.concat([df, pd.DataFrame(np.array(actions), columns=['action'])], axis=1)
    df = pd.concat([df, pd.DataFrame(np.array(rewards), columns=['reward'])], axis=1)
    df = pd.concat([df, pd.DataFrame(np.array(states_), columns=['state1_', 'state2_'])], axis=1)
    df = pd.concat([df, pd.DataFrame(np.array(dones), columns=['dones'])], axis=1)
    df['episode'] = df['dones'].cumsum() - df['dones']  # number of episode
    run_time = (time.time() - run_start_time)
    if verbose >= 1:
        print("Run {} episodes in {:.02f} seconds. Average reward {}".format(episodes, run_time, average_reward(df)))
    return df
