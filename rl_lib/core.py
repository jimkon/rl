import numpy as np


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

        assert action >= 0 and action < self.actions_num, 'input action {}'.format(action)

        assert isinstance(state_, np.ndarray)
        assert state_.shape == self.state_shape
        pass


def run_env(env, episodes, model, verbose=False, reward_wrapper=lambda reward, done, step: reward):
    ep_rewards = []
    ep_steps = []
    for ep in range(episodes):
        s = env.reset()
        done = False
        total_reward = .0
        steps = 0
        while not done:
            a = model.act(s)

            s_, r, done, _ = env.step(a)

            r = reward_wrapper(r, done, steps)

            total_reward += r

            model.observe(s, a, r, s_, episode=ep, step=steps)

            s = s_

            steps += 1

        ep_rewards.append(total_reward)
        ep_steps.append(steps)
        if verbose:
            print('Ep {} total reward {} after {} steps'.format(ep, total_reward, steps))

    print(
        'Run {} episodes, last 100 average {}, last 100 steps {}'.format(len(ep_rewards), np.average(ep_rewards[-100:]),
                                                                         np.average(ep_steps[-100:])))
    return ep_rewards, ep_steps
