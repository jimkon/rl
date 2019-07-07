import numpy as np

from rl import *
from rl.utils import epsilon


class QLearningAgent(Agent):
    """
    Q-Learning implementation

    act: returns argmax_a_(Q(s, a_)
    observe: apply Q value update
    """

    def act(self, state):
        super().act(state)
        return np.argmax(self.Q(state))

    def observe(self, state, action, reward, state_, episode=-1, step=-1):
        super().observe(state, action, reward, state_, episode=-1, step=-1)
        Q_ = self.Q_learning_formula(state, action, reward, state_)
        self.Q_update(state, action, Q_)

    def Q_learning_formula(self, s, a, r, s_, alpha=10e-2, gamma=.9):
        """
        Q(s, a) = Q(s, a) + a*[ r + gamma * argmax_a_(Q(s_, a_)) - Q(s, a) ]
        """
        incr = alpha * (r + gamma * np.max(self.Q(s_)) - self.Q(s, a))
        return self.Q(s, a) + incr

    def Q(self, state, action=None):
        """
        :param list state: state vector
        :param list action: action vector
        :return list: action vector
        """
        return 0

    def Q_update(self, s, a, q_value):
        pass


class QLearningTabularAgent(QLearningAgent):

    def __init__(self, state_low, state_high, actions_num, bins_per_dim=10, epsilon_factor=1):

        self.episode = -1
        self.bins_per_dim = bins_per_dim
        self.epsilon_factor = epsilon_factor

        self.s_low = np.array(state_low)
        self.s_high = np.array(state_high)

        self.state_dimensions = len(state_high)
        self.actions_num = actions_num

        self.q_low, self.q_high = np.array([*tuple(self.s_low), 0]), np.array(
                [*tuple(self.s_high), self.actions_num - 1])

        shape = (*tuple([self.bins_per_dim] * self.state_dimensions), self.actions_num)
        self.q_table = np.random.uniform(low=-1, high=1, size=shape)
        # self.q_table = np.zeros(shape)

    def act(self, state):
        if np.random.random() < epsilon(self.episode * self.epsilon_factor):
            return np.random.randint(self.actions_num)
        return super().act(state)

    def observe(self, state, action, reward, state_, episode=-1, step=-1):
        super().observe(state, action, reward, state_, episode=episode, step=step)
        self.episode = episode

    def Q_update(self, s, a, q_value):
        self.q_table[self.Q_index(s, a)] = q_value

    def Q(self, s, a=None):
        if a is None:
            return self.q_table[self.Q_index(s)]
        return self.q_table[self.Q_index(s, a)]

    def Q_index(self, s, a=None):
        s = np.array(s)
        raw_index = (self.q_table.shape[1:] * (s - self.s_low) / (self.s_high - self.s_low))
        s_ind = np.clip(raw_index.astype(np.int), [0] * self.state_dimensions,
                        [self.bins_per_dim - 1] * self.state_dimensions)

        if a is None:
            return s_ind[0], s_ind[1], s_ind[2], s_ind[3]
        return s_ind[0], s_ind[1], s_ind[2], s_ind[3], int(a)
