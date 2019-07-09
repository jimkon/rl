import numpy as np
import tensorflow as tf

from rl import *
from rl.utils import epsilon, RBFNet, FullyConnectedDNN


class QLearningAgent(Agent):
    """
    Q-Learning implementation

    act: returns argmax_a_(Q(s, a_)
    observe: apply Q value update
    """

    def __init__(self, state_dims=-1, actions_num=-1, epsilon_factor=1):
        self.state_dims = state_dims
        self.actions_num = actions_num
        self.epsilon_factor = epsilon_factor
        self.episode = -1
        pass

    def act(self, state):
        if self.actions_num > 0 and np.random.random() < epsilon(self.episode * self.epsilon_factor):
            return np.random.randint(self.actions_num)
        super().act(state)
        return np.argmax(self.Q(state))

    def observe(self, state, action, reward, state_, episode=-1, step=-1):
        super().observe(state, action, reward, state_, episode=episode, step=step)

        Q_ = self.Q_learning_formula(state, action, reward, state_)
        self.Q_update(state, action, Q_)

        self.episode = episode

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


class TabularQLearningAgent(QLearningAgent):

    def __init__(self, state_low, state_high, actions_num, bins_per_dim=10, epsilon_factor=1):
        super().__init__(state_dims=len(state_high), actions_num=actions_num, epsilon_factor=epsilon_factor)
        self.bins_per_dim = bins_per_dim
        self.epsilon_factor = epsilon_factor

        self.s_low = np.array(state_low)
        self.s_high = np.array(state_high)

        self.q_low, self.q_high = np.array([*tuple(self.s_low), 0]), np.array(
                [*tuple(self.s_high), self.actions_num - 1])

        shape = (*tuple([self.bins_per_dim] * self.state_dims), self.actions_num)
        self.q_table = np.random.uniform(low=-1, high=1, size=shape)  # self.q_table = np.zeros(shape)

    def act(self, state):
        return super().act(state)

    def observe(self, state, action, reward, state_, episode=-1, step=-1):
        super().observe(state, action, reward, state_, episode=episode, step=step)

    def Q_update(self, s, a, q_value):
        self.q_table[self.Q_index(s, a)] = q_value

    def Q(self, s, a=None):
        if a is None:
            return self.q_table[self.Q_index(s)]
        return self.q_table[self.Q_index(s, a)]

    def Q_index(self, s, a=None):
        s = np.array(s)
        raw_index = (self.q_table.shape[1:] * (s - self.s_low) / (self.s_high - self.s_low))
        s_ind = np.clip(raw_index.astype(np.int), [0] * self.state_dims, [self.bins_per_dim - 1] * self.state_dims)

        if a is None:
            return s_ind[0], s_ind[1], s_ind[2], s_ind[3]
        return s_ind[0], s_ind[1], s_ind[2], s_ind[3], int(a)


class RBFQLearningAgent(QLearningAgent):

    def __init__(self, state_dims, actions_num, samplers=None, constant_samplers=False, constant_gammas=True,
                 epsilon_factor=1):
        super().__init__(state_dims=state_dims, actions_num=actions_num, epsilon_factor=epsilon_factor)
        self.nets = [RBFNet(samplers=samplers, constant_samplers=constant_samplers, constant_gammas=constant_gammas) for
                     _ in range(self.actions_num)]

        for net in self.nets:
            net.create_net(state_dims, 1)  # net.partial_fit(np.zeros(self.state_dims), np.array(0))

    def Q(self, state, action=None):
        if action is None:
            return np.array([self.nets[int(a)].predict(state) for a in range(self.actions_num)])
        return self.nets[int(action)].predict(state)

    def Q_update(self, s, a, q_value):
        # print("update", s, a, q_value, self.Q(s, a))
        self.nets[int(a)].partial_fit(s, np.array(q_value))


class NNQLearningAgent(QLearningAgent):

    def __init__(self, state_dims, actions_num, hidden_layers=[200, 100], activations=[tf.nn.relu, tf.nn.relu],
                 drop_out=True, drop_out_rate=.3, lr=1e-2, epsilon_factor=1):

        super().__init__(state_dims=state_dims, actions_num=actions_num, epsilon_factor=epsilon_factor)

        self.nets = [FullyConnectedDNN(state_dims, 1, hidden_layers=hidden_layers, activations=activations,
                                       drop_out=drop_out, drop_out_rate=drop_out_rate, lr=lr) for _ in
                     range(self.actions_num)]

    def Q(self, state, action=None):
        if action is None:
            return np.array([self.nets[int(a)].predict(state)[0] for a in range(self.actions_num)])
        return self.nets[int(action)].predict(state)[0]

    def Q_update(self, s, a, q_value):
        print("update", s, a, q_value, self.Q(s, a))
        self.nets[int(a)].partial_fit(np.array(s), np.array(q_value))
