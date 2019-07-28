import numpy as np
import tensorflow as tf

import rl_lib as rl

class PolicyModel(rl.nets.FullyConnectedDNN):

    def __init__(self, input_dims, output_dims, gamma=.99, lr=1e-2, **kwargs):
        super().__init__(input_dims, output_dims, **kwargs, output_activation=None, output_use_bias=False)

        self.gamma = gamma

        gammas_n = 1000
        self.GAMMAS = np.power(gamma*np.ones(gammas_n), np.arange(gammas_n, 0, -1)-1)

        self.pi_s = self.y

        self.actions = tf.placeholder(tf.int64, shape=(None,))
        self.rewards = tf.placeholder(tf.float64, shape=(None,))
        self.gammas = tf.placeholder(tf.float64, shape=(None,))
        self.vs = tf.placeholder(tf.float64, shape=(None,))

        self.pi_s_a = tf.reduce_sum(self.pi_s*tf.one_hot(self.actions, self.output_dims, dtype=tf.float64), axis=1)

        self.advantages = self.gammas*self.rewards-self.vs

        self.loss = -tf.reduce_sum(self.advantages*tf.log(self.pi_s_a))

        self.train = tf.compat.v1.train.AdamOptimizer(lr).minimize(self.loss)



    def policy(self, states, actions=None):
        states = np.atleast_2d(states)

        assert states.shape[1] == self.input_dims

        if actions is None:
            result = self.sess.run(self.pi_s, feed_dict={
                    self.x: states
            })
            assert result.shape[1] == self.output_dims
        else:
            actions = np.atleast_1d(actions)
            result = self.sess.run(self.pi_s_a, feed_dict={
                    self.x: states,
                    self.actions: actions
            })
            assert len(result.shape) == 1

        assert result.shape[0] == states.shape[0]

        return result

    def full_episode_update(self, states, actions, rewards, vs):
        states = np.atleast_2d(states)
        actions = np.atleast_1d(actions)
        rewards = np.atleast_1d(rewards)
        vs = np.atleast_1d(vs)

        size = len(states)
        assert len(actions) == size
        assert len(rewards) == size
        assert len(vs) == size

        self.sess.run(self.train, feed_dict={self.x: states,
                                             self.actions: actions,
                                             self.model.gammas: self.GAMMAS[-size:],
                                             self.model.rewards: rewards,
                                             self.model.vs: vs
                                             })



model = PolicyModel(input_dims=2, output_dims=3, drop_out=.0)
# print(model.policy([[0, 1], [0, 1], [0, 1]], [0, 1, 2]), model.policy([0, 1])[0])
# print((model.policy([[0, 1], [0, 1], [0, 1]], [0, 1, 2]) - model.policy([0, 1])[0])<1e-10)
# # print(model.policy([[0, 1], [0, 1]]))
# # print(model.policy([[0, 1], [0, 1]], [1, 0]))
# exit()
class PGAgent(rl.Agent):

    def __init__(self, state_dims, actions_num, rbf_args=None, nn_args=None, mapper=rl.utils.Mapper):
        super().__init__(state_dims=state_dims, actions_num=actions_num)

        if rbf_args is None:
            rbf_args = {'samplers':None,
                        'constant_samplers':False,
                        'constant_gammas:':True}

        if nn_args is None:
            nn_args = {'hidden_layers':[200, 100],
                       'activations':[tf.nn.relu, tf.nn.relu],
                       'use_biases':[True, True],
                       'output_activation':tf.nn.softmax}

        self.mapper = mapper

        self.policy_net = rl.nets.FullyConnectedDNN(input_dims=state_dims, output_dims=actions_num, **nn_args)

        self.nets = [rl.nets.RBFNet(**rbf_args) for _ in range(self.actions_num)]
        for net in self.nets:
            net.create_net(state_dims, 1)

        self.states = []
        self.actions = []
        self.rewards = []

        self.episode = 0

    def policy_model(self, input_dims, output_dims, **nn_args):
        nn = rl.nets.FullyConnectedDNN(input_dims=input_dims, output_dims=output_dims, **nn_args)


    def policy(self, state):
        res = self.policy_net.predict(state)
        return res

    def full_episode_update_policy(self, states, actions, rewards, gamma=.999):
        pass

    def observe(self, state, action, reward, state_, episode=-1, step=-1):
        super().observe(state, action, reward, state_, episode, step)

        if self.episode < episode:
            self.full_episode_update_policy(self.states, self.actions, self.rewards)

            self.states = []
            self.actions = []
            self.rewards = []

        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.episode = episode

