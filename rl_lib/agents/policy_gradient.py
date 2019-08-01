import numpy as np
import tensorflow as tf

import rl_lib as rl


class PolicyModel(rl.nets.FullyConnectedDNN):

    def __init__(self, input_dims, output_dims, gamma=.99, lr=1e-2, **kwargs):
        super().__init__(input_dims, output_dims, **kwargs, output_activation=tf.nn.softmax, output_use_bias=False)

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
                                             self.gammas: self.GAMMAS[-size:],
                                             self.rewards: rewards,
                                             self.vs: vs
                                             })

    def td_update(self, state, action, reward, v):
        raise NotImplementedError


class ValueModel(rl.nets.FullyConnectedDNN):

    def __init__(self, input_dims, gamma=.99, lr=1e-2, **kwargs):
        super().__init__(input_dims, output_dims=1, output_activation=None, output_use_bias=False, **kwargs)

    def value(self, states):
        states = np.atleast_2d(states)

        assert states.shape[1] == self.input_dims

        return self.predict(states).flatten()

    def full_episode_update(self, states, rewards):

        states = np.atleast_2d(states)
        rewards = np.atleast_2d(rewards)

        size = len(states)
        assert len(rewards) == size

        self.fit(states, rewards)

    def td_update(self, state, reward):
        raise NotImplementedError


model = PolicyModel(input_dims=2, output_dims=3, drop_out=.0)
# print(model.policy([[0, 1], [0, 1], [0, 1]], [0, 1, 2]), model.policy([0, 1])[0])
# print((model.policy([[0, 1], [0, 1], [0, 1]], [0, 1, 2]) - model.policy([0, 1])[0])<1e-10)
# # print(model.policy([[0, 1], [0, 1]]))
# # print(model.policy([[0, 1], [0, 1]], [1, 0]))
# exit()
class PGAgent(rl.Agent):

    def __init__(self, state_dims, actions_num, actor_args=None, critic_args=None, mapper=rl.utils.Mapper, td_update=False):
        super().__init__(state_dims=state_dims, actions_num=actions_num)

        assert not td_update, 'Temporal Difference update is not implemented yet'
        self.td_update = td_update

        if actor_args is None:
            actor_args = {}

        if critic_args is None:
            critic_args = {}

        self.mapper = mapper

        self.actor = PolicyModel(input_dims=state_dims, output_dims=actions_num, **actor_args)

        self.critic = ValueModel(input_dims=state_dims, **critic_args)

        self.states = []
        self.actions = []
        self.rewards = []

        self.episode = 0

    def act(self, state):
        probs = self.policy(state)
        return np.random.choice(len(probs), 1, p=probs)

    def observe(self, state, action, reward, state_, episode=-1, step=-1):
        super().observe(state, action, reward, state_, episode, step)

        if self.td_update:
            self.actor.td_update(state, action, reward)
            self.critic.td_update(state, reward)
        else:
            if self.episode < episode:
                self.actor.full_episode_update(self.states, self.actions, self.rewards)
                self.critic.full_episode_update(self.states, self.rewards)

                self.states = []
                self.actions = []
                self.rewards = []

            self.states.append(state)
            self.actions.append(action)
            self.rewards.append(reward)
            self.episode = episode

    def policy(self, state):
        res = self.actor.predict(state)
        return res

