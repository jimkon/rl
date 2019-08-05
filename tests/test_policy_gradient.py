import unittest
from rl_lib.agents.policy_gradient import *


class TestPolicyModel(unittest.TestCase):

    def setUp(self):
        self.model = PolicyModel(input_dims=2, output_dims=3, drop_out=.0, gamma=.9)

    def test_policy(self):
        self.assertTrue(self.model.policy([0, 1]).shape == (1, 3), self.model.policy([0, 1]).shape)
        self.assertTrue(self.model.policy([[0, 1]]).shape == (1, 3), self.model.policy([[0, 1]]).shape)
        self.assertTrue(self.model.policy([[0, 1], [2, 3]]).shape == (2, 3), self.model.policy([[0, 1], [2, 3], [4, 5]]).shape)

        self.assertTrue(self.model.policy([0, 1], 1).shape == (1,), self.model.policy([0, 1], 1).shape)
        self.assertTrue(self.model.policy([[0, 1]], [1]).shape == (1,), self.model.policy([[0, 1]], [1]).shape)
        self.assertTrue(self.model.policy([[0, 1], [2, 3]], [1, 0]).shape == (2,), self.model.policy([[0, 1], [2, 3]], [1, 0]).shape)

        self.assertTrue(((self.model.policy([[0, 1], [0, 1], [0, 1]], [0, 1, 2]) - self.model.policy([0, 1])[0])<1e-10).all(),
                        '{} != {}'.format(self.model.policy([[0, 1], [0, 1], [0, 1]], [0, 1, 2]),
                                          self.model.policy([0, 1])[0]))

    def test_advantages(self):
        def adv(g, r, v):
            return self.model.sess.run(self.model.advantages,
                                       feed_dict={self.model.gammas: g,
                                                  self.model.rewards: r,
                                                  self.model.vs: v
                                       })
        temp = adv([1, .9, .8, .7, .6, .5, .4, .3, .2, .1],
                   [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                   [.5, .5, .5, .5, .5, .5, .5, .5, .5, .5])
        self.assertTrue(temp.shape == (10,), temp.shape)
        self.assertTrue(((temp - np.array([.5, .4, .3, .2, .1, 0, -.1, -.2, -.3, -.4])) < 1e-10).all())


class TestValueModel(unittest.TestCase):

    def setUp(self):
        self.model = ValueModel(input_dims=2, drop_out=.0)

    def test_value(self):
        self.assertTrue(self.model.value([0, 1]).shape == (1,), self.model.value([0, 1]).shape)
        self.assertTrue(self.model.value([[0, 1]]).shape == (1,), self.model.value([[0, 1]]).shape)
        self.assertTrue(self.model.value([[0, 1], [2, 3], [4, 5]]).shape == (3,), self.model.value([[0, 1], [2, 3], [4, 5]]).shape)


class TestPolicyGradientAgent(unittest.TestCase):

    def setUp(self):
        self.agent = PolicyGradientAgent(state_dims=2, actions_num=3)

    def test_policy(self):
        self.assertTrue(len(self.agent.policy(np.array([0, 1]))) == 3, len(self.agent.policy(np.array([0, 1]))) == 3)

    def test_act(self):
        self.assertTrue(not hasattr(self.agent.act(np.array([0, 1])), '__len__'), not hasattr(self.agent.act(np.array([0, 1])), '__len__'))



if __name__ == '__main__':
    unittest.main()
