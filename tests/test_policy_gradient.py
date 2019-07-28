import unittest
from rl_lib.agents.policy_gradient import *


class TestPolicyModel(unittest.TestCase):

    def setUp(self):
        self.model = PolicyModel(input_dims=2, output_dims=3, drop_out=.0)

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
        temp = adv([1, .5], [1, .5], [1, .5])
        self.assertTrue(temp.shape == (2,), temp.shape)
        self.assertTrue(((temp - np.array([0, -.25]))<1e-10).all())


if __name__ == '__main__':
    unittest.main()
