import unittest
from rl_lib.utils.nets import *


class TestDNN(unittest.TestCase):

    def setUp(self):
        self.nn = FullyConnectedDNN(input_dims=2, output_dims=4)

    def test_predict_shape(self):
        self.assertTrue(self.nn.predict([0, 1]).shape == (1, 4), self.nn.predict([0, 1]).shape)
        self.assertTrue(self.nn.predict([[0, 1]]).shape == (1, 4), self.nn.predict([[0, 1]]).shape)
        self.assertTrue(self.nn.predict([[0, 1], [2, 3], [4, 5]]).shape == (3, 4), self.nn.predict([[0, 1], [2, 3], [4, 5]]).shape)

if __name__ == '__main__':
    unittest.main()
