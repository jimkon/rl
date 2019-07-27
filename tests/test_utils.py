import unittest

from rl_lib.utils.utils import *


class TestMapper(unittest.TestCase):

    def setUp(self):
        self.mapper = Mapper()

    def test_map(self):
        self.assertTrue((self.mapper.map([-2, -3, 5]) == np.array([-2, -3, 5])).all(), self.mapper.map([-2, -3, 5]))
        self.assertTrue((self.mapper.map([6, 1, 11]) == np.array([6, 1, 11])).all(), self.mapper.map([6, 1, 11]))
        self.assertTrue((self.mapper.map([2, -1, 8]) == np.array([2, -1, 8])).all(), self.mapper.map([2, -1, 8]))
        self.assertTrue((self.mapper.map([-1, 0, 5.6]) == np.array([-1, 0, 5.6])).all(), self.mapper.map([-1, 0, 5.6]))


class TestStandardMapper(unittest.TestCase):

    def setUp(self):
        self.mapper = StandardMapper([-2, -3, 5], [6, 1, 11])

    def test_map(self):
        self.assertTrue((self.mapper.map([-2, -3, 5]) == np.zeros(3)).all(), self.mapper.map([-2, -3, 5]))
        self.assertTrue((self.mapper.map([6, 1, 11]) == np.ones(3)).all(), self.mapper.map([6, 1, 11]))
        self.assertTrue((self.mapper.map([2, -1, 8]) == np.ones(3)*.5).all(), self.mapper.map([2, -1, 8]))
        self.assertTrue(((self.mapper.map([-1, 0, 5.6]) - [.125, .75, .1])<1e-10).all(), self.mapper.map([-1, 0, 5.6]))


class TestUnitMapper(unittest.TestCase):

    def setUp(self):
        self.mapper = UnitMapper([-2, -3, 5], [6, 1, 11])

    def test_map(self):
        self.assertTrue((self.mapper.map([-2, -3, 5]) == -np.ones(3)).all(), self.mapper.map([-2, -3, 5]))
        self.assertTrue((self.mapper.map([6, 1, 11]) == np.ones(3)).all(), self.mapper.map([6, 1, 11]))
        self.assertTrue((self.mapper.map([2, -1, 8]) == np.zeros(3)).all(), self.mapper.map([2, -1, 8]))
        self.assertTrue(((self.mapper.map([-1, 0, 5.6]) - [-.75, .5, -.8])<1e-10).all(), self.mapper.map([-1, 0, 5.6]))


if __name__ == '__main__':
    unittest.main(verbosity=2)