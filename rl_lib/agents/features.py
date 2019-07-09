import numpy as np

from rl import *


class FeatureAgent(Agent):

    def __init__(self, features):
        self.features = np.array(features)
        self.state_dimensions = self.features.shape[1]
        self.action_dimensions = self.features.shape[0]
        self.bins = np.arange(self.action_dimensions)

    def act(self, state):
        return np.digitize(np.matmul(self.features, state), bins=self.bins).astype(np.int)[0]

    def get_features(self):
        return self.features


class RandomFeatureAgent(FeatureAgent):

    def __init__(self, state_dims, actions_dims):
        super().__init__(np.random.random((actions_dims, state_dims)))


