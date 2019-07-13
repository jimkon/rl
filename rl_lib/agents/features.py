import numpy as np

import rl_lib as rl


class FeatureAgent(rl.Agent):

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

    def __init__(self, state_dims, actions_dims, distribution=np.random.normal, **distr_args):
        features = distribution(size=(actions_dims, state_dims), **distr_args)
        super().__init__(features)

