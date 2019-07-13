import numpy as np

import rl_lib as rl


class FeatureAgent(rl.Agent):

    def __init__(self, features, actions_num, include_beta=False):
        self.features = np.array(features if not include_beta else features[:-1])
        self.state_dims = self.features.shape[0] if not include_beta else self.features.shape[0]-1
        self.actions_num = actions_num
        self.bins = np.arange(self.actions_num-1)
        self.beta = features[-1] if include_beta else .0

    def act(self, state):
        return np.digitize(np.dot(self.features, state)+self.beta, bins=self.bins).astype(np.int)

    def get_features(self):
        return self.features, self.beta


class RandomFeatureAgent(FeatureAgent):

    def __init__(self, state_dims, actions_num, distribution=np.random.normal, **distr_args):
        features = distribution(size=(state_dims+1), **distr_args)
        super().__init__(features, actions_num, include_beta=True)
