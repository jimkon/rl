import numpy as np
import tensorflow as tf

import rl_lib as rl

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

        self.policy_net = rl.nets.FullyConnectedDNN(**nn_args)

        self.nets = [rl.nets.RBFNet(**rbf_args) for _ in range(self.actions_num)]
        for net in self.nets:
            net.create_net(state_dims, 1)

