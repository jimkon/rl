import numpy as np
import tensorflow as tf


class RBF_net:
    def __init__(self, samplers=None, constant_samplers=False):
        self.samplers = samplers
        self.samplers_num = -1
        self.constant_samplers = constant_samplers
        self.initialized = False

    def create_net(self):
        tf.reset_default_graph()
        assert self.samplers_num != 1, 'samplers must be a positive integer. took {} instead'.format(self.samplers)

        # variables
        if self.constant_samplers:
            self.centers = tf.constant(self.samplers)
        else:
            self.centers = tf.Variable(self.samplers)
        self.gammas = tf.Variable(np.random.random((self.samplers_num)))
        self.weights = tf.Variable(np.random.random((self.samplers_num, self.output_dimensions)))

        # placeholders
        self.x = tf.placeholder(tf.float64, (self.input_dimensions))
        self.y = tf.placeholder(tf.float64, (self.output_dimensions))

        # internal computations
        dists = tf.sqrt(tf.reduce_sum(tf.square(self.x - self.centers), axis=1))
        normalized = tf.reshape(tf.multiply(dists, self.gammas), shape=(1, self.samplers_num))
        self.output = tf.matmul(normalized, self.weights)

        # training computations
        cost = tf.abs(self.output - self.y)
        self.train = tf.train.GradientDescentOptimizer(10e-3).minimize(cost)

        # init
        self.init_op = tf.global_variables_initializer()

        self.sess = tf.Session()
        self.sess.run(self.init_op)

    def predict(self, X):
        return self.sess.run(self.output, feed_dict={self.x: X})[0]

    def partial_fit(self, X, y):
        if not self.initialized:
            self.input_dimensions = X.shape[0]
            self.output_dimensions = y.shape[0]

            if self.samplers is not None:
                # if samplers parameter is set
                if isinstance(self.samplers, int):
                    # if samplers parameter is integer
                    if self.samplers == -1:
                        self.samplers_num = self.input_dimensions
                    else:
                        self.samplers_num = self.samplers

                    self.samplers = np.random.random((self.samplers_num, self.input_dimensions))
                else:
                    # if samplers parameter is array
                    self.samplers = np.array(self.samplers)
                    self.samplers_num = len(self.samplers)
            else:
                self.samplers_num = self.input_dimensions
                self.samplers = np.random.random((self.samplers_num, self.input_dimensions))

            assert self.output_dimensions == 1, 'Not tested for more than 1 output dimensions'
            assert self.input_dimensions == self.samplers.shape[
                1], 'Samplers dimensions don\'t match with input dimensions'

            print('RBF net init in({}) samplers({}) out({})'.format(self.input_dimensions, self.samplers_num,
                                                                    self.output_dimensions))
            self.create_net()
            self.initialized = True

        self.sess.run(self.train, feed_dict={self.x: X, self.y: y})

    def info(self):
        centers = self.sess.run(self.centers)
        gammas = self.sess.run(self.gammas)
        weights = self.sess.run(self.weights)

        return centers, gammas, weights