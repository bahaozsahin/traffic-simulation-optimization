import os

import numpy as np
import tensorflow as tf
import keras
from keras.initializers import he_uniform

from core.neuralnets.neural_net import NeuralNet
from core.neuralnets.variable_state import VariableState
from utils.pickle_helper import save_data, load_data
from utils.helpers import check_and_make_dir


class DDPGActorNet:
    def __init__(self, input_d, hidden_d, hidden_act, output_d, output_act, lr, lre, name, batch_size, sess):
        tf.compat.v1.disable_eager_execution()
        with tf.compat.v1.variable_scope(name, reuse=tf.compat.v1.AUTO_REUSE):
            self.input = tf.compat.v1.placeholder(tf.float32,
                                        shape=[None, input_d],
                                        name='inputs')
                                                                                                     
            self.action_gradient = tf.compat.v1.placeholder(tf.float32,
                                          shape=[None, output_d],
                                          name='gradients')
                                                                                                     
            dense1 = tf.compat.v1.layers.dense(self.input, units=hidden_d[0],
                                     kernel_initializer=he_uniform())

            batch1 = tf.compat.v1.layers.batch_normalization(dense1)
            layer1_activation = tf.nn.elu(batch1)
            dense2 = tf.compat.v1.layers.dense(layer1_activation, units=hidden_d[1],
                                     kernel_initializer=he_uniform())

            batch2 = tf.compat.v1.layers.batch_normalization(dense2)
            layer2_activation = tf.nn.elu(batch2)
            mu = tf.compat.v1.layers.dense(layer2_activation, units=output_d,
                            activation='tanh',
                            kernel_initializer=he_uniform())

            self.mu = mu
                                                                                                     
            self.params = tf.compat.v1.trainable_variables(scope=name)

            self.unnormalized_actor_gradients = tf.gradients(self.mu, self.params, -self.action_gradient, unconnected_gradients='zero')
            
            self.actor_gradients = list(map(lambda x: tf.math.divide(x, batch_size), self.unnormalized_actor_gradients))
                                                                                                    
            self.optimize = tf.compat.v1.train.AdamOptimizer(learning_rate = lr, epsilon=lre).apply_gradients(zip(self.actor_gradients, self.params))

            self.variable_state = VariableState(sess, self.params)


class DDPGActor(NeuralNet):
    def __init__(self, input_d, hidden_d, hidden_act, output_d, output_act, lr, lre, tau, learner=False, name='', batch_size=32, sess=None):
        self.lr = lr
        self.lre = lre
        self.sess = sess
        self.batch_size = batch_size
        self.name = name
        super().__init__(input_d, hidden_d, hidden_act, output_d, output_act, learner=learner)
        self.tau = tau
        self.new_w = []

        if learner:
            self.update_actor = [self.models['target'].params[i].assign(
                                 tf.multiply(self.models['online'].params[i], self.tau)
                                 + tf.multiply(self.models['target'].params[i], 1. - self.tau))
                                 for i in range(len(self.models['target'].params))]

    def create_model(self, input_d, hidden_d, hidden_act, output_d, output_act):
        return DDPGActorNet(input_d, hidden_d, hidden_act, output_d, output_act, self.lr, self.lre, self.name, self.batch_size, self.sess)

    def forward(self, x, nettype):
        return self.sess.run(self.models[nettype].mu, feed_dict={self.models[nettype].input: x})

    def backward(self, states, grads):
        self.sess.run(self.models['online'].optimize,
                      feed_dict={self.models['online'].input: states,
                                 self.models['online'].action_gradient: grads})

    def transfer_weights(self):
        self.sess.run(self.update_actor)

    def get_weights(self, nettype):
        return self.models[nettype].variable_state.export_variables()

    def set_weights(self, weights, nettype):
        self.models[nettype].variable_state.import_variables(weights)

    def save_weights(self, nettype, path, fname):
        check_and_make_dir(path)
        weights = self.get_weights('online')
        save_data(path+fname+'.p', weights)

    def load_weights(self, path):
        path += '.p'
        if os.path.exists(path):
            weights = load_data(path)
            self.set_weights(weights, 'online')
        else:
            assert 0, 'Failed to load weights, supplied weight file path '+str(path)+' does not exist.'
