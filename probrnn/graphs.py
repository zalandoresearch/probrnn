import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.contrib.rnn import LSTMStateTuple
from tensorflow.contrib.keras import losses
import numpy as np


class BaseGraph:
    """
    Base graph initializing summary statistics for saving.
    """
    def initialize(self):
        """
        Add summary statistics dictionary to self.summary        
        """
        self.session.run(tf.global_variables_initializer())
        self.indices = {}
        for x in tf.global_variables():
            try:
                self.indices[x.name] = np.random.choice(len(self.session.run(x).flatten()), size=5)
            except IndexError:
                self.indices[x.name] = [0]

        self.summary = dict([(x.name, {"mean": [], "max": [], "min": [], "random": []})
                             for x in tf.global_variables()])

    def load(self, fn):
        """
        :param fn: filename to saved model (without '.ckpt')
        """
        saver = tf.train.Saver()
        saver.restore(self.session, fn + ".ckpt")

    def save(self, fn):
        """
        :param fn: filename to saved model (without '.ckpt')
        """
        saver = tf.train.Saver()
        saver.save(self.session, fn + ".ckpt")

    def get_stats(self):
        """
        Get stats and add to self.summary
        :return: 
        """
        for x in tf.global_variables():
            self.summary[x.name]["mean"].append(self.session.run(tf.reduce_mean(x)))
            try:
                self.summary[x.name]["min"].append(self.session.run(tf.reduce_min(x)))
                self.summary[x.name]["max"].append(self.session.run(tf.reduce_max(x)))
                self.summary[x.name]["random"].append(list(self.session.run(x).flatten()[self.indices[x.name]]))

            except ValueError:
                pass


class NADE(BaseGraph):
    """Neural auto-regressive density estimation."""

    def __init__(self, params):
        """
        
        :param params: dictionary with fields:
            "N_HIDDEN": number of hidden states
            "N_BINS": number of bins on input/ output
            "LEARNING_RATE": learning rate in optimizer
        """
        self.params = params

        tf.reset_default_graph()

        self.session = tf.Session()

        self.inputs = tf.placeholder(tf.float32, (None, None, params['N_BINS']))

        self.cell = tf.contrib.rnn.LSTMCell(params['N_HIDDEN'], state_is_tuple=True)
        self.batch_size = tf.shape(self.inputs)[1]

        self.h_init = tf.Variable(tf.zeros([1, params['N_HIDDEN']]), trainable=True)
        self.h_init_til = tf.tile(self.h_init, [self.batch_size, 1])

        self.c_init = tf.Variable(tf.zeros([1, params['N_HIDDEN']]), trainable=True)
        self.c_init_til = tf.tile(self.c_init, [self.batch_size, 1])

        self.initial_state = LSTMStateTuple(self.c_init_til, self.h_init_til)

        self.rnn_outputs, self.rnn_states = \
            tf.nn.dynamic_rnn(self.cell,
                              self.inputs,
                              initial_state=self.initial_state,
                              time_major=True)

        with tf.variable_scope("output"):

            self.intermediate_projection = \
                lambda x: layers.fully_connected(x, num_outputs=params['N_HIDDEN'])

            self.final_projection = \
                lambda x: layers.linear(x, num_outputs=params['N_BINS'])

            self.intermediate_features = tf.map_fn(self.intermediate_projection, self.rnn_outputs)
            self.final_features = tf.map_fn(self.final_projection, self.intermediate_features)
            self.predicted_outputs = layers.softmax(self.final_features)

        with tf.variable_scope("train"):
            self.outputs = \
                tf.placeholder(tf.float32, (None, None, params['N_BINS']))

            self.mask = tf.placeholder(tf.float32, (None, None, params['N_BINS']))

            self.all_errors = losses.categorical_crossentropy(self.outputs * self.mask, self.predicted_outputs)

            self.error = tf.reduce_mean(self.all_errors)

            self.train_fn = \
                tf.train.AdamOptimizer(learning_rate=params['LEARNING_RATE']) \
                    .minimize(self.error)

    def train_step(self, batch):
        """
        Take a step with optimizer
        
        :param batch: batch of data
        :return: error on batch
        """
        data = batch[0]
        mask = batch[1]
        return self.session.run(
            [self.error, self.train_fn],
            {
                self.inputs: data[:-1],
                self.outputs: data[1:],
                self.mask: mask[1:],
            }
        )[0]

    def test_error(self, batch):
        """
        Measure error on test batch

        :param batch: batch of data
        :return: error on batch
        """
        data = batch[0]
        mask = batch[1]
        return self.session.run(
            [self.error],
            {
                self.inputs: data[:-1],
                self.outputs: data[1:],
                self.mask: mask[1:],
            }
        )[0]

    def predict(self, batch, states, batchsize=1):
        """
        
        :param batch: batch of data
        :param states: prior states
        :param batchsize: batchsize if getting initial state
        :return: 
            predicted_outputs: softmax predictions
            states: states of RNN
        """

        if batch is None:

            init_states = self.session.run(self.initial_state, {self.batch_size: batchsize})
            predicted_outputs = self.session.run(self.predicted_outputs,
                                                 {self.rnn_outputs: init_states.h[None, :, :]})
            return predicted_outputs, init_states

        else:
            if states is None:
                return self.session.run([self.predicted_outputs, self.rnn_states], {self.inputs: batch})
            else:
                c = states[0]
                h = states[1]
                return self.session.run([self.predicted_outputs, self.rnn_states],
                                        {self.inputs: batch,
                                         self.initial_state: LSTMStateTuple(c, h)})



class TimeSeriesPrediction(BaseGraph):
    """Neural auto-regressive density estimation."""

    def __init__(self, params):
        """

        :param params: dictionary with fields:
            "N_HIDDEN": number of hidden states
            "N_BINS": number of bins on input/ output
            "WINDOW_LENGTH": number of time-steps in training window
        """
        tf.reset_default_graph()

        self.session = tf.Session()

        self.inputs = tf.placeholder(tf.float32, (None, None, params['N_BINS']))

        self.cell = tf.contrib.rnn.LSTMCell(params['N_HIDDEN'], state_is_tuple=True)
        self.batch_size = tf.shape(self.inputs)[1]

        self.h_init = tf.Variable(tf.zeros([1, params['N_HIDDEN']]), trainable=True)
        self.h_init_til = tf.tile(self.h_init, [self.batch_size, 1])

        self.c_init = tf.Variable(tf.zeros([1, params['N_HIDDEN']]), trainable=True)
        self.c_init_til = tf.tile(self.c_init, [self.batch_size, 1])

        self.initial_state = LSTMStateTuple(self.c_init_til, self.h_init_til)

        self.rnn_outputs, self.rnn_states = \
            tf.nn.dynamic_rnn(self.cell,
                              self.inputs,
                              initial_state=self.initial_state,
                              time_major=True)

        self.intermediate_projection = layers.fully_connected(self.rnn_outputs[params["WINDOW_LENGTH"] - 2, :, :], num_outputs=params['N_HIDDEN'])
        self.final_features = layers.linear(self.intermediate_projection, num_outputs=params["N_BINS"])

        self.predicted_outputs = layers.softmax(self.final_features)

        with tf.variable_scope("train"):
            # get element-wise error
            self.outputs = \
                tf.placeholder(tf.float32, (None, params['N_BINS']))

            self.all_errors = losses.categorical_crossentropy(self.outputs, self.predicted_outputs)

            # measure error
            self.error = tf.reduce_mean(self.all_errors)

            self.train_fn = \
                tf.train.AdamOptimizer(learning_rate=params['LEARNING_RATE']) \
                    .minimize(self.error)

    def train_step(self, batch):
        """
        Take a step with optimizer

         :param batch: batch of data
         :return: error on batch
         """
        data = batch[0]
        return self.session.run(
            [self.error, self.train_fn],
            {
                self.inputs: data[:-1],
                self.outputs: data[-1],
            }
        )[0]

    def test_error(self, batch):
        """
        Measure error on test batch

        :param batch: batch of data
        :return: error on batch
        """
        data = batch[0]
        return self.session.run(
            [self.error],
            {
                self.inputs: data[:-1],
                self.outputs: data[-1],
            }
        )[0]

    def predict(self, batch, states, batchsize=1):
        """

        :param batch: batch of data
        :param states: prior states
        :param batchsize: batchsize if getting initial state
        :return: 
            predicted_outputs: softmax predictions
            states: states of RNN
        """

        if states is None:
            return self.session.run([self.predicted_outputs, self.rnn_states], {self.inputs: batch})
        else:
            c = states[0]
            h = states[1]
            return self.session.run([self.predicted_outputs, self.rnn_states],
                                    {self.inputs: batch,
                                     self.initial_state: LSTMStateTuple(c, h)})
