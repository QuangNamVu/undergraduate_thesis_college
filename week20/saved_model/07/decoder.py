import tensorflow as tf


# x constructed generate from pdf: p(x_con | z)

class Decoder(object):
    def __init__(self, params):
        self.D = params.d_inputs
        self.T = params.n_time_steps
        self.params = params

    def output(self, inputs, reuse=False):
        with tf.variable_scope('encoder', reuse=reuse):
            M, n_z = inputs.shape
            # # normal contructed

            # flat_l1 = tf.contrib.layers.fully_connected(inputs=inputs, num_outputs=T * D,
            #                                             activation_fn=tf.nn.relu)
            #
            # rs_l1 = tf.reshape(flat_l1, shape=[-1, T, D])

            # decode with shift time Tau using LSTM

            # _, n_z -> _, C* T : FC
            fc_l1 = tf.contrib.layers.fully_connected(inputs=inputs,
                                                      num_outputs=self.T * self.D, activation_fn=tf.nn.relu)

            rs_l1 = tf.reshape(fc_l1, shape=[-1, self.D, self.T])

            lstm_cell = tf.contrib.rnn.OutputProjectionWrapper(
                tf.nn.rnn_cell.LSTMCell(name='basic_lstm_cell', num_units=12, activation=tf.nn.relu),
                output_size=self.T)  # N, D, T

            lstm_l2, _ = tf.nn.dynamic_rnn(lstm_cell, rs_l1, dtype=tf.float32)

            rs_l2 = tf.reshape(fc_l1, shape=[-1, self.T, self.D])

            return rs_l2
