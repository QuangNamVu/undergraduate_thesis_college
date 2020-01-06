import tensorflow as tf


# x constructed generate from pdf: p(x_con | z)

class Decoder(object):
    def __init__(self, params):
        self.params = params

    def output(self, inputs, reuse=False):
        with tf.variable_scope('encoder', reuse=reuse):
            N, T, D = self.params.n_batch, self.params.n_time_steps, self.params.d_inputs
            # M, n_z => M, T, D

            # contructed

            flat_l1 = tf.contrib.layers.fully_connected(inputs=inputs, num_outputs=T * D,
                                                        activation_fn=tf.nn.relu)

            rs_l2 = tf.reshape(flat_l1, shape=[-1, T, D])

            # decode with shift time Tau using LSTM
            # z_reshape = tf.expand_dims(inputs, 1)  # M, 1, 50
            #
            # lstm_cell = tf.contrib.rnn.OutputProjectionWrapper(
            #     tf.nn.rnn_cell.LSTMCell(name='basic_lstm_cell', num_units=12, activation=tf.nn.relu),
            #     output_size=1024)  # N, 1, 1024
            #
            # x_con, _ = tf.nn.dynamic_rnn(lstm_cell, z_reshape, dtype=tf.float32)

            # x_con_flat = tf.reshape(x_con, shape=[-1, 1024])

            # x_hat_flat = tf.contrib.layers.fully_connected(inputs=inputs, num_outputs=1024, activation_fn=tf.nn.relu)

            return rs_l2
