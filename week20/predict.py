import tensorflow as tf


# predict trend, classify problem
class Predict_trend(object):

    def __init__(self, params):
        self.params = params
        self.T = params.n_time_steps  # 128
        self.C = params.n_classes  # 2
        self.tau = params.pred_seq_len  # 64

    def output(self, inputs, reuse=False):
        with tf.variable_scope('predict', reuse=reuse):
            # # reshape first

            # _, n_z -> _, C* T : FC
            fc_l1 = tf.contrib.layers.fully_connected(inputs=inputs,
                                                      num_outputs=self.C * self.T)  # ,activation_fn=tf.nn.relu)

            # _, C*T -> _, C, T : reshape
            rs_l2 = tf.reshape(fc_l1, shape=[-1, self.C, self.T])

            # _, C, T -> _, C, tau
            lstm_cell = tf.contrib.rnn.OutputProjectionWrapper(
                tf.nn.rnn_cell.LSTMCell(name='lstm_cell', num_units=self.params.lstm_units),
                output_size=self.tau)

            y_predict, _ = tf.nn.dynamic_rnn(lstm_cell, rs_l2, dtype=tf.float32)

            y_predict = tf.reshape(y_predict, shape=[-1, self.tau, self.C])

        return y_predict
