import tensorflow as tf

from info_params import params
from read_data import *

if __name__ == '__main__':
    info_params = params()
    info_params.idx_split = 1000

    X_train_seq, X_test_seq, y_train_seq, y_test_seq = load_data(info_params)

    N_seq, D = np.shape(X_train_seq)

    # N_seq, D =
    lstm_cell = tf.contrib.rnn.OutputProjectionWrapper(
        tf.nn.rnn_cell.LSTMCell(name='basic_lstm_cell', num_units=10, activation=tf.nn.relu),
        output_size=1)

    X = tf.placeholder(tf.float32, [None, 8])
    X_3d = tf.expand_dims(X, axis=2)

    y = tf.placeholder(tf.float32, [None, 8])

    lstm_outputs, lstm_states = tf.nn.dynamic_rnn(lstm_cell, X_3d, dtype=tf.float32)

    y_hat = tf.reshape(lstm_outputs, shape=[-1, 8])

    ## MSE in train
    loss = tf.reduce_mean(tf.math.square(y_hat - y))

    original_optimizer = tf.train.AdamOptimizer(learning_rate=.001)

    optimizer = tf.contrib.estimator.clip_gradients_by_norm(original_optimizer, clip_norm=1.0)

    train_err = optimizer.minimize(loss)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init)
        saver.restore(sess, "/tmp/model.ckpt")

        for iterations in range(100000):
            sess.run(train_err, feed_dict={X: X_train_seq, y: y_train_seq})

            if iterations % 1000 == 0:
                mse = loss.eval(feed_dict={X: X_train_seq, y: y_train_seq})

                print('iterations {}  MSE val {}'.format(iterations, mse))

                saver.save(sess, '/tmp/model.ckpt')

