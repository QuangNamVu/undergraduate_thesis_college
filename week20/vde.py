###
'''
Replication of M1 from http://arxiv.org/abs/1406.5298
Title: Semi-Supervised Learning with Deep Generative Models
Authors: Diederik P. Kingma, Danilo J. Rezende, Shakir Mohamed, Max Welling
Original Implementation (Theano): https://github.com/dpkingma/nips14-ssl
---
Code By: VQ Nam
'''
###

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import numpy as np
import tensorflow as tf

import utils
from decoder import Decoder
from encoder import Encoder
from predict import Predict_trend


# Variational Dynamical Encoder
class VariationalDynamicalEncoder(object):

    def __init__(self, params,
                 p_x='bernoulli',
                 q_z='gaussian_marg',
                 p_z='gaussian_marg'):

        self.params = params

        N, M, T, D, n_z = self.params.n_batch, self.params.batch_size, self.params.n_time_steps, self.params.d_inputs, self.params.n_z
        Tau_pred, C = self.params.pred_seq_len, self.params.n_classes
        l2_loss = params.l2_loss

        self.distributions = {'p_x': p_x, 'q_z': q_z, 'p_z': p_z}

        ''' Create Graph '''

        self.G = tf.Graph()

        with self.G.as_default():
            self.x = tf.placeholder(tf.float32, [None, T, D])  # train M, T, D

            self.x_con = tf.placeholder(tf.float32, [None, T, D])

            self.y_one_hot = tf.placeholder(tf.float32, [None, Tau_pred, C])

            self.encoder = Encoder(params=self.params)

            self.decoder = Decoder(params=self.params)

            self.predict = Predict_trend(params=self.params)

            self._objective()
            self.session = tf.Session(graph=self.G)
            self.saver = tf.train.Saver()

    def _gen_sample(self, mu, log_sigma_sq):

        epsilon = tf.random_normal((tf.shape(mu)), 0, 1)

        sample = tf.add(mu, tf.multiply(tf.exp(0.5 * log_sigma_sq), epsilon))

        return sample

    def _generate_zx(self, x, reuse=False):

        with tf.variable_scope('encoder', reuse=reuse):
            z_mu, z_lsgms = self.encoder.output(x, reuse=reuse)

            z_sample = self._gen_sample(z_mu, z_lsgms)

            return z_sample, z_mu, z_lsgms

    def _generate_xz(self, z, reuse=False):

        with tf.variable_scope('decoder', reuse=reuse):
            x_hat = self.decoder.output(z, reuse=reuse)

        return x_hat

    def _generate_yz(self, z, reuse=False):
        with tf.variable_scope('decoder', reuse=reuse):
            y_hat = self.predict.output(z, reuse=reuse)

        return y_hat

    def _objective(self):

        ############
        ''' Cost '''
        ############

        self.z_sample, self.z_mu, self.z_lsgms = self._generate_zx(self.x)

        self.x_hat = self._generate_xz(self.z_sample)

        self.z_tau, _, _ = self._generate_zx(self.x_hat, reuse=True)

        # if self.distributions['p_z'] == 'gaussian_marg':
        #     prior_z = tf.reduce_sum(utils.tf_gaussian_marg(self.z_mu, self.z_lsgms), 1)
        #
        # if self.distributions['q_z'] == 'gaussian_marg':
        #     post_z = tf.reduce_sum(utils.tf_gaussian_ent(self.z_lsgms), 1)
        #
        # if self.distributions['p_x'] == 'bernoulli':
        #     self.log_lik = - tf.reduce_sum(utils.tf_binary_xentropy(self.x, self.x_hat), 1)

        l2 = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()])

        latent_cost = -0.5 * tf.reduce_sum(1 + self.z_lsgms - tf.square(self.z_mu) - tf.exp(self.z_lsgms), axis=1)
        latent_loss = tf.reduce_mean(latent_cost)

        z_mean, z_var = tf.nn.moments(self.z_sample, axes=[0], keep_dims=True)
        z_tau_mean, z_tau_var = tf.nn.moments(self.z_tau, axes=[0], keep_dims=True)

        num = tf.reduce_mean(tf.multiply((self.z_sample - z_mean), (self.z_tau - z_tau_mean)), axis=[0, 1])
        den = tf.reduce_mean(tf.multiply(z_var, tf.transpose(z_tau_var)))

        self.y_pred = self._generate_yz(self.z_sample)  # _, T, C

        # # TODO format code into metric
        # eval_metric_ops = {
        #     "accuracy": tf.metrics.accuracy(labels=self.y_one_hot, predictions=self.predictions["classes"])
        # }
        #
        # self.predictions = {
        #     "classes": tf.argmax(input=self.y_pred, axis=2),
        #     "class_target": tf.argmax(input=self.y_one_hot, axis=2),
        #     "probabilities": tf.nn.softmax(self.y_pred, name="softmax")
        # }

        _, self.accuracy = tf.metrics.accuracy(labels=tf.argmax(self.y_one_hot, 2),
                                               predictions=tf.argmax(self.y_pred, 2))

        # classify
        self.predict_loss = tf.losses.softmax_cross_entropy(onehot_labels=self.y_one_hot, logits=self.y_pred)

        self.corr_loss = - num / (den + 1e-6)

        self.mse_loss = tf.losses.mean_squared_error(labels=self.x_con, predictions=self.x_hat)

        # self.cost = tf.reduce_mean(post_z - prior_z) + self.corr_loss + self.mse_loss + self.l2_loss * l2

        self.cost = self.mse_loss + self.predict_loss + self.corr_loss

        ##################
        ''' Evaluation '''
        ##################

        self.z_sample_eval, _, _ = self._generate_zx(self.x, reuse=True)
        self.x_hat_eval = self._generate_xz(self.z_sample_eval, reuse=True)

        # self.eval_log_lik = - tf.reduce_mean(tf.reduce_sum(utils.tf_binary_xentropy(self.x, self.x_hat_eval), 1))

    def train(self, x, x_con, x_valid, x_con_valid, params,
              y_one_hot_train=None,
              y_one_hot_valid=None,
              save_path=None,
              load_path=None
              ):
        stop_iter = self.params.stop_iter
        N, M = params.n_batch, params.batch_size
        lr, seed, epochs, print_every = params.learning_rate, params.seed, params.epochs, params.print_every

        ''' Session and Summary '''
        self.save_path = save_path
        if save_path is None:
            self.save_ckpt_path = 'checkpoints/model_VAE_{}-{}_{}.cpkt'.format(lr, M, time.time())
        else:
            self.save_ckpt_path = save_path + 'model_VAE_{}-{}_{}.cpkt'.format(lr, M, time.time())

        np.random.seed(seed)
        tf.set_random_seed(seed)

        with self.G.as_default():

            self.optimizer_origin = tf.train.AdamOptimizer(learning_rate=lr)
            self.optimizer = tf.contrib.estimator.clip_gradients_by_norm(self.optimizer_origin, clip_norm=1.0)
            self.train_op = self.optimizer.minimize(self.cost)

            self.init_g = tf.global_variables_initializer()
            self.init_l = tf.local_variables_initializer()
            self._test_vars = None

        with self.session as sess:
            sess.run(self.init_g)
            sess.run(self.init_l)

            if load_path == 'default':
                full_path = tf.train.latest_checkpoint(self.save_path)
                print('restore model from {}'.format(full_path))
                self.saver.restore(sess, full_path)

            elif load_path is not None:
                full_path = tf.train.latest_checkpoint(load_path)
                print('restore model from {}'.format(full_path))
                self.saver.restore(sess, full_path)

            best_eval_log_lik = - np.inf
            best_mse = np.inf

            stop_counter = 0

            for epoch in range(epochs):

                # TODO create shuffle Data
                # X_train_shuffle, X_tau_train_shuffle, y_classify_train_shuffle = shuffle_data(X_train, X_tau_train, y_classify_train)

                ''' Training '''
                training_cost, accuracy, mse = 0, 0, 0
                for x_batch, x_con_batch, y_one_hot_batch in utils.feed_numpy(x, x_con, batch_size=M,
                                                                              y=y_one_hot_train):
                    training_result = sess.run([self.train_op, self.cost, self.accuracy, self.mse_loss],
                                               feed_dict={self.x: x_batch, self.x_con: x_con_batch,
                                                          self.y_one_hot: y_one_hot_batch})

                    training_cost += training_result[1]
                    accuracy += training_result[2]
                    mse += training_result[3]

                training_cost, accuracy, mse = training_cost / N, accuracy / N, mse / N
                ''' Evaluation '''

                stop_counter += 1

                if epoch % print_every == 0:

                    # mse = sess.run(self.mse_loss, feed_dict={self.x: x, self.x_con: x_con})

                    val_result = sess.run(self.accuracy,
                                          feed_dict={self.x: x_valid, self.x_con: x_con_valid,
                                                     self.y_one_hot: y_one_hot_valid})

                    if mse < best_mse:
                        best_eval_log_lik = mse
                        self.saver.save(sess, self.save_ckpt_path)
                        stop_counter = 0

                    utils.print_metrics(epoch + 1,
                                        ['Training cost', training_cost],
                                        ['Accuracy', accuracy],
                                        ['MSE', mse],
                                        ['Validation accuracy', val_result]
                                        )

                if stop_counter >= stop_iter:
                    print('Stopping No change in validation log-likelihood for {} iterations'.format(stop_iter))
                    print('Best validation log-likelihood: {}'.format(best_eval_log_lik))
                    print('Model saved in {}'.format(self.save_path))
                    break

    def encode(self, x, sample=False):

        if sample:
            return self.session.run([self.z_sample, self.z_mu, self.z_lsgms], feed_dict={self.x: x})
        else:
            return self.session.run([self.z_mu, self.z_lsgms], feed_dict={self.x: x})

    def decode(self, z):

        return self.session.run([self.x_hat], feed_dict={self.z_sample: z})
