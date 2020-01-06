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


# Variational Dynamical Encoder
class VariationalDynamicalEncoder(object):

    def __init__(self,
                 dim_x, dim_z, dim_recon,
                 p_x='bernoulli',
                 q_z='gaussian_marg',
                 p_z='gaussian_marg',
                 l2_loss=1e-6):

        self.dim_x, self.dim_z, self.dim_recon = dim_x, dim_z, dim_recon

        self.distributions = {'p_x': p_x, 'q_z': q_z, 'p_z': p_z}

        self.l2_loss = l2_loss

        ''' Create Graph '''

        self.G = tf.Graph()

        with self.G.as_default():
            self.x = tf.placeholder(tf.float32, [None, self.dim_x])

            self.y = tf.placeholder(tf.float32, [None, self.dim_recon])

            self.encoder = Encoder(dim_output=2 * self.dim_z)

            self.decoder = Decoder(dim_output=self.dim_x)

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
            x_hat = self.decoder.output(z)

        return x_hat

    def _objective(self):

        ############
        ''' Cost '''
        ############

        self.z_sample, self.z_mu, self.z_lsgms = self._generate_zx(self.x)

        self.x_hat = self._generate_xz(self.z_sample)

        self.z_tau, _, _ = self._generate_zx(self.x_hat, reuse=True)

        if self.distributions['p_z'] == 'gaussian_marg':
            prior_z = tf.reduce_sum(utils.tf_gaussian_marg(self.z_mu, self.z_lsgms), 1)

        if self.distributions['q_z'] == 'gaussian_marg':
            post_z = tf.reduce_sum(utils.tf_gaussian_ent(self.z_lsgms), 1)

        if self.distributions['p_x'] == 'bernoulli':
            self.log_lik = - tf.reduce_sum(utils.tf_binary_xentropy(self.x, self.x_hat), 1)

        l2 = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()])

        latent_cost = -0.5 * tf.reduce_sum(1 + self.z_lsgms - tf.square(self.z_mu) - tf.exp(self.z_lsgms), axis=1)
        latent_loss = tf.reduce_mean(latent_cost)

        z_mean, z_var = tf.nn.moments(self.z_sample, axes=[0], keep_dims=True)
        z_tau_mean, z_tau_var = tf.nn.moments(self.z_tau, axes=[0], keep_dims=True)

        num = tf.reduce_mean(tf.multiply(tf.transpose(self.z_sample - z_mean), (self.z_tau - z_tau_mean)), axis=[0, 1])
        den = tf.reduce_mean(tf.multiply(z_var, tf.transpose(z_tau_var)))

        self.corr_loss = - num / (den + 1e-6)

        self.mse_loss = tf.losses.mean_squared_error(labels=self.y, predictions=self.x_hat)

        # self.cost = tf.reduce_mean(post_z - prior_z) + self.corr_loss + self.mse_loss + self.l2_loss * l2

        self.cost = tf.reduce_mean(post_z - prior_z) + self.mse_loss + self.corr_loss + self.l2_loss * l2

        ##################
        ''' Evaluation '''
        ##################

        self.z_sample_eval, _, _ = self._generate_zx(self.x, reuse=True)
        self.x_hat_eval = self._generate_xz(self.z_sample_eval, reuse=True)

        self.eval_log_lik = - tf.reduce_mean(tf.reduce_sum(utils.tf_binary_xentropy(self.x, self.x_hat_eval), 1))

    def train(self, x, y, x_valid, y_valid,
              epochs, num_batches,
              print_every=1,
              learning_rate=3e-4,
              beta1=0.9,
              beta2=0.999,
              seed=31415,
              stop_iter=100,
              save_path=None,
              load_path=None):

        self.num_examples = x.shape[0]
        self.num_batches = num_batches

        assert self.num_examples % self.num_batches == 0, '#Examples % #Batches != 0'

        self.batch_size = self.num_examples // self.num_batches

        ''' Session and Summary '''
        self.save_path = save_path
        if save_path is None:
            self.save_ckpt_path = 'checkpoints/model_VAE_{}-{}_{}.cpkt'.format(learning_rate, self.batch_size,
                                                                               time.time())
        else:
            self.save_ckpt_path = save_path + 'model_VAE_{}-{}_{}.cpkt'.format(learning_rate, self.batch_size,
                                                                               time.time())

        np.random.seed(seed)
        tf.set_random_seed(seed)

        with self.G.as_default():

            self.optimizer_origin = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1, beta2=beta2)
            self.optimizer = tf.contrib.estimator.clip_gradients_by_norm(self.optimizer_origin, clip_norm=1.0)
            self.train_op = self.optimizer.minimize(self.cost)

            self.init = tf.global_variables_initializer()
            self._test_vars = None


        with self.session as sess:

            sess.run(self.init)

            if load_path == 'default':
                full_path = tf.train.latest_checkpoint(self.save_path)
                print('restore model from {}'.format(full_path))
                self.saver.restore(sess, full_path)

            elif load_path is not None:
                full_path = tf.train.latest_checkpoint(load_path)
                print('restore model from {}'.format(full_path))
                self.saver.restore(sess, full_path)

            training_cost = 0.
            best_eval_log_lik = - np.inf
            best_mse = np.inf

            stop_counter = 0

            for epoch in range(epochs):

                # TODO create shuffle Data

                ''' Training '''

                for x_batch, y_batch in utils.feed_numpy(self.batch_size, x, y):
                    training_result = sess.run([self.train_op, self.cost], feed_dict={self.x: x_batch, self.y: y_batch})

                    training_cost = training_result[1]

                ''' Evaluation '''

                stop_counter += 1

                if epoch % print_every == 0:

                    # test_vars = tf.get_collection(bookkeeper.GraphKeys.TEST_VARIABLES)
                    # if test_vars:
                    #     if test_vars != self._test_vars:
                    #         self._test_vars = list(test_vars)
                    #         self._test_var_init_op = tf.initialize_variables(test_vars)
                    #     self._test_var_init_op.run()

                    mse = sess.run(self.mse_loss, feed_dict={self.x: x, self.y: y})

                    # corr_loss = self.corr_loss.eval(feed_dict={self.x: x, self.y: y})

                    if mse < best_mse:
                        best_eval_log_lik = mse
                        self.saver.save(sess, self.save_ckpt_path)
                        stop_counter = 0

                    utils.print_metrics(epoch + 1,
                                        ['Training', 'cost', training_cost],
                                        # ['AutoCorr', 'train', corr_loss],
                                        ['MSE    ', 'train', mse]
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
