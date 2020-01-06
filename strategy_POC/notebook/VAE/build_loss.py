import tensorflow as tf
from tensorpack import *
from tensorpack.tfutils.summary import *
from .tf_utils.common import *


def build_losses(self, y_one_hot, x_hat):
    with tf.name_scope("encode"):
        if self.hps.is_VAE:
            # latent_loss = 0.5 * tf.reduce_sum(tf.square(self.z_mu) + tf.square(self.std) - 2 * tf.log(self.std) - 1,axis=1)

            log_post_z = -0.5 * \
                tf.reduce_sum((np.log(2. * np.pi) + 1 +
                               tf.log(self.std)), 1)  # N: qz

            log_prior_z = -0.5 * \
                tf.reduce_sum((np.log(2. * np.pi) + self.std), 1)  # N: pz

            # max elbo: => max(logpx + logpz - logqz)
            latent_loss = tf.reduce_mean(
                (log_post_z - log_prior_z))  # - log_p_x)

            self.latent_loss = tf.reduce_mean(latent_loss, name='latent_loss')

            if self.hps.is_IAF:
                self.latent_loss = tf.subtract(
                    self.latent_loss, tf.reduce_mean(self.z_lgsm_iaf), name='iaf_loss')
        else:
            self.latent_loss = tf.constant(0.0)

        add_moving_summary(self.latent_loss)
        if self.hps.is_IAF:
            z_mean, z_std = reduce_mean_std(
                self.z_iaf, axis=[1], keepdims=True)
            z_tau_mean, z_tau_std = reduce_mean_std(
                self.z_tau_iaf, axis=[1], keepdims=True)

            num = (self.z_iaf - z_mean) * (self.z_tau_iaf - z_tau_mean)

        else:
            z_mean, z_std = reduce_mean_std(self.z, axis=[1], keepdims=True)
            z_tau_mean, z_tau_std = reduce_mean_std(
                self.z_tau, axis=[1], keepdims=True)

            num = (self.z - z_mean) * (self.z_tau - z_tau_mean)

        if self.hps.is_VDE:
            den = z_std * z_tau_std

            self.auto_corr_loss = tf.reduce_mean(
                - tf.truediv(num, den), name='auto-corr-loss')
            add_moving_summary(self.auto_corr_loss)

        else:
            self.auto_corr_loss = tf.constant(0.0)

        if self.hps.check_error_z:
            # total_x_con = tf.reduce_mean(self.x_con, name='check_error')
            # a, b = tf.nn.moments(self.z, axes=[1], keep_dims=False)
            # tf.summary.scalar(name='check_z_mu', tensor=tf.reduce_mean(a))
            # tf.summary.scalar(name='check_z_std', tensor=tf.reduce_mean(b))
            tf.summary.scalar(name='check_z_norm',
                              tensor=tf.reduce_mean(tf.abs(self.z)))

    with tf.name_scope("predict_trend"):
        predict_loss = tf.losses.softmax_cross_entropy(
            onehot_labels=y_one_hot, logits=self.y_pred)
        self.predict_loss = tf.identity(predict_loss, name='predict_loss')

        add_moving_summary(self.predict_loss)

        trend_labels_idx = tf.argmax(
            y_one_hot, axis=-1, name='trend_labels_idx')
        y_pred_idx = tf.argmax(self.y_pred, axis=-1, name='y_pred_idx')
        y_pred_one_hot = tf.one_hot(
            y_pred_idx, depth=self.hps.C, name='y_pred_one_hot')
        _, accuracy = tf.metrics.accuracy(
            labels=trend_labels_idx, predictions=y_pred_idx)

        # correct_prediction = tf.equal(tf.argmax(y_one_hot, -1), tf.argmax(self.y_pred, -1))
        # accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        self.predict_accuracy = tf.identity(accuracy, name='accuracy_')

        # tf.summary.scalar('prediction-accuracy', self.predict_accuracy)
        # tf.summary.scalar('prediction-loss', self.predict_loss)
        add_moving_summary(self.predict_accuracy)

    with tf.name_scope("decode"):
        if self.hps.check_error_x_recon:
            # total_x_con = tf.reduce_mean(self.x_con, name='check_error')
            a, b = tf.nn.moments(self.x_con, axes=[1, 2], keep_dims=False)
            tf.summary.scalar(name='check_recon_mu', tensor=tf.reduce_mean(a))
            tf.summary.scalar(name='check_recon_std', tensor=tf.reduce_mean(b))
            tf.summary.scalar(name='check_recon_norm',
                              tensor=tf.reduce_mean(tf.abs(self.x_con)))

        self.log_mse_loss = tf.log(tf.losses.mean_squared_error(
            labels=self.x_con, predictions=x_hat))

        tf.summary.scalar('log_mse', self.log_mse_loss)

        # # cross_entropy
        eps = 1e-10
        x_con_clip = tf.clip_by_value(self.x_con, eps, 1 - eps)

        if self.hps.normalize_data is ('min_max' or 'min_max_centralize'):
            self.log_lik_loss = - tf.reduce_mean(
                x_hat * tf.log(self.x_con) + (1 - x_con_clip) * tf.log(1 - x_con_clip))
            tf.summary.scalar('log_likelihood_reconstructed',
                              self.log_lik_loss)
            self.px_loss = self.log_lik_loss

        elif self.hps.normalize_data is 'z_score':
            self.mse_loss = tf.losses.mean_squared_error(
                labels=self.x_con, predictions=x_hat)
            tf.summary.scalar('mse', self.mse_loss)
            self.px_loss = self.log_mse_loss

    with tf.name_scope("regularization"):
        if self.hps.l2_loss_eta is not 0.0:
            vars = tf.trainable_variables()
            self.l2_loss = tf.add_n([tf.nn.l2_loss(v)
                                     for v in vars]) * self.hps.l2_loss_eta

            tf.summary.scalar('regularization_loss', self.l2_loss)
        else:
            self.l2_loss = 0.0

    # self.total_cost = tf.add_n(
    #     # [self.hps.alpha * self.px_loss, 0.05 * self.auto_corr_loss, self.hps.beta * self.predict_loss,
    #     #  self.hps.alpha * self.teacher_force_loss, self.hps.gamma * self.latent_loss, self.l2_loss], name="total_cost")
    #     [self.hps.alpha * self.px_loss, self.auto_corr_loss, self.hps.beta * self.predict_loss, self.hps.gamma * self.latent_loss, self.l2_loss], name="total_cost")

    # self.mse_loss
# [1110 11:24:13 @monitor.py:467] QueueInput/queue_size: 50
# [1110 11:24:13 @monitor.py:467] decode_1/check_recon_mu: 0.15587
# [1110 11:24:13 @monitor.py:467] decode_1/check_recon_norm: 0.15587
# [1110 11:24:13 @monitor.py:467] decode_1/check_recon_std: 0.15565
# [1110 11:24:13 @monitor.py:467] decode_1/log_mse: -0.27268
# [1110 11:24:13 @monitor.py:467] decode_1/mse: 0.76134
# [1110 11:24:13 @monitor.py:467] encode_2/check_z_norm: 6.6843
# [1110 11:24:13 @monitor.py:467] encode_2/latent_loss: 2.6277e-05
# [1110 11:24:13 @monitor.py:467] predict_trend/accuracy_: 0.68539
# [1110 11:24:13 @monitor.py:467] predict_trend/predict_loss: 0.41498
# [1110 11:24:13 @monitor.py:467] regularization/regularization_loss: 0.00015878
# [1110 11:24:13 @monitor.py:467] total-loss: 0.12574
# [1110 11:24:13 @monitor.py:467] val_precision: 0.52832
# [1110 11:24:13 @monitor.py:467] val_recall: 0.52832
# [1110 11:24:13 @monitor.py:467] validation_predict_trend/accuracy_: 0.52862

    params = [1./.27268, 1./.041498, 1./2.6277e-05, 1./.00015878]
    self.total_loss = tf.add_n(
        [
            params[0] * self.px_loss,
            params[1] * self.predict_loss,
            params[2] * self.latent_loss,
            params[3] * self.l2_loss
        ],
        name="total_cost")

    tf.summary.scalar('total-loss', self.total_loss)
