# -*- coding: utf-8 -*-
# Ref: Yuxin Wu file GAN.py
# Author: VQNam

import tensorflow as tf
import numpy as np
from tf_utils.ar_layers import *
from tf_utils.common import *
from tf_utils.distributions import *
from tensorpack import *
from tensorpack import ModelDescBase, StagingInput, TowerTrainer
from tensorpack.tfutils.tower import TowerContext, TowerFuncWrapper
from tensorpack.tfutils.summary import *
from tensorpack.utils.argtools import memoized_method

# from utils import *
from encoder import encoder
from decoder import decoder
from predicter import predict
from multi_iaf import multi_iaf
from build_loss import build_losses


class VDEModelDesc(ModelDescBase):
    def __init__(self, hps):
        self.hps = hps

    def inputs(self):
        return [
            tf.TensorSpec((None, self.hps.T, self.hps.D), tf.float32, 'x'),
            tf.TensorSpec((None, self.hps.T, self.hps.D), tf.float32, 'x_con'),
            tf.TensorSpec((None, self.hps.Tau, self.hps.C), tf.float32, 'y_one_hot')
        ]

    def collect_variables(self, encode_scope='encode', predict_scope='predict', decode_scope='decode'):
        self.encode_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, encode_scope)
        assert self.encode_vars, "Encode graph not found"

        self.predict_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, predict_scope)
        assert self.predict_vars, "Predict graph not found"

        self.decode_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, decode_scope)
        assert self.decode_vars, "Decode graph not found"

    def encoder(self, x):
        return encoder(self, x)

    def predict(self):
        return predict(self)

    def decoder(self, z):
        return decoder(self, z)

    def multi_iaf(self, z):
        return multi_iaf(self, z)

    def build_losses(self, y_one_hot, x_hat):
        build_losses(self, y_one_hot, x_hat)

    def build_graph(self, x, x_hat, y_one_hot):

        with tf.variable_scope('encode', reuse=False):
            self.z_mu, self.std, self.z, self.state_c = self.encoder(x)
            # noise = get_noise(shape=tf.shape(self.z_mu))
            # self.z = self.z_mu +  noise * tf.exp(0.5 * self.z_lsgms)

        if self.hps.is_IAF:
            with tf.variable_scope('encode', reuse=False):
                self.z_iaf, self.z_lgsm_iaf = self.multi_iaf(self.z)

            with tf.variable_scope('predict', reuse=False):
                self.y_pred = self.predict()
            with tf.variable_scope('decode', reuse=False):
                self.x_con = self.decoder(self.z_iaf)
            with tf.variable_scope('encode', reuse=True):
                _, _, self.z_tau, _ = self.encoder(self.x_con)
                # self.z_tau = self.z_tau_mu + noise * tf.exp(0.5 * self.z_tau_lsgms)
                self.z_tau_lsgm_iaf, self.z_tau_iaf = self.multi_iaf(self.z_tau)

        else:
            with tf.variable_scope('predict', reuse=False):
                self.y_pred = self.predict()
            with tf.variable_scope('decode', reuse=False):
                self.x_con = self.decoder(self.z)
            with tf.variable_scope('encode', reuse=True):
                _, _, self.z_tau, _ = self.encoder(self.x_con)
                # self.z_tau = self.z_tau_mu + noise * tf.exp(0.5 * self.z_tau_lsgms)

        self.build_losses(y_one_hot=y_one_hot, x_hat=x_hat)

        self.collect_variables()

    def optimizer(self):
        optimizer_origin = tf.train.AdamOptimizer(learning_rate=self.hps.learning_rate)

        return tf.contrib.estimator.clip_gradients_by_norm(optimizer_origin, clip_norm=1.0)

    @memoized_method
    def get_optimizer(self):
        return self.optimizer()


class VDETrainer(TowerTrainer):

    def __init__(self, input, model, num_gpu=1):
        """
        Args:
            input (InputSource):
            model (VDEModelDesc):
        """
        super(VDETrainer, self).__init__()
        assert isinstance(model, VDEModelDesc), model

        if num_gpu > 1:
            input = StagingInput(input)

        # Setup input
        cbs = input.setup(model.get_input_signature())
        self.register_callback(cbs)

        assert num_gpu <= 1, "Should be 1 gpu for small data"

        self._build_vde_trainer(input, model)

    def _build_vde_trainer(self, input, model):
        """
        Args:
            input (InputSource):
            model (VDEModelDesc):
        """
        # Build the graph
        self.tower_func = TowerFuncWrapper(model.build_graph, model.get_input_signature())
        with TowerContext('', is_training=True):
            self.tower_func(*input.get_input_tensors())
        opt = model.get_optimizer()

        with tf.name_scope('optimize'):
            vde_min = opt.minimize(model.total_loss,
                                   var_list=[model.encode_vars, model.predict_vars, model.decode_vars], name='train_op')
        self.train_op = vde_min
