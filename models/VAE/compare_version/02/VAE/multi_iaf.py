import tensorflow as tf
from tensorpack import *
from .tf_utils.ar_layers import *
from .tf_utils.common import *

def multi_iaf(self, z, reuse=False):
    with tf.variable_scope('multi_iaf', reuse=reuse):
        # [M, f1, n_z] -> [ M, f1, 2 *nz]
        lst_ms = ar_conv1d(name="ar_conv_l1", value=z, kernel_size=self.hps.lst_kernels_iaf[0], in_C=self.hps.n_z,
                           out_C=2 * self.hps.n_z, reuse=reuse)

        # [M, f1, 2* n_z] -> [M, f1, nz] & [M, f1, nz]
        m, s = split(lst_ms, 2, [self.hps.n_z, self.hps.n_z])
        sigma1 = tf.sigmoid(s)
        z = sigma1 * z + (1 - sigma1) * m

        # [M, f1, n_z] -> [ M, f1, 2 *nz]
        lst_ms = ar_conv1d(name="ar_conv_l2", value=z, kernel_size=self.hps.lst_kernels_iaf[1], in_C=self.hps.n_z,
                           out_C=2 * self.hps.n_z, reuse=reuse)

        # [M, f1, 2* n_z] -> [M, f1, nz] & [M, f1, nz]
        m, s = split(lst_ms, 2, [self.hps.n_z, self.hps.n_z])
        sigma2 = tf.sigmoid(s)
        z = sigma2 * z + (1 - sigma2) * m
        lgsm_iaf = tf.log(sigma1) + tf.log(sigma2)

        return z, lgsm_iaf
