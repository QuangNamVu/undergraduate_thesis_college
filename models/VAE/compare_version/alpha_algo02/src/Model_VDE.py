#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Ref: DCGAN.py Author: Yuxin Wu
# Author: Nam VQ


import argparse
import numpy as np
import os
import tensorflow as tf
from tensorpack.utils import logger
from tensorpack import *
from tensorpack.tfutils.scope_utils import auto_reuse_variable_scope
# from tensorpack.tfutils import summary

from Model import VDEModelDesc, VDETrainer  # , RandomZData
from info_params import get_default_hparams
from load_data import *


class RandomEpsilon(DataFlow):
    def __init__(self, n_z):
        super(RandomEpsilon, self).__init__()
        self.n_z = n_z

    def __iter__(self):
        while True:
            yield [np.random.normal(loc=0, scale=1, size=[self.n_z, self.n_z])]


# prefetch data
def get_data(hps):

    X_train_seq, X_test_seq, X_tau_train_seq, X_tau_test_seq, target_one_hot_train_seq, target_one_hot_test_seq = load_data_seq(
        hps)

    X_train_segment, X_tau_train_segment, y_one_hot_train_segment = segment_seq(X_train_seq, X_tau_train_seq, hps,
                                                                                target_one_hot=target_one_hot_train_seq)

    X_test_segment, X_tau_test_segment, y_one_hot_test_segment = segment_seq(X_test_seq, X_tau_test_seq, hps,
                                                                             target_one_hot=target_one_hot_test_seq)

    # X constructed = X tau
    if hps.is_VDE:
        train_data = LoadData(X_train_segment, X_tau_train_segment, y_one_hot_train_segment, shuffle=True)
        test_data = LoadData(X_test_segment, X_tau_test_segment, y_one_hot_test_segment, shuffle=False)

    else:
        # X constructed = X
        train_data = LoadData(X_train_segment, X_train_segment, y_one_hot_train_segment, shuffle=True)
        test_data = LoadData(X_test_segment, X_test_segment, y_one_hot_test_segment, shuffle=False)

    # ds_eps = RandomEpsilon(info_params.n_z)

    ds_train = ConcatData([train_data])
    ds_test = ConcatData([test_data])

    # ds = train_data
    ds_train = BatchData(ds_train, batch_size=hps.M)
    ds_test = BatchData(ds_test, batch_size=hps.M)

    return ds_train, ds_test


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')
    parser.add_argument('--load', help='load model')
    parser.add_argument('--flops', action='store_true', help='print flops and exit')
    global args
    args = parser.parse_args()
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    return args


if __name__ == '__main__':
    args = get_args()

    hps = get_default_hparams()
    M = VDEModelDesc(hps)

    logger.auto_set_dir(action='d')
    ds_train, ds_test = get_data(hps)

    VDETrainer(
        input=QueueInput(ds_train), model=M).train_with_defaults(
        callbacks=[
            ModelSaver(),
            callbacks.MergeAllSummaries(),
            MinSaver('total_loss'),
            InferenceRunner(ds_test, [ScalarStats('predict_trend/accuracy_')])
        ],
        steps_per_epoch=hps.steps_per_epoch,
        max_epoch=hps.epochs,
        session_init=SaverRestore(args.load) if args.load else None
    )
