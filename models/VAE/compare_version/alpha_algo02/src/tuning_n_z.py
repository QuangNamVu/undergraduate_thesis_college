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

from Model import ModelDesc, Trainer  # , RandomZData
from info_params import get_default_hparams
from load_data import *
from tqdm import tqdm


# prefetch data
def get_data(hps):
    dfX, df_next_deltaClose = load_data_seq(hps)

    segment, next_segment, target_one_hot = segment_seq(dfX, df_next_deltaClose, hps)

    train_segment, test_segment, train_next_shift, test_next_shift, train_target_one_hot, test_target_one_hot = \
        train_test_split(segment, next_segment, target_one_hot, hps)

    # X constructed = X tau
    if hps.is_VDE:
        train_data = LoadData(train_segment, train_next_shift, train_target_one_hot, shuffle=True)
        test_data = LoadData(test_segment, test_next_shift, test_target_one_hot, shuffle=False)

    else:
        # X constructed = X
        train_data = LoadData(train_segment, train_segment, train_target_one_hot, shuffle=True)
        test_data = LoadData(test_segment, test_segment, test_target_one_hot, shuffle=False)

    ds_train = ConcatData([train_data])
    ds_test = ConcatData([test_data])

    ds_train = BatchData(ds_train, batch_size=hps.M)
    ds_test = BatchData(ds_test, batch_size=hps.M)

    return ds_train, ds_test


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list oillf GPU(s) to use.')
    parser.add_argument('--load', help='load model')
    parser.add_argument('--flops', action='store_true', help='print flops and exit')
    global args
    args = parser.parse_args()
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    return args


lst_T = list(range(30, 100, 10)) + list(range(1, 10, 1)) + list(range(100, 200, 50))
# lst_D = list(range(1, 10, 1)) + list(range(10, 100, 5)) + list(range(100, 200, 50))
lst_n_z = list(range(5, 20, 2)) + list(range(20, 200, 20))
# lst_D = [1]

hps = get_default_hparams()
hps.steps_per_epoch = 4
hps.epochs = 500
hps.is_VAE = True
hps.is_VDE = True
hps.is_IAF = False
hps.is_differencing = True
hps.lag_time = 1

args = get_args()

for t in tqdm(lst_T):
    for n_z in lst_n_z:
        ckpt_dir = os.path.expanduser("~") + '/tuning/%dT_%dn_z2' % (t, n_z)
        os.makedirs(ckpt_dir, exist_ok=True)
        args.logdir = ckpt_dir
        # hps.lag_time = d
        hps.n_z = n_z
        # hps.lst_kernels[1] = k1
        hps.T = t
        M = ModelDesc(hps)
        ds_train, ds_test = get_data(hps)
        x = Trainer(
            input=QueueInput(ds_train), model=M).train_with_defaults(
            callbacks=[
                ModelSaver(max_to_keep=1, checkpoint_dir=ckpt_dir),
                # ModelSaver(),
                callbacks.MergeAllSummaries(),
                MinSaver('total_loss'),
                InferenceRunner(ds_test, [ScalarStats('predict_trend/accuracy_')])
            ],
            steps_per_epoch=hps.steps_per_epoch,
            max_epoch=hps.epochs,
            session_init=None
        )
        # tf.get_variable_scope().reuse_variables()
        tf.reset_default_graph()
        del M
        del ds_train
        del ds_test
        del x
