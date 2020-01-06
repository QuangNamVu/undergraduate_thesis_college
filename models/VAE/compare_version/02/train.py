#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Ref: DCGAN.py Author: Yuxin Wu
# Author: Nam VQ


import argparse
import pickle
import numpy as np
import os
import tensorflow as tf
from tensorpack.utils import logger
from tensorpack import *
from tensorpack.tfutils.scope_utils import auto_reuse_variable_scope
# from tensorpack.tfutils import summary

from VAE.Model import ModelDesc, Trainer  # , RandomZData
from VAE.info_params import get_default_hparams
from VAE.load_data import *


# prefetch data
def get_data(hps):
    dfX, df_next_deltaClose = load_data_seq(hps)

    segment, next_segment, target_one_hot = segment_seq(dfX, df_next_deltaClose, hps)

    train_segment, test_segment, train_next_shift, test_next_shift, train_target_one_hot, test_target_one_hot =\
    train_test_split(segment, next_segment, target_one_hot, hps)

    print("Train shape: ", train_segment.shape)
    print("Test shape: ", test_segment.shape)

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

    with open("./hps/hps.pickle", "wb") as output_file:
        pickle.dump(hps, output_file)

    M = ModelDesc(hps)

    logger.auto_set_dir(action='d')
    ds_train, ds_test = get_data(hps)

    Trainer(
        input=QueueInput(ds_train), model=M).train_with_defaults(
        callbacks=[
            ModelSaver(),
            callbacks.MergeAllSummaries(),
            MinSaver('total_loss'),
            # InferenceRunner(ds_test, [ScalarStats('predict_trend/accuracy_')])
            InferenceRunner(ds_test, [
                ScalarStats('predict_trend/accuracy_'),
                BinaryClassificationStats(pred_tensor_name='predict_trend/y_pred_one_hot',
                                          label_tensor_name='y_one_hot')])
        ],
        steps_per_epoch=hps.steps_per_epoch,
        max_epoch=hps.epochs,
        session_init=SaverRestore(args.load) if args.load else None
    )
