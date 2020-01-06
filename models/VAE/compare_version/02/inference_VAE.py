import pymongo
import pandas as pd
from pymongo import MongoClient

import numpy as np
from VAE import get_default_hparams
from VAE.Model import ModelDesc, Trainer  # , RandomZData
from tensorpack.tfutils import SmartInit
from tensorpack.predict.config import PredictConfig
from tensorpack.predict import OfflinePredictor
import VAE
from VAE.load_data import *


def softmax(x):
    "x shape is [N, Tau, C] when Tau == 1"
    assert np.shape(x)[1] == 1, "Tau is not 1"
    x = np.squeeze(x, axis=(1,))
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / e_x.sum(axis=-1, keepdims=True)


def inference(input_data, hps, predictor=None):

    # input data is numpy has shape: [M, T, D]
    # return softmax with C classes
    if predictor is None:
        M = ModelDesc(hps)
        pred_config = PredictConfig(session_init=SmartInit(hps.checkpoint_path),
                                    model=M,
                                    input_names=['x'],
                                    output_names=['predict/y_pred']
                                    )
        predictor = OfflinePredictor(pred_config)

    outputs = predictor(input_data)
    rs = softmax(outputs[0])

    return rs


if __name__ == "__main__":
    checkpoint_path = './train_log/train/checkpoint'
    hps = get_default_hparams()

    mongo_client = MongoClient('localhost', 27017)
    db = mongo_client.crypto_currency
    collection = db['ohlcv']
    market = 'binance'
    symbol = 'BNB/BTC'
    timewindow = '1h'
    query = {'market': market, 'symbol': symbol, 'timewindow': timewindow}

    df_data = pd.DataFrame(list(collection.find(query)))[-hps.M - 100:]
    df_data = df_data[hps.attributes_normalize_mean]

    model = ModelDesc(hps)
    pred_config = PredictConfig(session_init=SmartInit(hps.checkpoint_path),
                                model=model,
                                input_names=['x'],
                                output_names=['predict/y_pred']
                                )
    predictor = OfflinePredictor(pred_config)

    # input_data = np.random.rand(hps.M, hps.T, hps.D)
    input_data =  test_segment(df_data, hps)

    rs = inference(input_data[-hps.M:], hps=hps, predictor=predictor)