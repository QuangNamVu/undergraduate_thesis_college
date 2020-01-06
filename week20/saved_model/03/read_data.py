import numpy as np
import pandas as pd


def load_data(info_params):
    df = pd.read_csv(info_params.data_file_name)

    attributes_normalize_mean = ['Timestamp', 'Open', 'High', 'Low', 'Close', 'Weighted_Price']
    attributes_normalize_log = ['Volume_(BTC)', 'Volume_(Currency)']

    n_attributes_total = attributes_normalize_mean + attributes_normalize_log

    X_selected = df[n_attributes_total]

    # y_selected = df['delta']
    X_0 = X_selected.copy()

    for att in attributes_normalize_log:
        X_0[att] = np.log(X_selected[att])

    idx_split = info_params.idx_split

    mu = np.mean(X_0[0:idx_split])
    X_max = np.max(X_0[0:idx_split])
    X_min = np.min(X_0[0:idx_split])
    X_std = np.std(X_0[0:idx_split])

    normalize = info_params.normalize_data
    if normalize is 'z_score':
        X_normalized = (X_0 - mu) / X_std

    elif normalize is 'min_max':
        X_normalized = (X_0 - X_min) / (X_max - X_min)

    else:
        print('Missing Scale')
        X_normalized = X_0

    X_normalized = X_normalized.values

    tau = info_params.offset_time
    target = np.roll(X_normalized, -tau, axis=0)

    X_train = X_normalized[0:idx_split]
    X_test = X_normalized[idx_split:]

    y_train = target[0:idx_split]
    y_test = target[idx_split:]

    return X_train, X_test, y_train, y_test


def next_batch(training_data, target_data, idx_T, batch_T):
    X_train = training_data[idx_T * batch_T:(idx_T + 1) * batch_T, :]
    y_train = target_data[idx_T * batch_T:(idx_T + 1) * batch_T, :]
    return X_train, y_train


def create_batch_data(X, y, batch_size=128, num_batch=782):
    assert batch_size * num_batch != 0, 'batch size = 0 num batch = 0'

    assert len(np.shape(X)) >= 2, 'Dim Tensor >= 3'

    N, D = np.shape(X)

    X_out = np.zeros(shape=(num_batch, batch_size, D))
    y_out = np.zeros(shape=(num_batch, batch_size, D))

    # window slide
    for i in range(num_batch):
        X_out[i] = X[i * batch_size: (i + 1) * batch_size]

        y_out[i] = y[i * batch_size: (i + 1) * batch_size]

    return X_out, y_out
