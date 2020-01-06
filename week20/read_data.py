import numpy as np
import pandas as pd


def load_data(info_params):
    df = pd.read_csv(info_params.data_file_name)

    attributes_normalize_mean = ['Timestamp', 'Open', 'High', 'Low', 'Close', 'Weighted_Price']
    attributes_normalize_log = ['Volume_(BTC)', 'Volume_(Currency)']

    attribute_trend = ['Target']  # Up or Down

    n_attributes_total = attributes_normalize_mean + attributes_normalize_log

    X_selected = df[n_attributes_total]

    trend_selected = df[attribute_trend]

    # y_selected = df['delta']
    X_0 = X_selected.copy()

    for att in attributes_normalize_log:
        X_0[att] = np.log(X_selected[att])

    idx_split = info_params.idx_split
    n_batch_test = info_params.n_batch_test
    T, M = info_params.n_time_steps, info_params.batch_size

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
    X_tau_seq = np.roll(X_normalized, - tau - 1, axis=0)

    X_train = X_normalized[0:idx_split]
    X_test = X_normalized[idx_split:idx_split + n_batch_test * M * T]

    X_tau_train = X_tau_seq[0:idx_split]
    X_tau_test = X_tau_seq[idx_split:idx_split + n_batch_test * M * T]

    if not info_params.create_next_trend:
        return X_train, X_test, X_tau_train, X_tau_test

    # tau + 1 + 1 -> next offset [0: 10] = [129:139] -> 131 in csv
    hidden_target_classify = np.roll(trend_selected.values, - tau - 1 - 1, axis=0)  # shift up

    # one-hot encode with numpy
    target_one_hot = np.eye(info_params.n_classes)[np.reshape(hidden_target_classify, -1)]

    # TODO reason why batch size train (128) >= batch size test classify (64)
    target_one_hot_train = target_one_hot[0:idx_split]
    target_one_hot_test = target_one_hot[idx_split:idx_split + n_batch_test * M * T]
    return X_train, X_test, X_tau_train, X_tau_test, target_one_hot_train, target_one_hot_test


def create_batch_data(X, X_tau, info_params, target_one_hot=None):
    assert info_params.n_time_steps * info_params.n_batch > 0, 'Batch size = 0 num batch = 0'

    assert len(np.shape(X)) >= 2, 'Dim Tensor >= 3'

    T, D = info_params.n_time_steps, info_params.d_inputs
    Tau = info_params.pred_seq_len

    # window slide
    X_out = np.reshape(X, newshape=[-1, T, D])
    X_tau_out = np.reshape(X_tau, newshape=[-1, T, D])

    if not info_params.create_next_trend:
        return X_out, X_tau_out

    assert type(target_one_hot) is np.ndarray, 'Trending classify is missing'
    n_samples = X_out.shape[0]
    y_one_hot_out = np.zeros(shape=(n_samples, Tau, info_params.n_classes))

    for i in range(n_samples):
        y_one_hot_out[i] = target_one_hot[i * T: i * T + Tau]

    return X_out, X_tau_out, y_one_hot_out


def shuffle_data(params, X, X_con, y_classify=None):
    perm = np.arange(X.shape[0])
    np.random.shuffle(perm)

    X_out = X[perm]
    X_con_out = X_con[perm]

    if not params.create_next_trend:
        return X_out, X_con_out

    assert type(y_classify) is np.ndarray, 'y classify missing'

    y_classify_out = y_classify[perm]

    return X_out, X_con_out, y_classify_out
