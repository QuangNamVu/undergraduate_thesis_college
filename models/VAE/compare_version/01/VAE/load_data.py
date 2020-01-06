import numpy as np
import pandas as pd
from tensorpack.dataflow.base import RNGDataFlow


class LoadData(RNGDataFlow):
    def __init__(self, X, X_tau, y_one_hot, shuffle=False):

        self.shuffle = shuffle

        self.x = X
        self.x_hat = X_tau
        self.y_one_hot = y_one_hot

    def __len__(self):
        return self.x.shape[0]

    def __iter__(self):
        idxs = list(range(self.__len__()))
        if self.shuffle:
            self.rng.shuffle(idxs)
        for k in idxs:
            x_element = self.x[k]
            x_hat_element = self.x_hat[k]
            y_one_hot_element = self.y_one_hot[k]
            yield [x_element, x_hat_element, y_one_hot_element]


def load_data_seq(hps):
    # df = resample_data(info_params)
    df = pd.read_csv(filepath_or_buffer=hps.data_file_name)

    attributes_normalize_mean = hps.attributes_normalize_mean
    attributes_normalize_log = hps.attributes_normalize_log

    next_deltaClose = df.Close.diff(periods=1).dropna()

    n_attributes_total = attributes_normalize_mean + attributes_normalize_log

    X = df[n_attributes_total]

    # if lagtime = 10: X[diff](index 10)= X[diff][0] = X_selected[10]- X_selected[0]
    if hps.is_differencing and hps.lag_time is not 0:
        X = X.diff(periods=hps.lag_time).dropna()
    else:
        hps.lag_time = 0  # Not shift when not differencing

    for att in attributes_normalize_log:
        X[att] = np.log(X[att] + 1e-20)

    if hps.normalize_data_idx:
        mu = X[0:hps.N_train_seq].mean(axis=0)
        X_max = X[0:hps.N_train_seq].max(axis=0)
        X_min = X[0:hps.N_train_seq].min(axis=0)
    else:
        mu = np.mean(X, axis=0)
        X_max = np.max(X, axis=0)
        X_min = np.min(X, axis=0)

    assert False not in (X_max > X_min).index.values, print(
        "No variance found X_max == X_min")

    np.savez(hps.scaler_path, mu=mu, X_min=X_min, X_max=X_max)
    normalization = hps.normalize_data
    if normalization is 'z_score':
        print('Normalize: Z score')
        if hps.normalize_data_idx:
            X_std = X[0:hps.N_train_seq].std(axis=0)
        else:
            X_std = np.std(X, axis=0)

        np.savez(hps.scaler_path, mu=mu, X_min=X_min, X_max=X_max, X_std=X_std)
        X = (X - mu) / X_std

    elif normalization is 'min_max':
        print('Normalize: Min Max')
        X = (X - X_min) / (X_max - X_min)

    elif normalization is 'min_max_centralize':
        print('Normalize: Min Max Centralize')
        X = (X - mu) / (X_max - X_min)
    else:
        print('Missing Normalization')

    return X, next_deltaClose


def segment_seq(dfX, df_deltaClose, hps):
    T, D, Tau = hps.T, hps.D, hps.Tau
    # window slide
    next_shift = 1
    start_idx = dfX.index[0]
    end_idx = dfX.index[-1] - T - Tau - 1
    n_segments = end_idx - start_idx
    segment = np.zeros(shape=(n_segments, T, D))
    next_segment = np.zeros(shape=(n_segments, T, D))

    delta_segment = np.zeros(shape=(n_segments, Tau))
    for i, idx_df in enumerate(range(start_idx, end_idx)):
        segment[i] = dfX.loc[idx_df: idx_df + T - 1].values
        next_segment[i] = dfX.loc[idx_df +
                                  next_shift: idx_df + next_shift + T - 1].values
        delta_segment[i] = df_deltaClose.loc[idx_df +
                                             T: idx_df + T + Tau - 1].values

    target_classify = np.copy(delta_segment)
    if hps.C is 3:
        # return -1 0 1 => 0 1 2 for classify
        # target_classify = np.sign(delta_segment).astype(int) + 1
        target_classify = np.where(
            delta_segment > 0, 2, (np.where(delta_segment == 0, 1, 0)))
    elif hps.C is 2:
        target_classify = np.where(delta_segment >= 0, 1, 0)

    target_one_hot = np.eye(hps.C)[target_classify]

    return segment, next_segment, target_one_hot


def train_test_split(segment, next_segment, target_one_hot, hps):
    idx_split = hps.N_train_seq - hps.T + 1
    assert idx_split < segment.shape[0], print(
        "len data must greater than idx split")

    train_segment = segment[:idx_split]
    test_segment = segment[idx_split:]

    train_next_shift = next_segment[:idx_split]
    test_next_shift = next_segment[idx_split:]

    train_target_one_hot = target_one_hot[:idx_split]
    test_target_one_hot = target_one_hot[idx_split:]

    return train_segment, test_segment, train_next_shift, test_next_shift, train_target_one_hot, test_target_one_hot


def test_segment(X, hps):
    """
    input df test  [N + 1, T, D] from 0 to T
    lag_time = 1
    hps: scaler

    return segment [N - T + 1, T, D] diff from 1-0 to T - (T - 1)
    """

    if hps.is_differencing and hps.lag_time is not 0:
        X = X.diff(periods=hps.lag_time).dropna()

    normalization = hps.normalize_data
    scaler = np.load(hps.scaler_path)
    mu = scaler['mu']
    X_min = scaler['X_min']
    X_max = scaler['X_max']

    if normalization is 'z_score':
        print('Inference Normalize: Z score')
        X_std = scaler['X_std']
        X = (X - mu) / X_std

    elif normalization is 'min_max':
        print('Inference Normalize: Min Max')
        X = (X - X_min) / (X_max - X_min)

    elif normalization is 'min_max_centralize':
        print('Inference Normalize: Min Max Centralize')
        X = (X - mu) / (X_max - X_min)
    else:
        print('Inference Missing Normalization')

    X = X.values
    N = X.shape[0] - 1  # N + 1, T, D
    T, D = hps.T, hps.D
    segment = np.zeros(shape=(N - T + 1, T, D))
    for i in range(N - T + 1):
        segment[i] = X[i:i+T]

    return segment
