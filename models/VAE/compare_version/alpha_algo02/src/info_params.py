import os
import math
from tf_utils.hparams import *

home_path = os.path.expanduser("~") + '/'

# data_file_name = home_path + 'data/cryptodatadownload/tac_CoinBase_BTC_1h.csv'
data_file_name = home_path + 'data/cryptodatadownload/tiny.csv'
# date_fmt = '%Y-%m-%d %H-%p'
date_fmt = '%Y-%m-%d %I-%p'
attributes_normalize_mean = ['Open', 'High', 'Low', 'Close', 'Volume BTC', 'Volume USD']
attributes_normalize_log = []
D = len(attributes_normalize_log + attributes_normalize_mean)

is_differencing = True
lag_time = 1  # X10 - X0 ;; X11 - X1

normalize_data = 'z_score'
# normalize_data = 'min_max'
# normalize_data = 'default'
# 'z_score' 'min_max' 'default' 'min_max_centralize'
normalize_data_idx = True

is_VAE = True
is_VDE = False
is_IAF = False
# normalize_data_idx = False

alpha = 20
beta = 10
gamma = 5
T = 30
Tau = 1
next_shift_VDE = 0
f = [64, 32, 16]
lst_kernels = [33, 13, 5]
lst_T = [30, 30, 15, 9]
# n_z = int(T / k)
n_z = 16

N_train_seq = 3500

check_error_x_recon = True
check_error_z = True

M = 1024  # Batch size
l2_loss = 1e-6

lstm_units = 50
C = 3
steps_per_epoch = 20
epochs = 3000

lst_kernels_iaf = [3, 3]

batch_norm_moment = .99

dropout_rate = .4


def get_default_hparams():
    return HParams(

        data_file_name=data_file_name,
        date_fmt=date_fmt,
        is_differencing=is_differencing,
        normalize_data=normalize_data,
        lst_T=lst_T,
        normalize_data_idx=normalize_data_idx,

        is_VAE=is_VAE,
        is_VDE=is_VDE,
        is_IAF=is_IAF,

        alpha=alpha,
        beta = beta,
        gamma = gamma,

        attributes_normalize_mean=attributes_normalize_mean,
        attributes_normalize_log=attributes_normalize_log,
        T=T,
        lag_time=lag_time,
        next_shift_VDE=next_shift_VDE,
        M=M,
        N_train_seq=N_train_seq,

        learning_rate=1e-3,

        l2_loss=l2_loss,  # l2 regularization
        d_inputs=D,
        d_outputs=D,
        D=D,
        n_z=n_z,
        lstm_units=lstm_units,
        Tau=Tau,
        C=C,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        lst_kernels=lst_kernels,
        lst_kernels_iaf=lst_kernels_iaf,
        f=f,
        batch_norm_moment=.99,
        dropout_rate=dropout_rate,
        check_error_x_recon=check_error_x_recon,
        check_error_z=check_error_z
    )

# self.data_file_name = home_path + 'data/cryptodatadownload/Coinbase_BTCUSD_1h.csv'
# self.data_file_name = home_path + 'data/yahoo_finance/processed_data.csv'
# data_file_name=home_path + 'data/cryptodatadownload/processed_data.csv',
# self.normalize_data = 'min_max_centralize'

# self.data_file_name = home_path + 'data/bitcoin-historical-data/bitstampUSD_1-min_data_2012-01-01_to_2018-06-27.csv'
# data_file_name = home_path + 'data/cryptodatadownload/processed_data.csv'


# attributes_normalize_mean = ['DeltaOpen', 'DeltaHigh', 'DeltaLow', 'DeltaClose', 'DeltaVolume_BTC', 'Delta_Volume_USD']
# attributes_normalize_mean = ['DeltaOpen200', 'DeltaHigh200', 'DeltaLow200', 'DeltaClose200', 'DeltaVolume_BTC200',
#                              'Delta_Volume_USD200']


# attributes_normalize_mean = ['DeltaOpen1', 'DeltaHigh1', 'DeltaLow1', 'DeltaClose1', 'DeltaVolume_BTC1',
#                              'Delta_Volume_USD1', 'DeltaOpen200', 'DeltaHigh200', 'DeltaLow200', 'DeltaClose200',
#                              'DeltaVolume_BTC200', 'Delta_Volume_USD200']
# attributes_normalize_mean = ['Open', 'High', 'Low', 'Close', 'Volume BTC', 'Volume USD']
# attributes_normalize_log = ['Date_Unix']
