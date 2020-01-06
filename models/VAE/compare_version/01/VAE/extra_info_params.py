import os
import math
from .tf_utils.hparams import *

home_path = os.path.expanduser("~") + '/'

# data_file_name = home_path + 'data/cryptodatadownload/tac_CoinBase_BTC_1h.csv'
data_file_name = home_path + '/data/ccxt/preprocessing_data/BTC_USDT_binance_5m.csv'
# data_file_name = home_path + 'data/ccxt/binance_BNB_BTC_1h.csv'
checkpoint_path = home_path + 'checkpoint/BTC_USDT_binance_5m/checkpoint'
scaler_path = './hps/scaler.npz'

# attributes_normalize_mean = ['Open', 'High', 'Low', 'Close', 'Volume BTC', 'Volume USD']

attributes_normalize_mean = "Open,High,Low,Close,Volume,N_buy,N_sell,buy_amount_avg,sell_amount_avg,buy_amount_std,sell_amount_std,price_avg,price_std,cost_avg,cost_std,Spread_Open_Close,Spread_High_Low,MA_Close_6,MA_Close_12,MA_Close_288,MA_Close_2880,_1d_Unnamed: 0,_1d_Timestamp,_1d_Open,_1d_High,_1d_Low,_1d_Close,_1d_Volume,_1d_N_buy,_1d_N_sell,_1d_buy_amount_avg,_1d_sell_amount_avg,_1d_buy_amount_std,_1d_sell_amount_std,_1d_price_avg,_1d_price_std,_1d_cost_avg,_1d_cost_std,_1d_Spread_Open_Close,_1d_Spread_High_Low,_1d_MA_Close_6,_1d_MA_Close_12,_1d_MA_Close_288,_1d_MA_Close_2880".split(
    ",")

# attributes_normalize_mean = ['DeltaOpen1', 'DeltaHigh1', 'DeltaLow1', 'DeltaClose1', 'DeltaVolume_BTC1',
#                              'Delta_Volume_USD1', 'DeltaOpen200', 'DeltaHigh200', 'DeltaLow200', 'DeltaClose200',
#                              'DeltaVolume_BTC200', 'Delta_Volume_USD200']

attributes_normalize_log = []
D = len(attributes_normalize_log + attributes_normalize_mean)

is_differencing = True
normalize_data = 'z_score'
# normalize_data = 'min_max'
# normalize_data = 'default'
# 'z_score' 'min_max' 'default' 'min_max_centralize'
normalize_data_idx = True

is_VAE = True
is_VDE = True
is_IAF = False
# normalize_data_idx = False
T = 60
lag_time = 10  # X10 - X0 ;; X11 - X1
Tau = 1
k = math.sqrt(T / Tau)
k = 5

# f = np.array(np.array([600, 150, 2]) * T * D / 100).astype(int)

f = [32, 8]
lst_kernels = [10, 3]
# n_z = int(T / k)
n_z = 4

row_count = sum(1 for row in data_file_name)
# N_train_seq = 35000
N_train_seq = 210383 - 2160 - T + 1

check_error_x_recon = True
check_error_z = True

M = 128  # Batch size
l2_loss = 1e-6

lstm_units = 128
C = 2
steps_per_epoch = 5
epochs = 100000
# epochs = 3000

lst_kernels_iaf = [3, 3]

batch_norm_moment = .99

dropout_rate = .5


# self.idx_split = 262144 // self.check_per_N_mins  # Split sequence to test and train
# self.n_test_seq = 131072 // self.check_per_N_mins


def get_default_hparams():
    return HParams(

        data_file_name=data_file_name,
        checkpoint_path=checkpoint_path,
        scaler_path=scaler_path,

        is_differencing=is_differencing,
        normalize_data=normalize_data,

        normalize_data_idx=normalize_data_idx,

        is_VAE=is_VAE,
        is_VDE=is_VDE,
        is_IAF=is_IAF,

        attributes_normalize_mean=attributes_normalize_mean,
        attributes_normalize_log=attributes_normalize_log,
        T=T,
        lag_time=lag_time,
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
