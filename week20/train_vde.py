import datetime
import timeit

from info_params import params
from read_data import *
from vde import VariationalDynamicalEncoder

if __name__ == '__main__':
    info_params = params()
    info_params.create_next_trend = True
    # info_params.data_file_name = '/content/gdrive/My\ Drive/Colab\ Notebooks/Data/week2/golden.csv'

    X_train_seq, X_val_seq, X_tau_train_seq, X_tau_val_seq, y_one_hot_train_seq, y_one_hot_val_seq = load_data(
        info_params)

    N_seq, D = np.shape(X_train_seq)

    X_train, X_tau_train, y_one_hot_train = create_batch_data(X_train_seq, X_tau_train_seq, info_params,
                                                              target_one_hot=y_one_hot_train_seq)

    X_val, X_tau_val, y_one_hot_val = create_batch_data(X_val_seq, X_tau_val_seq, info_params,
                                                        target_one_hot=y_one_hot_val_seq)

    VDE = VariationalDynamicalEncoder(params=info_params)

    start = timeit.default_timer()

    VDE.train(x=X_train, x_con=X_train, y_one_hot_train=y_one_hot_train,
              x_valid=X_val, x_con_valid=X_tau_val, y_one_hot_valid=y_one_hot_val,
              params=info_params,
              # load_path='/home/nam/tmp/vde',
              # load_path='default',
              save_path='/home/nam/tmp/vde/')

    stop = timeit.default_timer()

    print('Time: %.2f secs' % (stop - start))
    print('Current time is: %s' % datetime.datetime.now())
