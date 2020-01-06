import timeit

from info_params import params
from read_data import *
from vde import VariationalDynamicalEncoder

if __name__ == '__main__':
    info_params = params()
    info_params.create_next_trend = True  # True if create model with trend is part of target
    # info_params.data_file_name = '/content/gdrive/My\ Drive/Colab\ Notebooks/Data/week2/golden.csv'

    X_train_seq, X_test_seq, X_tau_train_seq, X_tau_test_seq, y_one_hot_train_seq, y_one_hot_test_seq = load_data(
        info_params)

    N_seq, D = np.shape(X_train_seq)

    dim_x = 128 * 8
    dim_z = 128
    epochs = 300
    learning_rate = 3e-4
    l2_loss = 1e-6  # L2 Regularisation weight
    seed = 9973

    X_train, X_tau_train, y_one_hot_train = create_batch_data(X_train_seq, X_tau_train_seq, info_params,
                                                              target_one_hot=y_one_hot_train_seq)

    VDE = VariationalDynamicalEncoder(params=info_params, dim_x=dim_x, dim_z=dim_z, dim_recon=dim_x, l2_loss=l2_loss)

    start = timeit.default_timer()

    VDE.train(x=X_train, x_con=X_train, x_valid=X_train, x_con_valid=X_train, y_one_hot=y_one_hot_train, epochs=epochs,
              num_batches=32, learning_rate=learning_rate, seed=seed, stop_iter=350, print_every=10,
              # load_path='/home/nam/tmp/vde',4
              # load_path='default',
              save_path='/home/nam/tmp/vde/')

    stop = timeit.default_timer()

    print('Time: %.2f secs' % (stop - start))
