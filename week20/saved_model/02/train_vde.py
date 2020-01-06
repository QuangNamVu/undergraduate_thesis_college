import timeit

from info_params import params
from read_data import *
from vde import VariationalDynamicalEncoder

if __name__ == '__main__':
    info_params = params()
    X_train_seq, X_test_seq, y_train_seq, y_test_seq = load_data(info_params)

    N_seq, D = np.shape(X_train_seq)

    dim_x = 128 * 8
    dim_z = 128
    epochs = 300
    learning_rate = 3e-4
    l2_loss = 1e-6  # L2 Regularisation weight
    seed = 9973

    X_train, y_train = create_batch_data(X_train_seq, y_train_seq,
                                         batch_size=info_params.n_time_steps,
                                         num_batch=info_params.n_batch)

    X_train = np.reshape(X_train, newshape=(-1, info_params.n_time_steps * info_params.d_inputs))

    y_train = np.reshape(y_train, newshape=(-1, info_params.n_time_steps * info_params.d_inputs))

    VDE = VariationalDynamicalEncoder(dim_x=dim_x, dim_z=dim_z, dim_recon=dim_x, l2_loss=l2_loss)

    start = timeit.default_timer()

    VDE.train(x=X_train, y=y_train, x_valid=X_train, y_valid=y_train, epochs=epochs, num_batches=info_params.n_batch,
              learning_rate=learning_rate, seed=seed, stop_iter=350, print_every=10,
              # load_path='/home/nam/tmp/vde',
              # load_path='default',
              save_path='/home/nam/tmp/vde/')

    stop = timeit.default_timer()

    print('Time: %.2f secs' % (stop - start))
