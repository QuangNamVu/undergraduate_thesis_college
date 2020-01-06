# class params(object, params_file=''):
class params(object):

    def __init__(self):
        self.data_file_name = '/home/nam/data/week2/golden.csv'
        self.normalize_data = 'z_score'
        # self.normalize_data = 'min_max'
        # self.normalize_data = 'default'
        self.attributes_normalize_mean = ['Timestamp', 'Open', 'High', 'Low', 'Close', 'Weighted_Price']
        self.attributes_normalize_log = ['Volume_(BTC)', 'Volume_(Currency)']

        self.idx_split = 131072  # Split sequence to test and train
        self.learning_rate = 1e-3

        self.n_time_steps = 128  # T
        # T' = T + 128 => offset_time = 128  offset_time > n_time_steps
        self.offset_time = 128
        self.print_every = 10

        self.n_samples = self.idx_split // self.n_time_steps  # N sample
        self.batch_size = 64  # M
        self.n_batch = self.n_samples // self.batch_size  # N
        self.n_batch_test = 12  # T test 1d approx 12 batch * 128 transaction

        self.l2_loss = 1e-6  # l2 regularization
        self.d_inputs = 8
        self.d_outputs = 8
        self.n_z = 50
        self.lstm_units = 16
        self.create_next_trend = True  # True if predict next seq
        self.pred_seq_len = 64  # Tau

        self.num_iterations = 300

        self.n_classes = 2  # TODO Up Down and stable
        self.stop_iter = 10000
        self.epochs = 3000
        self.seed = 9973
