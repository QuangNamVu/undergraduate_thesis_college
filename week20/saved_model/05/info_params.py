# idx_split = 99968
idx_split = 100096
learning_rate = .0001
d_inputs = 8  # D
d_outputs = 8  # D int time T+1
lstm_units = 100
batch_size = 1  # N

n_time_steps = 128  # T
offset_time = 128   # tau

n_batch_test = 12  # T test 1d = 1440
data_file_name = '/home/nam/data/week2/golden.csv'
normalize_data = 'z_score'
# normalize_data = 'min_max'
n_z = 32


num_iterations = 10000

attributes_normalize_mean = ['Timestamp', 'Open', 'High', 'Low', 'Close', 'Weighted_Price']
attributes_normalize_log = ['Volume_(BTC)', 'Volume_(Currency)']

checkpoint_path = '/home/nam/tmp/01/checkpoint.ckpt'

# class params(object, params_file=''):
class params(object):

    def __init__(self):
        self.idx_split = idx_split
        self.learning_rate = learning_rate
        self.d_inputs = d_inputs
        self.d_outputs = d_outputs
        self.lstm_units = lstm_units
        self.batch_size = batch_size
        self.n_time_steps = n_time_steps
        self.offset_time = offset_time

        self.n_batch_test = n_batch_test
        self.data_file_name = data_file_name
        self.normalize_data = normalize_data

        self.attributes_normalize_mean = attributes_normalize_mean
        self.attributes_normalize_log = attributes_normalize_log

        self.num_iterations = num_iterations
        self.checkpoint_path = checkpoint_path

        self.n_z = n_z
        # cacl
        self.n_batch = self.idx_split // self.n_time_steps