import os
class Config():
    def __init__(self):
        self.basepath = "/data/eric/CSIRE"
        self.data = '/data/eric/CSIRE/mimic3_benchmarks/mimic3models/in_hospital_mortality/'
        self.timestep = 1.0
        self.normalizer_state = "/data/eric/CSIRE/mimic3_benchmarks/mimic3models/in_hospital_mortality/ihm_ts1.0.input_str-previous.start_time-zero.normalizer"
        self.imputation = 'previous'
        self.small_part = False
        self.textdata = self.basepath + 'text/'
        self.buffer_size = 100
        self.learning_rate = 2e-5 #5e-6 #
        self.max_len = 128
        self.break_text_at = 300
        self.padding_type = 'Zero'
        self.ihm_path = "/data/eric/CSIRE/scr/data-mimic3/root/"
        self.textdata_fixed = "/data/eric/CSIRE/scr/data-mimic3/root/text_fixed"
        self.starttime_path = "/data/eric/CSIRE/scr/data-mimic3/root/T0/train_starttime.pkl"
        self.maximum_number_events = 150
        self.test_textdata_fixed = "/data/eric/CSIRE/mimic3_benchmarks/data/root/test_text_fixed/"
        self.test_starttime_path = "/data/eric/CSIRE/scr/data-mimic3/root/T0/test_starttime.pkl"
        self.dropout = 0.9 #keep_prob