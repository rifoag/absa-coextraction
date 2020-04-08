import os
class Config(object):
    def __init__(self):
        self.dim_domain = 100
        self.dim_general = 300
        
        # model control
        self.rnn_cell = 'regu'
        self.do_cross_share = True
        self.dropout_rate = 0.50
        self.cross_share_k = 5
        self.hidden_size = 300
        
        # data
        self.max_iter = None
        
        # hyperparameter
        self.batch_size = 4
        self.epoch = 3
        self.patience = 1
        self.verbose = 1

        # derivative variable
        self.max_sentence_size = 40
        self.max_word_size = None
