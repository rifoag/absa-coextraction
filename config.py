import os
class Config(object):
    def __init__(self):
        self.dim_domain = 100
        self.dim_general = 300
        
        # model control
        self.rnn_cell = 'lstm'
        self.do_cross_share = False
        self.dropout_rate = 0.50
        self.cross_share_k = 5
        self.hidden_size = 300
        
        # data
        self.max_iter = None
        
        # hyperparameter
        self.batch_size = 16
        self.epoch = 15
        self.patience = 1
        self.verbose = 1

        # derivative variable
        self.n_aspect_tags = 0
        self.n_polarity_tags = 0
        self.n_joint_tags = 0
        self.n_poss = 0
        self.n_chunks = 0
        self.n_words = 0
        self.max_sentence_size = 0
        self.max_word_size = None
