import os
class Config(object):
    def __init__(self):
        self.mpqa_lexicon = None
        
        self.dim_domain = 100
        self.dim_general = 300
        
        # model control
        self.rnn_cell = 'regu'
        self.hidden_size = 300
        self.cross_share_k = 5
        self.ate_output_topology = 'same' # value : { 'same', 'diff'}
        self.do_aul = True
        self. do_aus = True
        self.dropout_rate = 0.50 # controlled
        
        # data
        self.max_iter = None
        
        # hyperparameter
        self.batch_size = 4
        self.epoch = 15
        self.patience = 5
        self.verbose = 1

        # derivative variable
        self.max_sentence_size = 40
        self.max_word_size = None
