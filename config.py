import os
class Config(object):
    def __init__(self):
        self.dim_domain = 100
        self.dim_general = 300
        
        # model control
        self.rnn_cell = "lstm"
        self.do_cross_share = False
        self.cross_share_k = 5
        
        # default
        self.crf_loss = True
        self.train_embeddings = False
        self.current_path = "."
        
        # data
        self.max_iter = None

        self.nepochs = 50
        self.dropout_rate = 0.55
        self.batch_size = 16
        self.learning_rate = 0.001
        self.learning_rate_decay = 0.95
        self.decay_steps = 500
        self.nepoch_no_imprv = 10

        self.test_batch_size = 256
        self.hidden_size = 300
        self.char_hidden_size = 100

        # derivative variable
        self.n_aspect_tags = 0
        self.n_polarity_tags = 0
        self.n_joint_tags = 0
        self.n_poss = 0
        self.n_chunks = 0
        self.n_words = 0
        self.max_sentence_size = 0
        self.max_word_size = None
