from keras import backend as K
from keras.layers import Layer
import numpy as np

class CrossSharedUnit(Layer):
    def __init__(self, output_dim, config, **kwargs):
        self.output_dim = output_dim
        self.config = config
        super(CrossSharedUnit, self).__init__(**kwargs)

    def build(self, input_shape):
        assert isinstance(input_shape, list)
        self.G_aspect_polarity = self.add_weight(name='G_aspect_polarity',
                                        shape=(self.config.hidden_size, self.config.cross_share_k, self.config.hidden_size),
                                        initializer='uniform',
                                        trainable=True)
        self.G_polarity_aspect = self.add_weight(name='G_polarity_aspect',
                                        shape=(self.config.hidden_size, self.config.cross_share_k, self.config.hidden_size),
                                        initializer='uniform',
                                        trainable=True)
        self.G_vector_aspect = self.add_weight(name='G_vector_aspect',
                                        shape=(self.config.cross_share_k, 1),
                                        initializer='uniform',
                                        trainable=True)
        self.G_vector_polarity = self.add_weight(name='G_vector_polarity',
                                        shape=(self.config.cross_share_k, 1),
                                        initializer='uniform',
                                        trainable=True)
        super(CrossSharedUnit, self).build(input_shape)

    def call(self, x):
        assert isinstance(x, list)
        aspect_hidden, polarity_hidden = x
        
        G_aspect_polarity = K.reshape(self.G_aspect_polarity, shape=[self.config.hidden_size, -1])
        G_aspect_polarity = K.tile(K.expand_dims(G_aspect_polarity, axis=0), multiples=K.stack([K.shape(aspect_hidden)[0], 1, 1]))
        shared_hidden_aspect_polarity = K.matmul(aspect_hidden, G_aspect_polarity)
        shared_hidden_aspect_polarity = K.reshape(shared_hidden_aspect_polarity, shape=[-1, self.config.max_sentence_size * self.config.cross_share_k, self.config.hidden_size])
        polarity_hidden_transpose = K.transpose(polarity_hidden, [0, 2, 1])
        shared_hidden_aspect_polarity = K.tanh(K.matmul(shared_hidden_aspect_polarity, polarity_hidden_transpose))
        shared_hidden_aspect_polarity = K.reshape(shared_hidden_aspect_polarity, [-1, self.config.max_sentence_size, self.config.cross_share_k, self.config.max_sentence_size])
        shared_hidden_aspect_polarity = K.transpose(shared_hidden_aspect_polarity, [0, 1, 3, 2])
        shared_hidden_aspect_polarity = K.reshape(shared_hidden_aspect_polarity, shape=[-1, self.config.max_sentence_size * self.config.max_sentence_size, self.config.cross_share_k])
        G_vector_aspect = K.tile(K.expand_dims(self.G_vector_aspect, axis=0), multiples=K.stack([K.shape(aspect_hidden)[0], 1, 1]))
        shared_hidden_aspect_polarity = K.matmul(shared_hidden_aspect_polarity, G_vector_aspect)
        aspect_vector = tf.reshape(shared_hidden_aspect_polarity, shape=[-1, self.config.max_sentence_size, self.config.max_sentence_size])

        G_polarity_aspect = K.reshape(self.G_polarity_aspect, shape=[self.config.hidden_size, -1])
        G_polarity_aspect = K.tile(K.expand_dims(G_polarity_aspect, axis=0), multiples=K.stack([K.shape(polarity_hidden)[0], 1, 1]))
        shared_hidden_polarity_aspect = K.matmul(aspect_hidden, G_polarity_aspect)
        shared_hidden_polarity_aspect = K.reshape(shared_hidden_polarity_aspect, shape=[-1, self.config.max_sentence_size * self.config.cross_share_k, self.config.hidden_size])
        aspect_hidden_transpose = K.transpose(aspect_hidden, [0, 2, 1])
        shared_hidden_polarity_aspect = K.tanh(K.matmul(shared_hidden_polarity_aspect, aspect_hidden_transpose))
        shared_hidden_polarity_aspect = K.reshape(shared_hidden_polarity_aspect, [-1, self.config.max_sentence_size, self.config.cross_share_k, self.config.max_sentence_size])
        shared_hidden_polarity_aspect = K.transpose(shared_hidden_polarity_aspect, [0, 1, 3, 2])
        shared_hidden_polarity_aspect = K.reshape(shared_hidden_polarity_aspect, shape=[-1, self.config.max_sentence_size * self.config.max_sentence_size, self.config.cross_share_k])
        G_vector_polarity = K.tile(K.expand_dims(self.G_vector_polarity, axis=0), multiples=K.stack([K.shape(polarity_hidden)[0], 1, 1]))
        shared_hidden_polarity_aspect = K.matmul(shared_hidden_polarity_aspect, G_vector_polarity)
        polarity_vector = tf.reshape(shared_hidden_polarity_aspect, shape=[-1, self.config.max_sentence_size, self.config.max_sentence_size])
        
        # Get attention vector
        aspect_attention_vector = K.nn.softmax(aspect_vector, dim=-1)
        polarity_attention_vector = K.nn.softmax(polarity_vector, dim=-1)

        aspect_hidden_v = K.matmul(aspect_attention_vector, polarity_hidden)
        polarity_hidden_v = K.matmul(polarity_attention_vector, aspect_hidden)

        aspect_hidden = aspect_hidden + aspect_hidden_v
        polarity_hidden = polarity_hidden + polarity_hidden_v

        aspect_hidden = K.reshape(aspect_hidden, shape=[-1, self.config.max_sentence_size, hidden_size])
        polarity_hidden = K.reshape(polarity_hidden, shape=[-1, self.config.max_sentence_size, hidden_size])
        
        return aspect_hidden, polarity_hidden

    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        return input_shape