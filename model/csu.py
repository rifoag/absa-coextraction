from tensorflow.keras.layers import Layer
import numpy as np
from tensorflow import Variable, float32, reshape, tile, expand_dims, stack, shape, matmul, transpose, tanh, nn

class CrossSharedUnit(Layer):
    def __init__(self, config, **kwargs):
        self.config = config
        super(CrossSharedUnit, self).__init__(**kwargs)

    def build(self, input_shape):
        assert isinstance(input_shape, list)
        self.cross_share_k, self.g_hidden_size = self.config.cross_share_k, 2 * self.config.hidden_size
        init_narray = np.random.randn(self.g_hidden_size, self.cross_share_k, self.g_hidden_size)
        self.G_aspect_polarity = Variable(init_narray, name="G_aspect_polarity", dtype=float32)
        init_narray = np.random.randn(self.g_hidden_size, self.cross_share_k, self.g_hidden_size)
        self.G_polarity_aspect = Variable(init_narray, name="G_polarity_aspect", dtype=float32)

        init_narray = np.random.randn(self.cross_share_k, 1)
        self.G_vector_aspect = Variable(init_narray, name="G_vector_aspect", dtype=float32)
        init_narray = np.random.randn(self.cross_share_k, 1)
        self.G_vector_polarity = Variable(init_narray, name="G_vector_polarity", dtype=float32)
        super(CrossSharedUnit, self).build(input_shape)

    def call(self, x):     
        assert isinstance(x, list)
        aspect_hidden, polarity_hidden = x
        
        G_aspect_polarity = self.G_aspect_polarity
        G_polarity_aspect = self.G_polarity_aspect
        G_vector_aspect = self.G_vector_aspect
        G_vector_polarity = self.G_vector_polarity
        
        G_aspect_polarity = reshape(G_aspect_polarity, shape=[self.g_hidden_size, -1])
        G_aspect_polarity = tile(expand_dims(G_aspect_polarity, axis=0), multiples=stack([shape(aspect_hidden)[0], 1, 1]))
        shared_hidden_aspect_polarity = matmul(aspect_hidden, G_aspect_polarity)
        shared_hidden_aspect_polarity = reshape(shared_hidden_aspect_polarity, shape=[-1, self.config.max_sentence_size * self.cross_share_k, self.g_hidden_size])
        polarity_hidden_transpose = transpose(polarity_hidden, [0, 2, 1])
        shared_hidden_aspect_polarity = tanh(matmul(shared_hidden_aspect_polarity, polarity_hidden_transpose))
        shared_hidden_aspect_polarity = reshape(shared_hidden_aspect_polarity, [-1, self.config.max_sentence_size, self.cross_share_k, self.config.max_sentence_size])
        shared_hidden_aspect_polarity = transpose(shared_hidden_aspect_polarity, [0, 1, 3, 2])
        shared_hidden_aspect_polarity = reshape(shared_hidden_aspect_polarity, shape=[-1, self.config.max_sentence_size * self.config.max_sentence_size, self.cross_share_k])
        G_vector_aspect = tile(expand_dims(G_vector_aspect, axis=0), multiples=stack([shape(aspect_hidden)[0], 1, 1]))
        shared_hidden_aspect_polarity = matmul(shared_hidden_aspect_polarity, G_vector_aspect)
        aspect_vector = reshape(shared_hidden_aspect_polarity, shape=[-1, self.config.max_sentence_size, self.config.max_sentence_size])

        G_polarity_aspect = reshape(G_polarity_aspect, shape=[self.g_hidden_size, -1])
        G_polarity_aspect = tile(expand_dims(G_polarity_aspect, axis=0), multiples=stack([shape(polarity_hidden)[0], 1, 1]))
        shared_hidden_polarity_aspect = matmul(aspect_hidden, G_polarity_aspect)
        shared_hidden_polarity_aspect = reshape(shared_hidden_polarity_aspect, shape=[-1, self.config.max_sentence_size * self.config.cross_share_k, self.g_hidden_size])
        aspect_hidden_transpose = transpose(aspect_hidden, [0, 2, 1])
        shared_hidden_polarity_aspect = tanh(matmul(shared_hidden_polarity_aspect, aspect_hidden_transpose))
        shared_hidden_polarity_aspect = reshape(shared_hidden_polarity_aspect, [-1, self.config.max_sentence_size, self.config.cross_share_k, self.config.max_sentence_size])
        shared_hidden_polarity_aspect = transpose(shared_hidden_polarity_aspect, [0, 1, 3, 2])
        shared_hidden_polarity_aspect = reshape(shared_hidden_polarity_aspect, shape=[-1, self.config.max_sentence_size * self.config.max_sentence_size, self.config.cross_share_k])
        G_vector_polarity = tile(expand_dims(G_vector_polarity, axis=0), multiples=stack([shape(polarity_hidden)[0], 1, 1]))
        shared_hidden_polarity_aspect = matmul(shared_hidden_polarity_aspect, G_vector_polarity)
        polarity_vector = reshape(shared_hidden_polarity_aspect, shape=[-1, self.config.max_sentence_size, self.config.max_sentence_size])
        
        # Get attention vector
        aspect_attention_vector = nn.softmax(aspect_vector)
        polarity_attention_vector = nn.softmax(polarity_vector)

        aspect_hidden_v = matmul(aspect_attention_vector, polarity_hidden)
        polarity_hidden_v = matmul(polarity_attention_vector, aspect_hidden)

        aspect_hidden = aspect_hidden + aspect_hidden_v
        polarity_hidden = polarity_hidden + polarity_hidden_v

        aspect_hidden = reshape(aspect_hidden, shape=[-1, self.config.max_sentence_size, self.g_hidden_size])
        polarity_hidden = reshape(polarity_hidden, shape=[-1, self.config.max_sentence_size, self.g_hidden_size])
        return [aspect_hidden, polarity_hidden]

    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        return input_shape