from keras import backend as K
from keras.layers import Layer

class ReguCell(Layer):
    def __init__(self, hidden_size, **kwargs):
        self.hidden_size = hidden_size

        super(ReguCell, self).__init__(**kwargs)

    def build(self, input_shape):        
        self.wufo = self.add_weight(name='forget_residual_gate_weights',
                                  shape=(input_shape[1], 2*self.hidden_size),
                                  initializer=K.contrib.layers.xavier_initializer(),
                                  trainable=True)
        self.bfo = self.add_weight(name='forget_residual_gate_biases',
                                  shape=(2*self.hidden_size),
                                  initializer=K.constant_initializer(0.0),
                                  trainable=True)
        self.wi = self.add_weight(name='cell_memory_weight',
                                  shape=(input_shape[1], self.hidden_size),
                                  initializer=K.contrib.layers.xavier_initializer(),
                                  trainable=True)
        self.bi = self.add_weight(name='cell_memory_bias',
                                  shape=(self.hidden_size),
                                  initializer=K.constant_initializer(0.0),
                                  trainable=True)
        self.wx = self.add_weight(name='input_projection_weight',
                                  shape=(input_shape[1], self.hidden_size),
                                  initializer=K.contrib.layers.xavier_initializer(),
                                  trainable=True)
        super(ReguCell, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x, state):
        sigmoid = K.math.sigmoid
        (c_prev, _) = state

        regu_matrix = K.matmul(K.concat([x, c_prev], 1), self.wufo)
        regu_matrix = K.nn.bias_add(regu_matrix, self.bfo)
        f, o = K.split(value=regu_matrix, num_or_size_splits=2, axis=1)

        input_size = x.get_shape().as_list()[-1]
        if input_size == self.hidden_size:
            x_proj = x
        else:
            x_proj = K.matmul(x, self.wx)
            x_proj = K.math.tanh(x_proj)
        
        c_ = K.matmul(x, self.wi)
        c_ = K.nn.bias_add(c_, self.bi)
        c_ = K.math.tanh(c_)

        c = (1-sigmoid(f)) * c_prev + sigmoid(f)*c_
        m = (1-sigmoid(o)) * c + sigmoid(o)*x_proj

        new_state = K.contrib.rnn.LSTMStateTuple(c, m)
        return m, new_state
