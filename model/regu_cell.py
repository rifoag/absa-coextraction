import tensorflow as K
from tensorflow.keras.layers import Layer
import tensorflow.keras

class ReguCell(Layer):
    def __init__(self, hidden_size, go_backwards=True, return_sequences=True, return_state=True, **kwargs):
        self.hidden_size = hidden_size
        self.go_backwards = go_backwards
        self.state_size = hidden_size
        self.output_size = hidden_size
        self.return_sequences= return_sequences
        self.return_state = return_state

        super(ReguCell, self).__init__(**kwargs)
        
    def build(self, input_shape):        
        self.wufo = self.add_weight(name='forget_residual_gate_weights',
                                  shape=(input_shape[-1]+self.state_size, 2*self.hidden_size),
                                  initializer=tensorflow.keras.initializers.GlorotUniform(),
                                  trainable=True)
        self.bfo = self.add_weight(name='forget_residual_gate_biases',
                                  shape=(2*self.hidden_size),
                                  initializer=K.constant_initializer(0.0),
                                  trainable=True)
        self.wi = self.add_weight(name='cell_memory_weight',
                                  shape=(input_shape[-1], self.hidden_size),
                                  initializer=tensorflow.keras.initializers.GlorotUniform(),
                                  trainable=True)
        self.bi = self.add_weight(name='cell_memory_bias',
                                  shape=(self.hidden_size),
                                  initializer=K.constant_initializer(0.0),
                                  trainable=True)
        self.wx = self.add_weight(name='input_projection_weight',
                                  shape=(input_shape[-1], self.hidden_size),
                                  initializer=tensorflow.keras.initializers.GlorotUniform(),
                                  trainable=True)
        self.built = True
        super(ReguCell, self).build(input_shape)  # Be sure to call this at the end

    def call(self, inputs, states):
        c_prev = states[0]

        regu_matrix = K.matmul(K.concat([inputs, c_prev], 1), self.wufo)
        regu_matrix = K.nn.bias_add(regu_matrix, self.bfo)
        f, o = K.split(value=regu_matrix, num_or_size_splits=2, axis=1)

        input_size = inputs.get_shape().as_list()[-1]
        if input_size == self.hidden_size:
            x_proj = inputs
        else:
            x_proj = K.matmul(inputs, self.wx)
            x_proj = K.math.tanh(x_proj)
        
        c_ = K.matmul(inputs, self.wi)
        c_ = K.nn.bias_add(c_, self.bi)
        c_ = K.math.tanh(c_)

        c = (1-K.math.sigmoid(f)) * c_prev + K.math.sigmoid(f)*c_
        m = (1-K.math.sigmoid(o)) * c + K.math.sigmoid(o)*x_proj

        return m, [c]
    
    def get_config(self):
        config = { 
            'hidden_size': self.hidden_size,
            'go_backwards': True
        }

        base_config = super(ReguCell, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
