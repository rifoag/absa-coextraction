from tensorflow.keras import layers
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow import initializers
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
# from seqeval.metrics import classification_report, performance_measure
import tensorflow as tf
import numpy as np

from model.csu import CrossSharedUnit
from model.regu_cell import ReguCell

class Coextractor(object):    
    def __init__(self, config):
        self.model = None
        self.config = config
        self.feature = None

        input = layers.Input(shape=(None, config.dim_general + config.dim_domain))
        
        # first RNN layer
        if config.rnn_cell == 'regu':
            first_ate_rnn = layers.Bidirectional(ReguCell(hidden_size=config.hidden_size))(input)
            first_ate_rnn = layers.Bidirectional(ReguCell(hidden_size=config.hidden_size))(input)
        if config.rnn_cell == 'lstm':
            first_ate_rnn = layers.Bidirectional(layers.LSTM(units=config.hidden_size,
                                                    recurrent_activation='sigmoid',
                                                    return_sequences=True,
                                                    kernel_initializer=initializers.RandomUniform(-0.2, 0.2),
                                                    recurrent_initializer=initializers.RandomUniform(-0.2, 0.2),
                                                    dropout=config.dropout_rate))(input)
            first_asc_rnn = layers.Bidirectional(layers.LSTM(units=config.hidden_size,
                                                    recurrent_activation='sigmoid',
                                                    return_sequences=True,
                                                    kernel_initializer=initializers.RandomUniform(-0.2, 0.2),
                                                    recurrent_initializer=initializers.RandomUniform(-0.2, 0.2),
                                                    dropout=config.dropout_rate))(input)
        elif config.rnn_cell == 'gru':
            first_ate_rnn = layers.Bidirectional(layers.GRU(units=config.hidden_size,
                                                    recurrent_activation='sigmoid',
                                                    return_sequences=True,
                                                    kernel_initializer=initializers.RandomUniform(-0.2, 0.2),
                                                    recurrent_initializer=initializers.RandomUniform(-0.2, 0.2),
                                                    dropout=config.dropout_rate))(input)
            first_asc_rnn = layers.Bidirectional(layers.GRU(units=config.hidden_size,
                                                    recurrent_activation='sigmoid',
                                                    return_sequences=True,
                                                    kernel_initializer=initializers.RandomUniform(-0.2, 0.2),
                                                    recurrent_initializer=initializers.RandomUniform(-0.2, 0.2),
                                                    dropout=config.dropout_rate))(input)
        
        first_ate_dropout = layers.Dropout(config.dropout_rate)(first_ate_rnn)
        first_asc_dropout = layers.Dropout(config.dropout_rate)(first_asc_rnn)

#         csu = CrossSharedUnit(config=self.config)([first_ate_dropout, first_asc_dropout])

        # second RNN layer
        if config.rnn_cell == 'lstm':
            second_ate_rnn = layers.Bidirectional(layers.LSTM(units=config.hidden_size,
                                                    recurrent_activation='sigmoid',
                                                    return_sequences=True,
                                                    kernel_initializer=initializers.RandomUniform(-0.2, 0.2),
                                                    recurrent_initializer=initializers.RandomUniform(-0.2, 0.2),
                                                    dropout=config.dropout_rate))(input)
            second_asc_rnn = layers.Bidirectional(layers.LSTM(units=config.hidden_size,
                                                    recurrent_activation='sigmoid',
                                                    return_sequences=True,
                                                    kernel_initializer=initializers.RandomUniform(-0.2, 0.2),
                                                    recurrent_initializer=initializers.RandomUniform(-0.2, 0.2),
                                                    dropout=config.dropout_rate))(input)
        elif config.rnn_cell == 'gru':
            second_ate_rnn = layers.Bidirectional(layers.GRU(units=config.hidden_size,
                                                    recurrent_activation='sigmoid',
                                                    return_sequences=True,
                                                    kernel_initializer=initializers.RandomUniform(-0.2, 0.2),
                                                    recurrent_initializer=initializers.RandomUniform(-0.2, 0.2),
                                                    dropout=config.dropout_rate))(input)
            second_asc_rnn = layers.Bidirectional(layers.GRU(units=config.hidden_size,
                                                    recurrent_activation='sigmoid',
                                                    return_sequences=True,
                                                    kernel_initializer=initializers.RandomUniform(-0.2, 0.2),
                                                    recurrent_initializer=initializers.RandomUniform(-0.2, 0.2),
                                                    dropout=config.dropout_rate))(input)
        
        second_ate_dropout = layers.Dropout(config.dropout_rate)(second_ate_rnn)
        second_asc_dropout = layers.Dropout(config.dropout_rate)(second_asc_rnn)

        # interface layer
        # ate_crf = CRF(config.n_aspect_tags)(second_ate_dropout)
        # asc_crf = CRF(config.n_polarity_tags)(second_asc_dropout)
        ate_dense = layers.Dense(config.hidden_size, activation='softmax')(second_ate_dropout)
        asc_dense = layers.Dense(config.hidden_size, activation='softmax')(second_asc_dropout)

        self.model = tf.keras.Model(inputs=input, outputs=[ate_dense, asc_dense])
        self.model.compile(optimizer='nadam',
        loss='categorical_crossentropy',
        metrics=['accuracy'])
    