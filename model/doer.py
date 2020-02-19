from tensorflow.keras import layers
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow import initializers
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
# from seqeval.metrics import classification_report, performance_measure
import tensorflow as tf
import numpy as np

from keras_contrib.layers import crf

from csu import CrossSharedUnit
from regu_cell import ReguCell

class Coextractor(object):    
    def __init__(self, config):
        self.model = None
        self.config = config
        self.feature = None

        self.general_embeddings = general_embeddings

        self.domain_embeddings = domain_embeddings

        input = layers.Input(shape=(None, config.dim_general + config.dim_domain))
        
        # first RNN layer
        if config.rnn_cell == 'regu':
            first_ate_rnn = layers.Bidirectional(ReguCell(hidden_size=config.hidden_size))(input)
            first_ate_rnn = layers.Bidirectional(ReguCell(hidden_size=config.hidden_size))(input)
        if config.rnn_cell == 'lstm':
            first_ate_rnn = layers.Bidirectional(layers.LSTM(unit=config.hidden_size,
                                                    recurrent_activation='sigmoid',
                                                    return_sequences=True,
                                                    kernel_initializer=initializers.random_uniform(-0.2, 0.2),
                                                    recurrent_initializer=initializers.random_uniform(-0.2, 0.2),
                                                    dropout=config.dropout_rate))(input)
            first_asc_rnn = layers.Bidirectional(layers.LSTM(unit=config.hidden_size,
                                                    recurrent_activation='sigmoid',
                                                    return_sequences=True,
                                                    kernel_initializer=initializers.random_uniform(-0.2, 0.2),
                                                    recurrent_initializer=initializers.random_uniform(-0.2, 0.2),
                                                    dropout=config.dropout_rate))(input)
        elif config.rnn_cell == 'gru':
            first_ate_rnn = layers.Bidirectional(layers.GRU(unit=config.hidden_size,
                                                    recurrent_activation='sigmoid',
                                                    return_sequences=True,
                                                    kernel_initializer=initializers.random_uniform(-0.2, 0.2),
                                                    recurrent_initializer=initializers.random_uniform(-0.2, 0.2),
                                                    dropout=config.dropout_rate))(input)
            first_asc_rnn = layers.Bidirectional(layers.GRU(unit=config.hidden_size,
                                                    recurrent_activation='sigmoid',
                                                    return_sequences=True,
                                                    kernel_initializer=initializers.random_uniform(-0.2, 0.2),
                                                    recurrent_initializer=initializers.random_uniform(-0.2, 0.2),
                                                    dropout=config.dropout_rate))(input)
        
        first_ate_dropout = layers.Dropout(dropout_rate)(first_ate_rnn)
        first_asc_dropout = layers.Dropout(dropout_rate)(first_asc_rnn)

        csu = CrossSharedUnit(config=self.config)(first_ate_dropout, first_asc_dropout)

        # second RNN layer
        if config.rnn_cell == 'lstm':
            second_ate_rnn = layers.Bidirectional(layers.LSTM(unit=config.hidden_size,
                                                    recurrent_activation='sigmoid',
                                                    return_sequences=True,
                                                    kernel_initializer=initializers.random_uniform(-0.2, 0.2),
                                                    recurrent_initializer=initializers.random_uniform(-0.2, 0.2),
                                                    dropout=config.dropout_rate))(input)
            second_asc_rnn = layers.Bidirectional(layers.LSTM(unit=config.hidden_size,
                                                    recurrent_activation='sigmoid',
                                                    return_sequences=True,
                                                    kernel_initializer=initializers.random_uniform(-0.2, 0.2),
                                                    recurrent_initializer=initializers.random_uniform(-0.2, 0.2),
                                                    dropout=config.dropout_rate))(input)
        elif config.rnn_cell == 'gru':
            second_ate_rnn = layers.Bidirectional(layers.GRU(unit=config.hidden_size,
                                                    recurrent_activation='sigmoid',
                                                    return_sequences=True,
                                                    kernel_initializer=initializers.random_uniform(-0.2, 0.2),
                                                    recurrent_initializer=initializers.random_uniform(-0.2, 0.2),
                                                    dropout=config.dropout_rate))(input)
            second_asc_rnn = layers.Bidirectional(layers.GRU(unit=config.hidden_size,
                                                    recurrent_activation='sigmoid',
                                                    return_sequences=True,
                                                    kernel_initializer=initializers.random_uniform(-0.2, 0.2),
                                                    recurrent_initializer=initializers.random_uniform(-0.2, 0.2),
                                                    dropout=config.dropout_rate))(input)
        
        second_ate_dropout = layers.Dropout(dropout_rate)(second_ate_rnn)
        second_asc_dropout = layers.Dropout(dropout_rate)(second_asc_rnn)

        # interface layer
        ate_crf = CRF(config.n_aspect_tags)
        asc_crf = CRF(config.n_polarity_tags)
    