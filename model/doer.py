from tensorflow.keras import layers, Model
from tensorflow.keras.models import load_model, save_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow import initializers
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
# from seqeval.metrics import classification_report, performance_measure
import numpy as np
import tensorflow.keras.backend as K

from model.csu import CrossSharedUnit
from model.regu_cell import ReguCell

class Coextractor(object):    
    def __init__(self, config):
        self.model = None
        self.config = config
        self.feature = None
    
    def init_model(self, config):
        input = layers.Input(shape=(None, config.dim_general + config.dim_domain))
        
        # first RNN layer
        if config.rnn_cell == 'regu':
            first_ate_rnn = layers.Bidirectional(ReguCell(hidden_size=config.hidden_size))(input)
            first_ate_rnn = layers.Bidirectional(ReguCell(hidden_size=config.hidden_size))(input)
        elif config.rnn_cell == 'lstm':
            first_ate_rnn = layers.Bidirectional(layers.LSTM(config.hidden_size,
                                                    recurrent_activation='sigmoid',
                                                    return_sequences=True,
                                                    kernel_initializer=initializers.RandomUniform(-0.2, 0.2),
                                                    recurrent_initializer=initializers.RandomUniform(-0.2, 0.2),
                                                    dropout=config.dropout_rate))(input)
            first_asc_rnn = layers.Bidirectional(layers.LSTM(config.hidden_size,
                                                    recurrent_activation='sigmoid',
                                                    return_sequences=True,
                                                    kernel_initializer=initializers.RandomUniform(-0.2, 0.2),
                                                    recurrent_initializer=initializers.RandomUniform(-0.2, 0.2),
                                                    dropout=config.dropout_rate))(input)
        elif config.rnn_cell == 'gru':
            first_ate_rnn = layers.Bidirectional(layers.GRU(config.hidden_size,
                                                    recurrent_activation='sigmoid',
                                                    return_sequences=True,
                                                    kernel_initializer=initializers.RandomUniform(-0.2, 0.2),
                                                    recurrent_initializer=initializers.RandomUniform(-0.2, 0.2),
                                                    dropout=config.dropout_rate))(input)
            first_asc_rnn = layers.Bidirectional(layers.GRU(config.hidden_size,
                                                    recurrent_activation='sigmoid',
                                                    return_sequences=True,
                                                    kernel_initializer=initializers.RandomUniform(-0.2, 0.2),
                                                    recurrent_initializer=initializers.RandomUniform(-0.2, 0.2),
                                                    dropout=config.dropout_rate))(input)
        
        first_ate_dropout = layers.Dropout(config.dropout_rate)(first_ate_rnn)
        first_asc_dropout = layers.Dropout(config.dropout_rate)(first_asc_rnn)

        csu = CrossSharedUnit(config=self.config)([first_ate_dropout, first_asc_dropout])
        
        def split_layer_left(input_tensor):
            return input_tensor[0]
        def split_layer_left_output_shape(shape):
            return shape[0]
        
        def split_layer_right(input_tensor):
            return input_tensor[1]
        def split_layer_right_output_shape(shape):
            return shape[1]
        
        split_ate = layers.Lambda(split_layer_left, output_shape=split_layer_left_output_shape)(csu)
        split_asc = layers.Lambda(split_layer_right, output_shape=split_layer_right_output_shape)(csu)
        
        # second RNN layer
        if config.rnn_cell == 'regu':
            first_ate_rnn = layers.Bidirectional(ReguCell(hidden_size=config.hidden_size))(split_ate)
            first_asc_rnn = layers.Bidirectional(ReguCell(hidden_size=config.hidden_size))(split_asc)
        elif config.rnn_cell == 'lstm':
            second_ate_rnn = layers.Bidirectional(layers.LSTM(config.hidden_size,
                                                    recurrent_activation='sigmoid',
                                                    return_sequences=True,
                                                    kernel_initializer=initializers.RandomUniform(-0.2, 0.2),
                                                    recurrent_initializer=initializers.RandomUniform(-0.2, 0.2),
                                                    dropout=config.dropout_rate))(split_ate)
            second_asc_rnn = layers.Bidirectional(layers.LSTM(config.hidden_size,
                                                    recurrent_activation='sigmoid',
                                                    return_sequences=True,
                                                    kernel_initializer=initializers.RandomUniform(-0.2, 0.2),
                                                    recurrent_initializer=initializers.RandomUniform(-0.2, 0.2),
                                                    dropout=config.dropout_rate))(split_asc)
        elif config.rnn_cell == 'gru':
            second_ate_rnn = layers.Bidirectional(layers.GRU(config.hidden_size,
                                                    recurrent_activation='sigmoid',
                                                    return_sequences=True,
                                                    kernel_initializer=initializers.RandomUniform(-0.2, 0.2),
                                                    recurrent_initializer=initializers.RandomUniform(-0.2, 0.2),
                                                    dropout=config.dropout_rate))(split_ate)
            second_asc_rnn = layers.Bidirectional(layers.GRU(config.hidden_size,
                                                    recurrent_activation='sigmoid',
                                                    return_sequences=True,
                                                    kernel_initializer=initializers.RandomUniform(-0.2, 0.2),
                                                    recurrent_initializer=initializers.RandomUniform(-0.2, 0.2),
                                                    dropout=config.dropout_rate))(split_asc)
        
        second_ate_dropout = layers.Dropout(config.dropout_rate)(second_ate_rnn)
        second_asc_dropout = layers.Dropout(config.dropout_rate)(second_asc_rnn)

        # interface layer
        # ate_crf = CRF(config.n_aspect_tags)(second_ate_dropout)
        # asc_crf = CRF(config.n_polarity_tags)(second_asc_dropout)
        ate_dense = layers.Dense(5, activation='softmax')(second_ate_dropout)
        asc_dense = layers.Dense(5, activation='softmax')(second_asc_dropout)

        self.model = Model(inputs=input, outputs=[ate_dense, asc_dense])
        self.model.compile(optimizer='nadam',
        loss='categorical_crossentropy',
        metrics=['accuracy'])
    
    def train(self, X_train, y_train, config):
        es = EarlyStopping(monitor='loss', mode='min', patience=1)
#         mc = ModelCheckpoint(None, monitor='loss', mode='min', save_best_only=True)
        self.model.fit(X_train, y_train, batch_size=config.batch_size, callbacks=[es])

    def predict(self, X):
        y = []
        for i in range(len(X)):
            self.config.max_sentence_size = X[i].shape[1]
            yate_score, yasc_score = self.model.predict(np.asarray(X[i]), batch_size=None)
            K.clear_session()
            # Get the label index with the highest probability
            yate_pred = np.argmax(yate_score, 2)
            yasc_pred = np.argmax(yasc_score, 2)

            y1 = []
            y2 = []
            for j in range(len(yate_pred[0])):
                if yate_pred[0][j] == 0:
                    y1.append('O')
                elif yate_pred[0][j] == 1:
                    y1.append('B-ASPECT')
                elif yate_pred[0][j] == 2:
                    y1.append('I-ASPECT')
                elif yate_pred[0][j] == 3:
                    y1.append('B-SENTIMENT')
                elif yate_pred[0][j] == 4:
                    y1.append('I-SENTIMENT')

                if yasc_pred[0][j] == 0:
                    y2.append('O')
                elif yasc_pred[0][j] == 1:
                    y2.append('PO')
                elif yasc_pred[0][j] == 2:
                    y2.append('NG')
                elif yasc_pred[0][j] == 3:
                    y2.append('NT')
                elif yasc_pred[0][j] == 4:
                    y2.append('CF')

            y.append([y1, y2])
        return y
            
    def save(self, filename):
        self.model.save_weights(filename, save_format='tf')


    def load(self, filename, X_train, y_train):
        self.init_model(self.config)
        self.model.train_on_batch(X_train[:1], [y_train[0][:1], y_train[1][:1]])
        self.model.load_weights(filename)

        return self
    
    def evaluate(self, X, y):
        y_pred = self.predict(X)
        y_true_ate = []
        y_pred_ate = []
        y_true_asc = []
        y_pred_asc = []

        for seq in y:
            for i in range(len(seq[0])):
                if seq[0][i] == 'O':
                    y_true_ate.append(0)
                elif seq[0][i] == 'B-ASPECT':
                    y_true_ate.append(1)
                elif seq[0][i] == 'I-ASPECT':
                    y_true_ate.append(2)
                elif seq[0][i] == 'B-SENTIMENT':
                    y_true_ate.append(3)
                elif seq[0][i] == 'I-SENTIMENT':
                    y_true_ate.append(4)
                    
                if seq[1][i] == 'O':
                    y_true_asc.append(0)
                elif seq[1][i] == 'PO':
                    y_true_asc.append(1)
                elif seq[1][i] == 'NG':
                    y_true_asc.append(2)
                elif seq[1][i] == 'NT':
                    y_true_asc.append(3)
                elif seq[1][i] == 'CF':
                    y_true_asc.append(4)
        
        for seq in y_pred:
            for i in range(len(seq[0])):
                if seq[0][i] == 'O':
                    y_pred_ate.append(0)
                elif seq[0][i] == 'B-ASPECT':
                    y_pred_ate.append(1)
                elif seq[0][i] == 'I-ASPECT':
                    y_pred_ate.append(2)
                elif seq[0][i] == 'B-SENTIMENT':
                    y_pred_ate.append(3)
                elif seq[0][i] == 'I-SENTIMENT':
                    y_pred_ate.append(4)
                    
                if seq[1][i] == 'O':
                    y_pred_asc.append(0)
                elif seq[1][i] == 'PO':
                    y_pred_asc.append(1)
                elif seq[1][i] == 'NG':
                    y_pred_asc.append(2)
                elif seq[1][i] == 'NT':
                    y_pred_asc.append(3)
                elif seq[1][i] == 'CF':
                    y_pred_asc.append(4)
        
        self.print_evaluations("Aspect and Sentiment Term Extraction", y_true_ate, y_pred_ate)
        self.print_evaluations("Aspect Sentiment Classification", y_true_asc, y_pred_asc)
    
    def print_evaluations(self, task_name, y_true, y_pred):
        print(task_name)
        print("Confusion Matrix:")
        print(confusion_matrix(y_true, y_pred))
        print()
        print("Precision:")
        print("weighted : ", precision_score(y_true, y_pred, average='weighted'))
        print("average : ", precision_score(y_true, y_pred, average='macro'))
        print()
        print("Recall:")
        print("weighted : ", recall_score(y_true, y_pred, average='weighted'))
        print("macro : ", recall_score(y_true, y_pred, average='macro'))
        print()
        print("F1-score:")
        print("weighted : ", f1_score(y_true, y_pred, average='weighted'))
        print("macro : ", f1_score(y_true, y_pred, average='macro'))
        print()