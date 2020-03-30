from tensorflow.keras import layers, Model
from tensorflow.keras.models import load_model, save_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow import initializers
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from seqeval.metrics import classification_report, performance_measure
import numpy as np
import tensorflow.keras.backend as K
import tensorflow as tf

from model.csu import CrossSharedUnit
from model.regu_cell import ReguCell

class Coextractor(object):    
    def __init__(self, config):
        self.model = None
        self.config = config
        self.feature = None
    
    def init_model(self):
        input_shape = self.config.dim_general + self.config.dim_domain
        input = layers.Input(shape=(None, input_shape))
        
        # first RNN layerevaluate
        if self.config.rnn_cell == 'regu':
            first_ate_rnn = layers.Bidirectional(ReguCell(hidden_size=self.config.hidden_size, 
                                                    return_sequences=True))(input, [tf.zeros([input_shape, 2 * self.config.hidden_size]) for i in range(2)])
            first_ate_rnn = layers.Bidirectional(ReguCell(hidden_size=self.config.hidden_size,
                                                    return_sequences=True))(input, [tf.zeros([input_shape, 2 * self.config.hidden_size]) for i in range(2)])
        elif self.config.rnn_cell == 'lstm':
            first_ate_rnn = layers.Bidirectional(layers.LSTM(self.config.hidden_size,
                                                    recurrent_activation='sigmoid',
                                                    return_sequences=True,
                                                    kernel_initializer=initializers.RandomUniform(-0.2, 0.2),
                                                    recurrent_initializer=initializers.RandomUniform(-0.2, 0.2),
                                                    dropout=self.config.dropout_rate))(input)
            first_asc_rnn = layers.Bidirectional(layers.LSTM(self.config.hidden_size,
                                                    recurrent_activation='sigmoid',
                                                    return_sequences=True,
                                                    kernel_initializer=initializers.RandomUniform(-0.2, 0.2),
                                                    recurrent_initializer=initializers.RandomUniform(-0.2, 0.2),
                                                    dropout=self.config.dropout_rate))(input)
        elif self.config.rnn_cell == 'gru':
            first_ate_rnn = layers.Bidirectional(layers.GRU(self.config.hidden_size,
                                                    recurrent_activation='sigmoid',
                                                    return_sequences=True,
                                                    kernel_initializer=initializers.RandomUniform(-0.2, 0.2),
                                                    recurrent_initializer=initializers.RandomUniform(-0.2, 0.2),
                                                    dropout=self.config.dropout_rate))(input)
            first_asc_rnn = layers.Bidirectional(layers.GRU(self.config.hidden_size,
                                                    recurrent_activation='sigmoid',
                                                    return_sequences=True,
                                                    kernel_initializer=initializers.RandomUniform(-0.2, 0.2),
                                                    recurrent_initializer=initializers.RandomUniform(-0.2, 0.2),
                                                    dropout=self.config.dropout_rate))(input)
        
        first_ate_dropout = layers.Dropout(self.config.dropout_rate)(first_ate_rnn)
        first_asc_dropout = layers.Dropout(self.config.dropout_rate)(first_asc_rnn)

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
        if self.config.rnn_cell == 'regu':
            first_ate_rnn = layers.Bidirectional(ReguCell(hidden_size=self.config.hidden_size))(split_ate)
            first_asc_rnn = layers.Bidirectional(ReguCell(hidden_size=self.config.hidden_size))(split_asc)
        elif self.config.rnn_cell == 'lstm':
            second_ate_rnn = layers.Bidirectional(layers.LSTM(self.config.hidden_size,
                                                    recurrent_activation='sigmoid',
                                                    return_sequences=True,
                                                    kernel_initializer=initializers.RandomUniform(-0.2, 0.2),
                                                    recurrent_initializer=initializers.RandomUniform(-0.2, 0.2),
                                                    dropout=self.config.dropout_rate))(split_ate)
            second_asc_rnn = layers.Bidirectional(layers.LSTM(self.config.hidden_size,
                                                    recurrent_activation='sigmoid',
                                                    return_sequences=True,
                                                    kernel_initializer=initializers.RandomUniform(-0.2, 0.2),
                                                    recurrent_initializer=initializers.RandomUniform(-0.2, 0.2),
                                                    dropout=self.config.dropout_rate))(split_asc)
        elif self.config.rnn_cell == 'gru':
            second_ate_rnn = layers.Bidirectional(layers.GRU(self.config.hidden_size,
                                                    recurrent_activation='sigmoid',
                                                    return_sequences=True,
                                                    kernel_initializer=initializers.RandomUniform(-0.2, 0.2),
                                                    recurrent_initializer=initializers.RandomUniform(-0.2, 0.2),
                                                    dropout=self.config.dropout_rate))(split_ate)
            second_asc_rnn = layers.Bidirectional(layers.GRU(self.config.hidden_size,
                                                    recurrent_activation='sigmoid',
                                                    return_sequences=True,
                                                    kernel_initializer=initializers.RandomUniform(-0.2, 0.2),
                                                    recurrent_initializer=initializers.RandomUniform(-0.2, 0.2),
                                                    dropout=self.config.dropout_rate))(split_asc)
        
        second_ate_dropout = layers.Dropout(self.config.dropout_rate)(second_ate_rnn)
        second_asc_dropout = layers.Dropout(self.config.dropout_rate)(second_asc_rnn)

        # interface layer
        # ate_crf = CRF(self.config.n_aspect_tags)(second_ate_dropout)
        # asc_crf = CRF(self.config.n_polarity_tags)(second_asc_dropout)
        ate_dense = layers.Dense(5, activation='softmax')(second_ate_dropout)
        asc_dense = layers.Dense(5, activation='softmax')(second_asc_dropout)

        self.model = Model(inputs=input, outputs=[ate_dense, asc_dense])
        self.model.compile(optimizer='nadam',
        loss='categorical_crossentropy',
        metrics=['accuracy'])
    
    def train(self, X_train, y_train):
        es = EarlyStopping(monitor='loss', mode='min', patience=self.config.patience)
        mc = ModelCheckpoint('/output/model_doer', monitor='loss', mode='min', save_best_only=True, save_weights_only=True)
        self.model.fit(X_train, y_train,
                       batch_size=self.config.batch_size,
                       epochs=self.config.epoch,
                       verbose=self.config.verbose,
                       callbacks=[es])

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
        self.init_model()
        self.model.train_on_batch(X_train[:1], [y_train[0][:1], y_train[1][:1]])
        self.model.load_weights(filename)

        return self
    
    def evaluate(self, X, y, sentences):
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
#         self.print_report(y[0], y_pred[0])
#         if sentences != None:
#             self.get_wrong_predictions(y[0], y_pred[0], sentences)
            
        self.print_evaluations("Aspect Sentiment Classification", y_true_asc, y_pred_asc)
        print(y[0])
        print(y[1])
#         self.print_report(y[1], y_pred[1])
#         if sentences != None:
#             self.get_wrong_predictions(y[1], y_pred[1], sentences)

    
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
    
    def print_report(self, y_true, y_pred):
        print(classification_report(y_true, y_pred))
        print(performance_measure(y_true, y_pred))
        
        
    def get_wrong_predictions(self, y, y_pred, sentences):
        count = 0
        for idx in range(len(y)):
            wrong = False
            for idx2 in range(len(y[idx])):
                if y[idx][idx2] != y_pred[idx][idx2]:
                    if not(wrong):
                        print("")
                        print('sentence:', " ".join(sentences[idx]))
                        print('labels:', " ".join(y[idx]))
                    wrong = True
                    count += 1
                    print(sentences[idx][idx2], '\t| P:', y_pred[idx][idx2], '\t| A:', y[idx][idx2])
        print(count, 'words misclasified')