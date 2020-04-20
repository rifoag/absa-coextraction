from tensorflow.keras import layers, Model
from tensorflow.keras.models import load_model, save_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow import initializers
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, classification_report
# from seqeval.metrics import classification_report, performance_measure
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
        input = layers.Input(shape=(None, input_shape), name="input")

        # first RNN layer
        if self.config.rnn_cell == 'regu':
            first_ate_rnn = layers.Bidirectional(layers.RNN(ReguCell(hidden_size=self.config.hidden_size, return_sequences=True),
                      return_sequences=True), name="first_ate_rnn")(input, [tf.zeros([self.config.batch_size, self.config.hidden_size]) for i in range(2)])
            first_asc_rnn = layers.Bidirectional(layers.RNN(ReguCell(hidden_size=self.config.hidden_size, return_sequences=True),
                      return_sequences=True), name="first_asc_rnn")(input, [tf.zeros([self.config.batch_size, self.config.hidden_size]) for i in range(2)])
        elif self.config.rnn_cell == 'lstm':
            first_ate_rnn = layers.Bidirectional(layers.LSTM(self.config.hidden_size,
                                                    recurrent_activation='sigmoid',
                                                    return_sequences=True,
                                                    kernel_initializer=initializers.RandomUniform(-0.2, 0.2),
                                                    recurrent_initializer=initializers.RandomUniform(-0.2, 0.2),
                                                    dropout=self.config.dropout_rate), name="first_ate_rnn")(input)
            first_asc_rnn = layers.Bidirectional(layers.LSTM(self.config.hidden_size,
                                                    recurrent_activation='sigmoid',
                                                    return_sequences=True,
                                                    kernel_initializer=initializers.RandomUniform(-0.2, 0.2),
                                                    recurrent_initializer=initializers.RandomUniform(-0.2, 0.2),
                                                    dropout=self.config.dropout_rate), name="first_asc_rnn")(input)
        elif self.config.rnn_cell == 'gru':
            first_ate_rnn = layers.Bidirectional(layers.GRU(self.config.hidden_size,
                                                    recurrent_activation='sigmoid',
                                                    return_sequences=True,
                                                    kernel_initializer=initializers.RandomUniform(-0.2, 0.2),
                                                    recurrent_initializer=initializers.RandomUniform(-0.2, 0.2),
                                                    dropout=self.config.dropout_rate), name="first_ate_rnn")(input)
            first_asc_rnn = layers.Bidirectional(layers.GRU(self.config.hidden_size,
                                                    recurrent_activation='sigmoid',
                                                    return_sequences=True,
                                                    kernel_initializer=initializers.RandomUniform(-0.2, 0.2),
                                                    recurrent_initializer=initializers.RandomUniform(-0.2, 0.2),
                                                    dropout=self.config.dropout_rate), name="first_asc_rnn")(input)
        
        first_ate_dropout = layers.Dropout(self.config.dropout_rate, name="first_ate_dropout")(first_ate_rnn)
        first_asc_dropout = layers.Dropout(self.config.dropout_rate, name="first_asc_dropout")(first_asc_rnn)
        
        def max_pooling_layer(input_tensor):
            return tf.reduce_max(input_tensor, axis=-2)
        def max_pooling_layer_output_shape(shape):
            return (shape[0], shape[2])
        
        # auxiliary tasks
        sentiment_lexicon_enhancement = layers.Dense(3, activation='softmax', name="sentiment_lexicon_enhancement")(first_asc_dropout)
        
        max_pool_ate = layers.Lambda(max_pooling_layer, output_shape=max_pooling_layer_output_shape, name="max_pool_ate")(first_ate_rnn)
        aspect_term_length_enhancement = layers.Dense(1, activation='sigmoid', name="aspect_term_length_enhancement")(max_pool_ate)
        
        max_pool_asc = layers.Lambda(max_pooling_layer, output_shape=max_pooling_layer_output_shape, name="max_pool_asc")(first_asc_rnn)
        aspect_polarity_length_enhancement = layers.Dense(1, activation='sigmoid', name="aspect_polarity_length_enhancement")(max_pool_asc)
        
        csu = CrossSharedUnit(config=self.config, name="cross_shared_unit")([first_ate_dropout, first_asc_dropout])
        
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
            second_ate_rnn = layers.Bidirectional(layers.RNN(ReguCell(hidden_size=self.config.hidden_size, return_sequences=True), return_sequences=True), name="second_ate_rnn")(split_ate, [tf.zeros([self.config.batch_size, self.config.hidden_size]) for i in range(2)])
            second_asc_rnn = layers.Bidirectional(layers.RNN(ReguCell(hidden_size=self.config.hidden_size, return_sequences=True), return_sequences=True), name="second_asc_rnn")(split_asc, [tf.zeros([self.config.batch_size, self.config.hidden_size]) for i in range(2)])
        elif self.config.rnn_cell == 'lstm':
            second_ate_rnn = layers.Bidirectional(layers.LSTM(self.config.hidden_size,
                                                    recurrent_activation='sigmoid',
                                                    return_sequences=True,
                                                    kernel_initializer=initializers.RandomUniform(-0.2, 0.2),
                                                    recurrent_initializer=initializers.RandomUniform(-0.2, 0.2),
                                                    dropout=self.config.dropout_rate), name="second_ate_rnn")(split_ate)
            second_asc_rnn = layers.Bidirectional(layers.LSTM(self.config.hidden_size,
                                                    recurrent_activation='sigmoid',
                                                    return_sequences=True,
                                                    kernel_initializer=initializers.RandomUniform(-0.2, 0.2),
                                                    recurrent_initializer=initializers.RandomUniform(-0.2, 0.2),
                                                    dropout=self.config.dropout_rate), name="second_asc_rnn")(split_asc)
        elif self.config.rnn_cell == 'gru':
            second_ate_rnn = layers.Bidirectional(layers.GRU(self.config.hidden_size,
                                                    recurrent_activation='sigmoid',
                                                    return_sequences=True,
                                                    kernel_initializer=initializers.RandomUniform(-0.2, 0.2),
                                                    recurrent_initializer=initializers.RandomUniform(-0.2, 0.2),
                                                    dropout=self.config.dropout_rate), name="second_ate_rnn")(split_ate)
            second_asc_rnn = layers.Bidirectional(layers.GRU(self.config.hidden_size,
                                                    recurrent_activation='sigmoid',
                                                    return_sequences=True,
                                                    kernel_initializer=initializers.RandomUniform(-0.2, 0.2),
                                                    recurrent_initializer=initializers.RandomUniform(-0.2, 0.2),
                                                    dropout=self.config.dropout_rate), name="second_asc_rnn")(split_asc)
        
        second_ate_dropout = layers.Dropout(self.config.dropout_rate, name="second_ate_dropout")(second_ate_rnn)
        second_asc_dropout = layers.Dropout(self.config.dropout_rate, name="second_asc_dropout")(second_asc_rnn)

        # interface layer
        # ate_crf = CRF(self.config.n_aspect_tags)(second_ate_dropout)
        # asc_crf = CRF(self.config.n_polarity_tags)(second_asc_dropout)
        ate_dense = layers.Dense(5, activation='softmax', name="ate_output")(second_ate_dropout)
        asc_dense = layers.Dense(3, activation='softmax', name="asc_output")(second_asc_dropout)
        
        losses = {
            'ate_output': 'categorical_crossentropy',
            'asc_output': 'categorical_crossentropy',
            'sentiment_lexicon_enhancement': 'categorical_crossentropy',
            'aspect_term_length_enhancement': 'mean_squared_error',
            'aspect_polarity_length_enhancement': 'mean_squared_error'
        }
        
        self.model = Model(inputs=input, outputs=[ate_dense, asc_dense, sentiment_lexicon_enhancement, aspect_term_length_enhancement, aspect_polarity_length_enhancement])
        self.model.compile(optimizer='nadam',
        loss=losses,
        metrics=['accuracy'])
    
    def train(self, X_train, y_train, X_val, y_val):
        es = EarlyStopping(monitor='loss', mode='min', patience=self.config.patience)
        mc = ModelCheckpoint('/output/model_doer', monitor='loss', mode='min', save_best_only=True, save_weights_only=True)
        self.model.fit(X_train, y_train, validation_data=(X_val, y_val),
                       batch_size=self.config.batch_size,
                       epochs=self.config.epoch,
                       verbose=self.config.verbose,
                       callbacks=[es])

    def predict(self, X, y_true):
        y = []
        yate_scores, yasc_scores, lexicon_enhancement_scores, aspect_term_length_enhancement_scores, aspect_polarity_length_enhancement_scores = self.model.predict(np.asarray(X), batch_size=self.config.batch_size)
        
        for i in range(len(X)):
            yate_pred = np.argmax(yate_scores[i], 1)
            yasc_pred = np.argmax(yasc_scores[i], 1)
            
            y1 = []
            y2 = []
            max_iter = min(len(y_true[i][0]), self.config.max_sentence_size)
            for j in range(max_iter):
                if yate_pred[j] == 0:
                    y1.append('O')
                elif yate_pred[j] == 1:
                    y1.append('B-ASPECT')
                elif yate_pred[j] == 2:
                    y1.append('I-ASPECT')
                elif yate_pred[j] == 3:
                    y1.append('B-SENTIMENT')
                elif yate_pred[j] == 4:
                    y1.append('I-SENTIMENT')

                if yasc_pred[j] == 0:
                    y2.append('O')
                elif yasc_pred[j] == 1:
                    y2.append('PO')
                elif yasc_pred[j] == 2:
                    y2.append('NG')

            y.append([y1, y2])
        return y
            
    def save(self, filename):
        self.model.save_weights(filename, save_format='tf')

    def load(self, filename, X_train, y_train):
        self.init_model()
        self.model.fit(X_train[:4], [y_train[0][:4], y_train[1][:4], y_train[2][:4], y_train[3][:4], y_train[4][:4]], batch_size=self.config.batch_size, epochs=1)
        self.model.load_weights(filename)

        return self
    
    def evaluate(self, X, y, sentences=None):
        y_pred = self.predict(X, y)
        y_true_ate = []
        y_pred_ate = []
        y_true_asc = []
        y_pred_asc = []

        for seq in y:
            max_iter = min(len(seq[0]), self.config.max_sentence_size)
            for i in range(max_iter):
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
            max_iter = min(len(seq[0]), self.config.max_sentence_size)
            for i in range(max_iter):
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
        
        true_ate_label_seq = [x[0][:self.config.max_sentence_size] for x in y]
        true_asc_label_seq = [x[1][:self.config.max_sentence_size] for x in y]
            
        pred_ate_label_seq = [x[0][:self.config.max_sentence_size] for x in y_pred]
        pred_asc_label_seq = [x[1][:self.config.max_sentence_size] for x in y_pred]
            
        self.print_evaluations("Aspect and Sentiment Term Extraction", y_true_ate, y_pred_ate)
        target_names_ate = ['O', 'B-ASPECT', 'I-ASPECT', 'B-SENTIMENT', 'I-SENTIMENT']
        print(classification_report(y_true_ate, y_pred_ate, target_names=target_names_ate))
        if sentences != None:
            self.get_wrong_predictions(true_ate_label_seq, pred_ate_label_seq, sentences)
            
        self.print_evaluations("Aspect Sentiment Classification", y_true_asc, y_pred_asc)
        target_names_asc = ['O', 'PO', 'NG']
        print(classification_report(y_true_asc, y_pred_asc, target_names=target_names_asc))
        if sentences != None:
            self.get_wrong_predictions(true_asc_label_seq, pred_asc_label_seq, sentences)

    
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