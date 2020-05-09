from tensorflow.keras import layers, Model
from tensorflow.keras.models import load_model, save_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow import initializers
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, classification_report
# from seqeval.metrics import classification_report, performance_measure
import numpy as np
import tensorflow as tf

from model.csu import CrossSharedUnit
from model.regu_cell import ReguCell

import matplotlib.pyplot as plt

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
                      return_sequences=True), name="first_ate_rnn")(input)
            first_asc_rnn = layers.Bidirectional(layers.RNN(ReguCell(hidden_size=self.config.hidden_size, return_sequences=True),
                      return_sequences=True), name="first_asc_rnn")(input)
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
        
        # csu
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
            second_ate_rnn = layers.Bidirectional(layers.RNN(ReguCell(hidden_size=self.config.hidden_size, return_sequences=True), return_sequences=True), name="second_ate_rnn")(split_ate)
            second_asc_rnn = layers.Bidirectional(layers.RNN(ReguCell(hidden_size=self.config.hidden_size, return_sequences=True), return_sequences=True), name="second_asc_rnn")(split_asc)
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
        asc_dense = layers.Dense(3, activation='softmax', name="asc_output")(second_asc_dropout)
        if self.config.te_output_topology == 'diff':
            ate_dense = layers.Dense(3, activation='softmax', name="ate_output")(second_ate_dropout)
            ste_dense = layers.Dense(3, activation='softmax', name="ste_output")(second_ate_dropout)
        else:
            ate_dense = layers.Dense(5, activation='softmax', name="ate_output")(second_ate_dropout)
        
        losses = {
            'ate_output': 'categorical_crossentropy',
            'asc_output': 'categorical_crossentropy',
            'sentiment_lexicon_enhancement': 'categorical_crossentropy',
            'aspect_term_length_enhancement': 'mean_squared_error',
            'aspect_polarity_length_enhancement': 'mean_squared_error'
        }
        if self.config.te_output_topology == 'diff':
            losses['ste_output'] = 'categorical_crossentropy'
        
        
        if self.config.te_output_topology == 'diff':
            outputs = [ate_dense, ste_dense, asc_dense, sentiment_lexicon_enhancement, aspect_term_length_enhancement, aspect_polarity_length_enhancement]
        else:
            outputs = [ate_dense, asc_dense, sentiment_lexicon_enhancement, aspect_term_length_enhancement, aspect_polarity_length_enhancement]
        
        self.model = Model(inputs=input, outputs=outputs)
        self.model.compile(optimizer='nadam', loss=losses, metrics=['accuracy'])
    
    def train(self, X_train, y_train, X_val=None, y_val=None):
        es = EarlyStopping(monitor='val_loss', mode='min', patience=self.config.patience)
        mc = ModelCheckpoint('/output/model_doer', monitor='val_loss', mode='min', save_best_only=True, save_weights_only=True)
        history = self.model.fit(X_train, y_train, validation_data=(X_val, y_val),
                       batch_size=self.config.batch_size,
                       epochs=self.config.epoch,
                       verbose=self.config.verbose,
                       callbacks=[es])
        
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.savefig('loss')
        
    def evaluate_coextraction(self, y_true, y_pred):
        count = 0
        y_true = self.generate_aspect_polarity_pairs(y_true)
        y_pred = self.generate_aspect_polarity_pairs(y_pred)
        
        for true_label, pred_label in zip(y_true, y_pred):
            if true_label == pred_label:
                count += 1
        
        return (count/len(y_true))
    
    def get_aspect_polarity_pairs_indexes(self, labels):
        """
        Get list of aspect index - polarity pairs 
        example : [([0, 1], 'PO'), [[3], 'NG']]

        Parameter
        ---------
        data : list of review text tokens
        labels: list of aspect/sentiment terms and polarity labels

        Return
        ------
        List of aspect index-polarity pairs of a review text
        """

        pairs = []
        term_labels, polarity_labels = labels

        indexes = []
        curr_indexes = []
        prev_label = 'O'
        for i in range(len(term_labels)):
            if term_labels[i] == 'B-ASPECT':
                if 'ASPECT' in prev_label:
                    indexes.append(curr_indexes)
                    curr_indexes = []
                curr_indexes.append(i)
            elif term_labels[i] == 'I-ASPECT':
                curr_indexes.append(i)
            else:
                if 'ASPECT' in prev_label:
                    indexes.append(curr_indexes)
                    curr_indexes = []
            prev_label = term_labels[i]

        if 'ASPECT' in prev_label:
            indexes.append(curr_indexes)

        polarities = []
        for aspect_indexes in indexes:
            count = {}
            for idx in aspect_indexes:
                if polarity_labels[idx] in count:
                    count[polarity_labels[idx]] += 1
                else:
                    count[polarity_labels[idx]] = 1
            polarities.append(max(count, key=count.get))

        curr_pairs = [(aspect_indexes, polarities[i]) for i, aspect_indexes in enumerate(indexes)]
        pairs.append(curr_pairs)

        return pairs
    
    def get_sentiments_indexes(self, labels):
        """
        Get sentiment expressions indexes from a label sequence

        Parameters
        ----------

        labels: list of aspect/sentiment terms and polarity labels

        Returns
        -------
        List of sentiment expression indexes
        """
        sentiments = []
        term_labels, polarity_labels = labels

        indexes = []
        curr_indexes = []
        prev_label = 'O'
        for i in range(len(term_labels)):
            if term_labels[i] == 'B-SENTIMENT':
                if 'SENTIMENT' in prev_label:
                    indexes.append(curr_indexes)
                    curr_indexes = []
                curr_indexes.append(i)
            elif term_labels[i] == 'I-SENTIMENT':
                curr_indexes.append(i)
            else:
                if 'SENTIMENT' in prev_label:
                    indexes.append(curr_indexes)
                    curr_indexes = []
            prev_label = term_labels[i]

        if 'SENTIMENT' in prev_label:
            indexes.append(curr_indexes)

        sentiments.append(indexes)

        return sentiments
    
    def generate_aspect_polarity_pairs(self, labels):
        aspect_polarity_pairs_list = [self.get_aspect_polarity_pairs_indexes(x) for x in labels]
        
        return aspect_polarity_pairs_list
    
    def generate_aspect_term_sentiment_term_polarity_triples(self, sentence, labels):
        """
        Get aspect term-sentimen term-polarity triples from each sentence in data

        Parameters
        ----------
        sentence: list of tokens of a review text
        aspect_polarity_pairs: list of aspect-polarity pairs for each review sentence
        sentiment_terms: list of sentiment terms for each review sentence

        Returns
        -------
        List of triples for each sentence
        """
        result = []

        aspect_polarity_pairs = self.get_aspect_polarity_pairs_indexes(labels)
        sentiment_indexes = self.get_sentiments_indexes(labels)

        for pairs, sent_indexes in zip(aspect_polarity_pairs, sentiment_indexes):
            triples = []

            for pair in pairs:
                aspect_idx = pair[0][0] # aspect term first token index
                chosen_sentiment_idx = None
                min_dist = 100

                for i in range(len(sent_indexes)):
                    sentiment_idx = sent_indexes[i][0] # sentiment term first token index
                    if abs(aspect_idx - sentiment_idx) <= min_dist:
                        chosen_sentiment_idx = i
                        min_dist = abs(aspect_idx - sentiment_idx)

                aspect_terms = []
                for idx in pair[0]:
                    aspect_terms.append(sentence[idx])

                sentiment_terms = []
                for idx in sent_indexes[chosen_sentiment_idx]:
                    sentiment_terms.append(sentence[idx])

                triples.append((' '.join(aspect_terms), ' '.join(sentiment_terms), pair[1]))

            result.append(triples)

        return result
    
    def predict_one(self, x, feature_extractor):
        sentence_length = len(x.split())
        sentence = x.split()
        x = feature_extractor.get_features([sentence], 'double_embedding', self.config.max_sentence_size)    
        scores = self.model.predict(np.asarray(x))

        yate_scores = scores[0]
        yste_scores = scores[1]
        yasc_scores = scores[2]

        yate_pred = np.argmax(yate_scores[0], 1)
        yste_pred = np.argmax(yste_scores[0], 1)
        yasc_pred = np.argmax(yasc_scores[0], 1)

        y1 = []
        y2 = []

        for i in range(sentence_length):
            if yate_pred[i] == 0:
                # Both results are O
                if yste_pred[i] == 0:
                    y1.append('O')

                # Aspect prediction is O and opinion is not O
                else:
                    if yste_pred[i] == 1:
                        y1.append('B-SENTIMENT')
                    else:
                        y1.append('I-SENTIMENT')

            elif yste_pred[i] == 0:
                # Aspect prediction is not O and opinion is O
                if yate_pred[i] == 1:
                    y1.append('B-ASPECT')
                else:
                    y1.append('I-ASPECT')

            # Both results are not O
            else:
                if yate_scores[0][i][yate_pred[i]] >= yste_scores[0][i][[yste_pred[i]]]:
                    if yate_pred[i] == 1:
                        y1.append('B-ASPECT')
                    else:
                        y1.append('I-ASPECT')
                else:
                    if yste_pred[i] == 1:
                        y1.append('B-SENTIMENT')
                    else:
                        y1.append('I-SENTIMENT')

            if yasc_pred[i] == 0:
                y2.append('O')
            elif yasc_pred[i] == 1:
                y2.append('PO')
            elif yasc_pred[i] == 2:
                y2.append('NG')
                
        return self.generate_aspect_term_sentiment_term_polarity_triples(sentence, [y1, y2])
    
    def predict(self, X, y_true):
        y = []
        if self.config.te_output_topology == 'diff':
            if not self.config.do_aus:
                yate_scores, yste_scores, yasc_scores, aspect_term_length_enhancement_scores, aspect_polarity_length_enhancement_scores = self.model.predict(np.asarray(X), batch_size=1)
            else:
                yate_scores, yste_scores, yasc_scores, lexicon_enhancement_scores, aspect_term_length_enhancement_scores, aspect_polarity_length_enhancement_scores = self.model.predict(np.asarray(X), batch_size=1)
        else:
            yate_scores, yasc_scores, lexicon_enhancement_scores, aspect_term_length_enhancement_scores, aspect_polarity_length_enhancement_scores = self.model.predict(np.asarray(X), batch_size=1)        
        
        for i in range(len(X)):
            yate_pred = np.argmax(yate_scores[i], 1)
            yasc_pred = np.argmax(yasc_scores[i], 1)
            if self.config.te_output_topology == 'diff':
                yste_pred = np.argmax(yste_scores[i], 1)
            
            y1 = []
            y2 = []
            max_iter = min(len(y_true[i][0]), self.config.max_sentence_size)
            for j in range(max_iter):
                if self.config.te_output_topology == 'diff':
                    if yate_pred[j] == 0:
                        # Both results are O
                        if yste_pred[j] == 0:
                            y1.append('O')

                        # Aspect prediction is O and opinion is not O
                        else:
                            if yste_pred[j] == 1:
                                y1.append('B-SENTIMENT')
                            else:
                                y1.append('I-SENTIMENT')

                    elif yste_pred[j] == 0:
                        # Aspect prediction is not O and opinion is O
                        if yate_pred[j] == 1:
                            y1.append('B-ASPECT')
                        else:
                            y1.append('I-ASPECT')

                    # Both results are not O
                    else:
                        if yate_scores[i][j][yate_pred[j]] >= yste_scores[i][j][[yste_pred[j]]]:
                            if yate_pred[j] == 1:
                                y1.append('B-ASPECT')
                            else:
                                y1.append('I-ASPECT')
                        else:
                            if yste_pred[j] == 1:
                                y1.append('B-SENTIMENT')
                            else:
                                y1.append('I-SENTIMENT')
                
                else:
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

    def load(self, filename):
        self.init_model()
        self.model.load_weights(filename)
    
    def evaluate(self, X, y, sentences=None, print_report=None):
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
        if print_report:
            target_names_ate = ['O', 'B-ASPECT', 'I-ASPECT', 'B-SENTIMENT', 'I-SENTIMENT']
            print(classification_report(y_true_ate, y_pred_ate, target_names=target_names_ate))
        if sentences:
            self.get_wrong_predictions(true_ate_label_seq, pred_ate_label_seq, sentences)
            
#         self.print_evaluations("Aspect Sentiment Classification", y_true_asc, y_pred_asc)
#         if print_report:
#             target_names_asc = ['O', 'PO', 'NG']
#             print(classification_report(y_true_asc, y_pred_asc, target_names=target_names_asc))
#         if sentences:
#             self.get_wrong_predictions(true_asc_label_seq, pred_asc_label_seq, sentences)
        
        # coextraction F1/accuracy
        print('Coextraction f1/acc : ', self.evaluate_coextraction(y, y_pred))
        
        return [f1_score(y_true_ate, y_pred_ate, average='macro'), f1_score(y_true_asc, y_pred_asc, average='macro')]
    
    def print_evaluations(self, task_name, y_true, y_pred):
        print(task_name)
#         print("Confusion Matrix:")
#         print(confusion_matrix(y_true, y_pred))
#         print()
        print("Precision : ", precision_score(y_true, y_pred, average='macro'))
        print("Recall : ", recall_score(y_true, y_pred, average='macro'))
        print("F1-score : ", f1_score(y_true, y_pred, average='macro'))
            
        
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
        