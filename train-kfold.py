from model.doer import Coextractor
from model.feature_extractor import FeatureExtractor
from datetime import timedelta
from utils import load_data, prep_train_data, load_lexicon
from sklearn.model_selection import train_test_split
import argparse
import numpy as np
from config import Config
import time

if __name__ == "__main__":
    np.random.seed(42)
    train_data = 'dataset/annotated/train_small.txt'
    test_data = 'dataset/annotated/test_small.txt'
    mpqa_lexicon_data = 'dataset/annotated/mpqa_lexicon.txt'
    general_embedding_model = '../word_embedding/general_embedding/general_embedding_300.model'
    domain_embedding_model = '../word_embedding/domain_embedding/domain_embedding_100.model'
    config = Config()
    config.mpqa_lexicon = load_lexicon(mpqa_lexicon_data)
    
    X, y = load_data(train_data)
    X, y = X[:64], y[:64]
    feature_extractor = FeatureExtractor(general_embedding_model, domain_embedding_model, general_dim=config.dim_general, domain_dim=config.dim_domain)
    data_size = len(X)
    X, y = np.array(X), np.array(y)
    
    print('TRAIN:')
    f1_scores = []
    start_time = time.time()
    for i in range(5): # 5-fold
        print('FOLD ', i+1)
        test_start_index = (data_size/5)*i
        train_index = []
        test_index = []
        for i in range(data_size):
            if i >= test_start_index and i < test_start_index + 12:
                test_index.append(i)
            else:
                train_index.append(i)
        train_index, test_index = np.array(train_index), np.array(test_index)
        
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
    
        X_train, y_train = prep_train_data(X_train, y_train, feature_extractor, feature='double_embedding', config=config)
        X_test = feature_extractor.get_features(X_test, max_len=config.max_sentence_size)
            
        coextractor = Coextractor(config)
        coextractor.init_model()
        coextractor.train(X_train, y_train)
        coextractor.save('saved_models/model_weights_P1_ReGU')
        f1_scores.append(coextractor.evaluate(X_test, y_test))

    
    finish_time = time.time()
    ate_scores = []
    asc_scores = []
    for score in f1_scores:
        print("ATE : ", score[0])
        ate_scores.append(score[0])
        print("ASC : ", score[1])
        asc_scores.append(score[1])
        
    print("Average ATE : ", sum(ate_scores)/len(ate_scores))
    print("Average ASC : ", sum(asc_scores)/len(asc_scores))
    
    print('Elapsed time: {}'.format(timedelta(seconds=finish_time-start_time)))    
    