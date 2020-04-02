from model.doer import Coextractor
from model.feature_extractor import FeatureExtractor
from datetime import timedelta
from utils import load_data, prep_train_data
import argparse
import numpy as np
from config import Config
import time


if __name__ == "__main__":
    train_data = 'dataset/annotated/train_small.txt'
    test_data = 'dataset/annotated/test_small.txt'
    general_embedding_model = '../word_embedding/general_embedding/general_embedding_300.model'
    domain_embedding_model = '../word_embedding/domain_embedding/domain_embedding_100.model'
    config = Config()

    X_train, y_train = load_data(train_data)
    X_test, y_test = load_data(test_data)
    sentences = X_test
    
    feature_extractor = FeatureExtractor(general_embedding_model, domain_embedding_model, general_dim=config.dim_general, domain_dim=config.dim_domain)
    config.max_sentence_size = feature_extractor.get_max_len(X_train)

    X_train, y_train = prep_train_data(X_train, y_train, feature_extractor, feature='double_embedding', config=config)
    X_test = feature_extractor.get_features(X_test, max_len=config.max_sentence_size)
    
    coextractor = Coextractor(config)
    coextractor.init_model()
    print(coextractor.model.summary())
    np.random.seed(55)
    print('TRAIN:')
    start_time = time.time()
    coextractor.train(X_train, y_train)
    finish_time = time.time()
    print('Elapsed time: {}'.format(timedelta(seconds=finish_time-start_time)))
    coextractor.save('model_weights_324')
    coextractor.evaluate(X_test, y_test, sentences)
    