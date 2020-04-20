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
    train_data = 'dataset/annotated/train_409.txt'
    test_data = 'dataset/annotated/test_324.txt'
    mpqa_lexicon_data = 'dataset/annotated/mpqa_lexicon.txt'
    general_embedding_model = '../word_embedding/general_embedding/general_embedding_300.model'
    domain_embedding_model = '../word_embedding/domain_embedding/domain_embedding_100.model'
    config = Config()
    config.mpqa_lexicon = load_lexicon(mpqa_lexicon_data)
    
    X, y = load_data(train_data)
    X_test, y_test = load_data(test_data)
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=42)
    sentences = X_val
    
    feature_extractor = FeatureExtractor(general_embedding_model, domain_embedding_model, general_dim=config.dim_general, domain_dim=config.dim_domain)

    X_train, y_train2 = prep_train_data(X_train, y_train, feature_extractor, feature='double_embedding', config=config)
    
    X_test = feature_extractor.get_features(X_test, max_len=config.max_sentence_size)
    X_val2 = feature_extractor.get_features(X_val, max_len=config.max_sentence_size)
    X_val, y_val2 = prep_train_data(X_val, y_val, feature_extractor, feature='double_embedding', config=config)
    
    coextractor = Coextractor(config)
    coextractor.load("model_weights_402", X_train, y_train2)
    print(coextractor.model.summary())
    np.random.seed(42)
    coextractor.evaluate(X_test, y_test)
#     coextractor.evaluate(X_val2, y_val, sentences)
