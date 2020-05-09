from model.doer import Coextractor
from model.feature_extractor import FeatureExtractor
from config import Config
from utils import load_data, prep_train_data, load_lexicon

from sklearn.model_selection import train_test_split

from tensorflow.keras.backend import clear_session
import argparse
import numpy as np
import time

from datetime import timedelta

if __name__ == "__main__":
    np.random.seed(42)
    clear_session()
    
    train_data = 'dataset/train_4k.txt'
    test_data = 'dataset/test_1k.txt'
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

    X_train, y_train = prep_train_data(X_train, y_train, feature_extractor, feature='double_embedding', config=config)
    
    X_test = feature_extractor.get_features(X_test, max_len=config.max_sentence_size)
    X_val, y_val2 = prep_train_data(X_val, y_val, feature_extractor, feature='double_embedding', config=config)
    
    coextractor = Coextractor(config)
    coextractor.load("saved_models/P1_ReGU-P2_Same/P1_ReGU_weights")
    print(coextractor.model.summary())
    
    coextractor.evaluate(X_val, y_val)
#     print()
#     print("Test Data : ")
#     print()
#     coextractor.evaluate(X_test, y_test)
