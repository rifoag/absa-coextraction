from model.feature_extractor import FeatureExtractor
from sklearn import preprocessing
import numpy as np

def load_data(filename):
    data, labels = [], []
    with open(filename, encoding='utf-8') as f:
        tokens, asp_sent_tags, polarity_tags = [], [], []
        for line in f:
            line = line.rstrip()
            if line:
                token, asp_sent_tag, polarity_tag = line.split('\t')
                tokens.append(token)
                asp_sent_tags.append(asp_sent_tag)
                polarity_tags.append(polarity_tag)
            else:
                data.append(tokens)
                labels.append([asp_sent_tags, polarity_tags])
                tokens, asp_sent_tags, polarity_tags = [], [], []

    return data, labels

def prep_train_data(X, y, feature_extractor, feature='double_embedding', batch=False):
    """
    Convert data and labels into compatible format for training.

    Parameters
    ----------
    X: Train data (list).
    y: labels (list).
    feature_extractor: FeatureExtractor.
    batch: Train in batch or not. Default: False

    Returns
    -------
    tuple of data and labels in compatible format for training.
    """
    if batch:
        max_len = feature_extractor.get_max_len(X)
    else:
        max_len = None

    X_train = feature_extractor.get_features(X, feature, max_len)
    lb_asp_sent_term = preprocessing.LabelBinarizer()
    lb_polarity = preprocessing.LabelBinarizer()
    lb_asp_sent_term.fit(['B-ASPECT', 'I-ASPECT', 'B-SENTIMENT', 'I-SENTIMENT', 'O'])
    lb_polarity.fit(['PO', 'NG', 'NT', 'CF', 'O'])

    for asp_sent_term, polarity in y:
        lb_asp_sent_term.transform(asp_sent_term)
        lb_polarity.transform(polarity)
    
    return X_train, y