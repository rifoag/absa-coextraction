from model.feature_extractor import FeatureExtractor
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

def prep_train_data(X, y, feature_extractor, feature='double_embedding', batch=True):
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
    
    y_train = []
    y_asp_sent = []
    y_polarity = []
    
    for asp_sent_terms, polarities in y:
        ya = []
        yp = []
        
        for asp_sent_term in asp_sent_terms:
            if asp_sent_term == 'O':
                ya.append([1, 0, 0, 0, 0])
            elif asp_sent_term == 'B-ASPECT':
                ya.append([0, 1, 0, 0, 0])
            elif asp_sent_term == 'I-ASPECT':
                ya.append([0, 0, 1, 0, 0])
            elif asp_sent_term == 'B-SENTIMENT':
                ya.append([0, 0, 0, 1, 0])
            elif asp_sent_term == 'I-SENTIMENT':
                ya.append([0, 0, 0, 0, 1])
            
        for polarity in polarities:
            if polarity == 'O':
                yp.append([1, 0, 0, 0, 0])
            elif polarity == 'PO':
                yp.append([0, 1, 0, 0, 0])
            elif polarity == 'NG':
                yp.append([0, 0, 1, 0, 0])
            elif polarity == 'NT':
                yp.append([0, 0, 0, 1, 0])
            elif polarity == 'CF':
                yp.append([0, 0, 0, 0, 1])
        
        for j in range(len(asp_sent_terms), max_len):
            ya.append([1, 0, 0, 0, 0])
            yp.append([1, 0, 0, 0, 0])
        
        y_asp_sent.append(ya)
        y_polarity.append(yp)
    return np.asarray(X_train), [np.asarray(y_asp_sent), np.asarray(y_polarity)]
