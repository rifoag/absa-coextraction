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

def load_lexicon(filename):
    mpqa_lexicon = {}
    with open(filename, encoding='utf-8') as f:
        for line in f:
            line = line.rstrip()
            if line:
                token, subjectivity = line.split('\t')
                mpqa_lexicon[token] = subjectivity
    return mpqa_lexicon

def normalize_term_average_lengths(terms_average_lengths):
    max_el = max(terms_average_lengths)
    min_el = min(terms_average_lengths)
    return [[(x-min_el)/(max_el-min_el)] for x in terms_average_lengths]

def get_auxiliary_labels(x, y, mpqa_lexicon, max_len):
    subjectivities, aspect_term_average_lengths, polarity_average_lengths = [], [], []
    
    # Get token subjectivities 
    for sentence in x:
        subj_in_sentence = []

        i = 0
        for token in sentence:
            if i >= max_len:
                break
            try:
                subj = mpqa_lexicon[token]
                if subj == 'O':    
                    subj_in_sentence.append([1, 0, 0])
                elif subj == 'PO':
                    subj_in_sentence.append([0, 1, 0])
                elif subj == 'NG':
                    subj_in_sentence.append([0, 0, 1])

            except:
                subj_in_sentence.append([1, 0, 0])
            i += 1
        
        for j in range(len(subj_in_sentence), max_len):
            subj_in_sentence.append([1, 0, 0])
        
        subjectivities.append(subj_in_sentence)
    
    # Get aspect term average lengths
    for asp_sent_labels, polarities in y:
        aspect_term_average_lengths.append(get_aspect_term_average_length(asp_sent_labels))
        polarity_average_lengths.append(get_polarity_average_length(polarities))
    
    aspect_term_average_lengths = normalize_term_average_lengths(aspect_term_average_lengths)
    polarity_average_lengths = normalize_term_average_lengths(polarity_average_lengths)
    return subjectivities, aspect_term_average_lengths, polarity_average_lengths

def get_aspect_term_average_length(labels):
    asp_length_sum = 0
    asp_label_count = 0
    current_length = 0
    prev_label = ''

    for label in labels:
        if label == 'B-ASPECT':
            if (prev_label == 'B-ASPECT') or (prev_label == 'I-ASPECT'):
                asp_length_sum += current_length
                asp_label_count += 1
            current_length = 1
        elif label == 'I-ASPECT':
            current_length += 1
        else:
            if (prev_label == 'B-ASPECT') or (prev_label == 'I-ASPECT'):
                asp_length_sum += current_length
                asp_label_count += 1
            current_length = 0
        prev_label = label

    try:
        asp_length_average = asp_length_sum/asp_label_count
    except:
        asp_length_average = 0
    return asp_length_average
    
def get_polarity_average_length(labels):
    polarity_length_sum = 0
    polarity_label_count = 0
    current_length = 0
    prev_label = ''

    for label in labels:
        if label == 'O':
            if prev_label != 'O' and prev_label != '':
                polarity_length_sum += current_length
                polarity_label_count += 1
            current_length = 0
        elif label == 'PO':
            if prev_label == 'PO':
                current_length += 1
            elif prev_label == 'NG':
                polarity_length_sum += current_length
                polarity_label_count += 1
                current_length = 1
            else:
                current_length = 1
        elif label == 'NG':
            if prev_label == 'NG':
                current_length += 1
            elif prev_label == 'PO':
                polarity_length_sum += current_length
                polarity_label_count += 1
                current_length = 1
            else:
                current_length = 1
        prev_label = label

    try:
        polarity_length_average = polarity_length_sum/polarity_label_count
    except:
        polarity_length_average = 0

    return polarity_length_average

def get_labels(y, config):
    """
    Get TE and ASC labels in compatible format for training

    Parameters
    ----------
    y: labels (list).
    config: include max_sentence_size for batch training and term extraction output topology configuration

    Returns
    -------
    tuple (or triple) labels in compatible format for training.
    """
    y_aspects = []
    y_sentiments = []
    y_polarities = []
    max_len = config.max_sentence_size
    
    for asp_sent_labels, polarities in y:
        ya = []
        ys = []
        yp = []
        
        i = 0
        if config.te_output_topology == 'diff':
            for asp_sent_label in asp_sent_labels:
                if i >= max_len:
                    break
                if asp_sent_label == 'O':
                    ya.append([1, 0, 0])
                    ys.append([1, 0, 0])
                elif asp_sent_label == 'B-ASPECT':
                    ya.append([0, 1, 0])
                    ys.append([1, 0, 0])
                elif asp_sent_label == 'I-ASPECT':
                    ya.append([0, 0, 1])
                    ys.append([1, 0, 0])
                elif asp_sent_label == 'B-SENTIMENT':
                    ya.append([1, 0, 0])
                    ys.append([0, 1, 0])
                elif asp_sent_label == 'I-SENTIMENT':
                    ya.append([1, 0, 0])
                    ys.append([0, 0, 1])
                i += 1
        else:
            for asp_sent_label in asp_sent_labels:
                if i >= max_len:
                    break
                if asp_sent_label == 'O':
                    ya.append([1, 0, 0, 0, 0])
                elif asp_sent_label == 'B-ASPECT':
                    ya.append([0, 1, 0, 0, 0])
                elif asp_sent_label == 'I-ASPECT':
                    ya.append([0, 0, 1, 0, 0])
                elif asp_sent_label == 'B-SENTIMENT':
                    ya.append([0, 0, 0, 1, 0])
                elif asp_sent_label == 'I-SENTIMENT':
                    ya.append([0, 0, 0, 0, 1])
                i += 1
        
        i = 0
        for polarity in polarities:
            if i >= max_len:
                break
            if polarity == 'O':
                yp.append([1, 0, 0])
            elif polarity == 'PO':
                yp.append([0, 1, 0])
            elif polarity == 'NG':
                yp.append([0, 0, 1])
            i += 1
        
        # Add post-padding
        for j in range(len(asp_sent_labels), max_len):
            if config.te_output_topology == 'diff':
                ya.append([1, 0, 0])
                ys.append([1, 0, 0])
            else:
                ya.append([1, 0, 0, 0, 0])
            yp.append([1, 0, 0])
        
        y_aspects.append(ya)
        if config.te_output_topology == 'diff':
            y_sentiments.append(ys)
        y_polarities.append(yp)
        
    if config.te_output_topology == 'diff':
        result = [y_aspects, y_sentiments, y_polarities]
    else:
        result = [y_aspects, y_polarities]
    return result

def prep_train_data(X, y, feature_extractor, feature='double_embedding', config=None):
    """
    Convert data and labels into compatible format for training.

    Parameters
    ----------
    X: Train data (list).
    y: labels (list).
    feature_extractor: FeatureExtractor.
    config: include max_sentence_size for batch training and term extraction output topology configuration

    Returns
    -------
    tuple of data and labels in compatible format for training.
    """
    if config.batch_size > 0:
        max_len = config.max_sentence_size
    else:
        max_len = None
    
    X_train = feature_extractor.get_features(X, feature, max_len)
    
    y_mpqa, y_ate_avg_lengths, y_asc_avg_lengths = get_auxiliary_labels(X, y, config.mpqa_lexicon, config.max_sentence_size)
    
    if config.te_output_topology == 'diff':
        y_aspects, y_sentiments, y_polarities = get_labels(y, config)
        y_train = [np.asarray(y_aspects), np.asarray(y_sentiments), np.asarray(y_polarities), np.asarray(y_mpqa), np.asarray(y_ate_avg_lengths), np.asarray(y_asc_avg_lengths)]
        if not config.do_aul:
            y_train = [np.asarray(y_aspects), np.asarray(y_sentiments), np.asarray(y_polarities), np.asarray(y_mpqa)]
        elif not config.do_aus:
            y_train = [np.asarray(y_aspects), np.asarray(y_sentiments), np.asarray(y_polarities), np.asarray(y_ate_avg_lengths), np.asarray(y_asc_avg_lengths)]
    else:
        y_asp_sents, y_polarities = get_labels(y, config)
        y_train = [np.asarray(y_asp_sents), np.asarray(y_polarities), np.asarray(y_mpqa), np.asarray(y_ate_avg_lengths), np.asarray(y_asc_avg_lengths)]
        
    
        
    return np.asarray(X_train), y_train
