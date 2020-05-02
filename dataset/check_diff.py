import csv

def load_data(filename):
    data, labels = [], []
    with open(filename, encoding='utf-8') as f:
        tokens, ate_labels = [], []
        for line in f:
            line = line.rstrip()
            if line:
                line_arr = line.split('\t')
                if len(line_arr) > 1:
                    if len(line_arr) == 3:
                        token, ate_label, asc_label = line_arr
                    else:
                        token, ate_label = line_arr
                    tokens.append(token)
                    ate_labels.append(ate_label)
                else:
                    continue
            else:
                data.append(tokens)
                labels.append(ate_labels)
                tokens, ate_labels = [], []
    return data, labels

def get_aspects(labels):
    """
    Get aspect expressions from a label sequence

    Parameters
    ----------
    labels: ATE labels (list).
    
    Returns
    -------
    List of aspect expression
    """
    aspects = []
    curr_aspect = []
    prev_label = ''
    
    for i in range(len(labels)):
        if labels[i] == 'B-ASPECT':
            if 'ASPECT' in prev_label:
                aspects.append(curr_aspect)
                curr_aspect = []
            curr_aspect.append(i)
        elif labels[i] == 'I-ASPECT' and ('ASPECT' in prev_label):
            curr_aspect.append(i)
        else: # 'O'
            if curr_aspect:
                aspects.append(curr_aspect)
                curr_aspect = []
        prev_label = labels[i]
    
    if 'ASPECT' in prev_label:
        aspects.append(curr_aspect)

    return aspects

def get_sentiments(labels):
    """
    Get sentiment expressions from a label sequence

    Parameters
    ----------
    labels: ATE labels (list).
    
    Returns
    -------
    List of sentiment expression
    """
    sentiments = []
    curr_sentiment = []
    prev_label = ''
    
    for i in range(len(labels)):
        if labels[i] == 'B-SENTIMENT':
            if 'SENTIMENT' in prev_label:
                sentiments.append(curr_sentiment)
                curr_sentiment = []
            curr_sentiment.append(i)
        elif labels[i] == 'I-SENTIMENT' and ('SENTIMENT' in prev_label):
            curr_sentiment.append(i)
        else: # 'O'
            if curr_sentiment:
                sentiments.append(curr_sentiment)
                curr_sentiment = []
        prev_label = labels[i]
    
    if 'SENTIMENT' in prev_label:
        sentiments.append(curr_sentiment)

    return sentiments

def get_differences(labels_list, old_labels_list):
    """
    Get labeling difference between labels from two dataset

    Parameters
    ----------
    labels_list: list of ATE labels (list).
    old_labels_list: list of ATE labels (list).
    
    Returns
    -------
    List of list consist of [row_id, old_aspects, aspects, old_sentiments, sentiments] of row with different labeling
    """
    diff_matrix = []
    for i in range(len(labels_list)):
        aspects = get_aspects(labels_list[i])
        old_aspects = get_aspects(old_labels_list[i])
        sentiments = get_sentiments(labels_list[i])
        old_sentiments = get_sentiments(old_labels_list[i])
        if aspects != old_aspects or sentiments != old_sentiments:
            diff_matrix.append([i, old_aspects, aspects, old_sentiments, sentiments])

    return diff_matrix

def get_terms(sentence, positions):
    """
    Get actual terms from list of positions 

    Parameters
    ----------
    sentence : token (list)
    positions : aspect position in sentence
    """
    terms = []

    for pos in positions:
        curr_term = []
        for idx in pos:
            curr_term.append(sentence[idx])
        terms.append(' '.join(curr_term))

    return ', '.join(terms)

def write_difference(data, diff_matrix, filename):
    """
    write labeling differences to filename.
    row format : [sentence, old_aspects, aspects, old_sentiments, sentiments]

    Parameters
    ----------
    data: list of tokens (list).
    diff_matrix: [[row_id, old_aspects, aspects, old_sentiments, sentiments]]
    """
    with open('{}'.format(filename), 'w') as out_file:
        tsv_writer = csv.writer(out_file, delimiter='\t')
        for diff in diff_matrix:
            row_id, old_aspects, aspects, old_sentiments, sentiments = diff
            sentence = data[row_id]
            old_aspects = get_terms(sentence, old_aspects)
            aspects = get_terms(sentence, aspects)
            old_sentiments = get_terms(sentence, old_sentiments)
            sentiments = get_terms(sentence, sentiments)
            tsv_writer.writerow([' '.join(sentence), old_aspects, aspects, old_sentiments, sentiments])


data, labels_list = load_data("test_1k.txt")
old_data, old_labels_list = load_data("old/test_1k.txt")
diff_matrix = get_differences(labels_list, old_labels_list)
write_difference(data, diff_matrix, "test_difference.tsv")
