import csv

def load_data(filename):
    data, labels = [], []
    with open(filename) as f:
        tokens, ate_labels, asc_labels = [], [], []
        for line in f:
            line = line.rstrip()
            if line:
                line_arr = line.split('\t')
                if len(line_arr) > 1:
                    if len(line_arr) == 3:
                        token, ate_label, asc_label = line_arr
                        asc_labels.append(asc_label)
                    else:
                        token, ate_label = line_arr
                    tokens.append(token)
                    ate_labels.append(ate_label)
                else:
                    continue
            else:
                data.append(tokens)
                labels.append([ate_labels, asc_labels])
                tokens, ate_labels, asc_labels = [], [], []
    return data, labels

def merge_labels(old_labels_list, labels_list):
    result = []

    for old_labels, labels in zip(old_labels_list, labels_list):
        result.append([old_labels[0], labels[1]])

    return result

def write_final(data, labels, filename):
    with open('{}'.format(filename), 'w') as out_file:
        tsv_writer = csv.writer(out_file, delimiter='\t')
        for review, label_line in zip(data, labels):
            aspect_sentiment_labels, polarity_labels = label_line
            for token, aspect_sentiment_label, polarity_label in zip(review, aspect_sentiment_labels, polarity_labels):
                tsv_writer.writerow([token, aspect_sentiment_label, polarity_label])
            tsv_writer.writerow([])

data, labels_list = load_data("annotated/train_4k.txt")
old_data, old_labels_list = load_data("old/train_4k.txt")

labels = merge_labels(old_labels_list, labels_list)

write_final(data, labels, "train_4k.txt")