import numpy as np
import csv

def load_data(filename):
    data, labels = [], []
    with open(filename, encoding='utf-8') as f:
        tokens, asp_sent_term_tags, polarity_tags = [], [], []
        for line in f:
            line = line.rstrip()
            if line:
                line_arr = line.split('\t')
                if len(line_arr) > 1:
                    token = line_arr[2]
                    asp_sent_term_tag = line_arr[3]
                    polarity_tag = line_arr[4]
                    tokens.append(token)
                    asp_sent_term_tags.append(asp_sent_term_tag)
                    polarity_tags.append(polarity_tag)
                else:
                    continue
            else:
                data.append(tokens)
                labels.append([asp_sent_term_tags, polarity_tags])
                tokens, asp_sent_term_tags, polarity_tags = [], [], []
    return data, labels

def convert_labels(labels):
    for label_line in labels:
        aspect_sent_tags = label_line[0]
        polarity_tags = label_line[1]
        aspect_flag = False
        sent_flag = False
        for i in range(len(polarity_tags)):
            if 'ASPECT' in aspect_sent_tags[i]:
                sent_flag = False
                if aspect_flag:
                    aspect_sent_tags[i] = 'I-ASPECT'
                else:
                    aspect_sent_tags[i] = 'B-ASPECT'
                    aspect_flag = True
            elif 'SENTIMENT' in aspect_sent_tags[i]:
                aspect_flag = False
                if sent_flag:
                    aspect_sent_tags[i] = 'I-SENTIMENT'
                else:
                    aspect_sent_tags[i] = 'B-SENTIMENT'
                    sent_flag = True
            else:
                aspect_flag = False
                sent_flag = False
                aspect_sent_tags[i] = 'O'

            if 'PO' in polarity_tags[i]:
                polarity_tags[i] = 'PO'
            elif 'NG' in polarity_tags[i]:
                polarity_tags[i] = 'NG'
            elif 'NT' in polarity_tags[i]:
                polarity_tags[i] = 'NT'
            elif 'CF' in polarity_tags[i]:
                polarity_tags[i] = 'CF'
            else:
                polarity_tags[i] = 'O'
    return labels

def write_data(data, labels, filename):
    with open('{}'.format(filename), 'w') as out_file:
        tsv_writer = csv.writer(out_file, delimiter='\t')
        for review, label_line in zip(data, labels):
            aspect_sentiment_labels, polarity_labels = label_line
            for token, aspect_sentiment_label, polarity_label in zip(review, aspect_sentiment_labels, polarity_labels):
                tsv_writer.writerow([token, aspect_sentiment_label, polarity_label])
            tsv_writer.writerow([])
        
data, labels = load_data("train_4k_output_Mar_19_2020.txt")
data.pop(0)
labels.pop(0)
labels = convert_labels(labels)
data, labels = data[2000:], labels[2000:]
print(len(data))
write_data(data, labels, "train.txt")

