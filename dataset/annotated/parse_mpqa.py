import numpy as np
import csv

def get_unique_tokens(filename):
    tokens = []
    with open(filename, encoding='utf-8') as f:
        for line in f:
            line = line.rstrip()
            if line:
                line_arr = line.split('\t')
                token = line_arr[0]
                if token not in tokens:
                    tokens.append(token)

    return tokens

def write_data(tokens, filename):
    with open('{}'.format(filename), 'w') as out_file:
        tsv_writer = csv.writer(out_file, delimiter='\t')
        for token in tokens:
            tsv_writer.writerow([token, 'O'])
        tsv_writer.writerow([])
        
tokens = get_unique_tokens("train_326.txt")
tokens_test = get_unique_tokens("test_324.txt")
for token in tokens_test:
    if token not in tokens:
        tokens.append(token)
tokens.sort()
print(len(tokens))
write_data(tokens, "mpqa.txt")

