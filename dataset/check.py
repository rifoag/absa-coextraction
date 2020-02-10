def load_data(filename):
    data, labels = [], []
    with open(filename) as f:
        tokens, tags = [], []
        for line in f:
            line = line.rstrip()
            if line:
                token, tag = line.split('\t')
                tokens.append(token)
                tags.append(tag)
            else:
                data.append(tokens)
                labels.append(tags)
                tokens, tags = [], []

    return data, labels

data_4k, labels_4k = load_data("train_4k.txt")
data_extra, labels_extra = load_data("extra_train_4k.txt")

data_4k_zipped = list(zip(data_4k, labels_4k))
data_extra_zipped = list(zip(data_extra, labels_extra))

