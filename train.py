# from model.doer import Coextractor
from model.feature_extractor import FeatureExtractor
from datetime import timedelta
from utils import load_data, prep_train_data
import argparse
import numpy as np
from config import Config

def config_from_args(args):
    config = Config()
    for key, value in vars(args).items():
        config.__dict__[key] = value
    # config.auto_config()
    return config

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--train_data', default='dataset/train.txt')
    parser.add_argument('--test_data', default='dataset/test.txt')

    parser.add_argument('--general_embedding_model', default='../word_embedding/general_embedding.vec')
    parser.add_argument('--domain_embedding_model', default='../word_embedding/domain_embedding/domain_embedding_100.model')
    parser.add_argument('--dim_general', type=int, default=300)
    parser.add_argument('--dim_domain', type=int, default=100)

    parser.add_argument('--hidden_size', type=int, default=50)
    # parser.add_argument('--n_tensors', type=int, default=20)
    # parser.add_argument('--n_layers', type=int, default=2)
    parser.add_argument('--dropout_rate', type=float, default=0.5)
    parser.add_argument('--rnn_cell', default='gru')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--nepochs', type=int, default=15)
    # parser.add_argument('--verbose', type=int, default=0)

    args = parser.parse_args()
    config = config_from_args(args)

    batch = False if args.batch_size == 1 else True

    X_train, y_train = load_data(args.train_data)
    X_test, y_test = load_data(args.test_data)

    feature_extractor = FeatureExtractor(args.general_embedding_model, args.domain_embedding_model, args.dim_general, args.dim_domain)

    X_train, y_train = prep_train_data(X_train, y_train, feature_extractor, batch)
    X_test = feature_extractor.get_features(X_test)

    input_size = args.general_embedding_dim + args.domain_embedding_dim
    print(X_train)
    # extractor = AspectOpinionExtractor()
    # extractor.init_model(input_size=input_size,
    #                      n_hidden = args.hidden_units,
    #                      n_tensors = args.n_tensors,
    #                      n_layers = args.n_layers,
    #                      dropout_rate = args.dropout_rate,
    #                      rnn_type = args.rnn_type)

    # print(extractor.get_summary())

    # print("TRAIN:")
    # start_time = time.time()
    # np.random.seed(42)
    # extractor.fit(X_train, y_train, args.epoch, args.batch_size, args.verbose)
    # finish_time = time.time()
    # print('Elapsed time: {}'.format(timedelta(seconds=finish_time-start_time)))
    # extractor.evaluate(X_test, y_test)
    # extractor.save("final_model.mdl")
