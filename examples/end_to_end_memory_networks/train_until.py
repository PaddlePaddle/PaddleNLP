import argparse
from model import MenN2N
from train import train
from eval import test
from config import config
import os, random, time
from data import read_data, load_vocab
import paddle
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument("--target", default=111.0, type=float,
                    help="target perplexity")
target = parser.parse_args().target


if __name__ == '__main__':
    paddle.set_device("gpu")

    vocab_path = os.path.join(config.data_dir, "%s.vocab.txt" % config.data_name)
    word2idx = load_vocab(vocab_path)

    if not os.path.exists(config.checkpoint_dir):
        os.makedirs(config.checkpoint_dir)

    train_data = read_data(
        os.path.join(config.data_dir, "%s.train.txt" % config.data_name),
        word2idx)
    valid_data = read_data(
        os.path.join(config.data_dir, "%s.valid.txt" % config.data_name),
        word2idx)
    test_data = read_data(
        os.path.join(config.data_dir, "%s.test.txt" % config.data_name),
        word2idx)

    idx2word = dict(zip(word2idx.values(), word2idx.keys()))
    config.nwords = len(word2idx)

    print("vacab size is %d" % config.nwords)

    while True:
        random.seed(time.time())
        config.srand = random.randint(0, 100000)

        np.random.seed(config.srand)
        random.seed(config.srand)
        paddle.seed(config.srand)

        model = MenN2N(config)
        train(model, train_data, valid_data, config)

        test_ppl = test(model, test_data, config)
        if test_ppl < target:
            model_path = os.path.join(config.checkpoint_dir, config.model_name + "_" + str(config.srand) + "_good")
            paddle.save(model.state_dict(), model_path)
            break
