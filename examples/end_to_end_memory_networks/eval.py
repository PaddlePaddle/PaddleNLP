import paddle
from paddle import nn
import numpy as np
from model import MenN2N
from data import read_data, load_vocab
import math, os
from importlib import import_module
from config import config


@paddle.no_grad()
def eval(model: MenN2N, data, config, title="Test"):
    """
    测试
    :param model: 用来测试的模型
    :param data: 测试数据
    :param config: 配置信息
    :param title: 本论测试的Title(Valid 或 Test)
    :return: 平均loss
    """
    model.eval()
    lossfn = nn.CrossEntropyLoss(reduction='sum')
    N = int(math.ceil(len(data) / config.batch_size))
    total_loss = 0

    context = np.ndarray([config.batch_size, config.mem_size], dtype=np.int64)
    target = np.ndarray([config.batch_size], dtype=np.int64)

    if config.show:
        ProgressBar = getattr(import_module('utils'), 'ProgressBar')
        bar = ProgressBar(title, max=N - 1)

    m = config.mem_size
    for batch in range(N):
        if config.show:
            bar.next()

        for i in range(config.batch_size):
            if m >= len(data):
                break
            target[i] = data[m]
            context[i, :] = data[m - config.mem_size: m]
            m += 1
        if m >= len(data):
            break

        batch_data = paddle.to_tensor(context)
        batch_label = paddle.to_tensor(target)

        preict = model(batch_data)
        loss = lossfn(preict, batch_label)

        total_loss += loss

    if config.show:
        bar.finish()

    return total_loss / N / config.batch_size


def test(model: MenN2N, test_data, config):
    test_loss = eval(model, test_data, config, "Test")
    test_perplexity = math.exp(test_loss)
    print("Perplexity on Test: %f" % test_perplexity)


if __name__ == '__main__':
    paddle.set_device("gpu")

    vocab_path = os.path.join(config.data_dir,
                              "%s.vocab.txt" % config.data_name)
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

    model = MenN2N(config)

    model_path = os.path.join(config.checkpoint_dir, config.model_name)
    state_dict = paddle.load(model_path)
    model.set_dict(state_dict)
    test(model, test_data, config)