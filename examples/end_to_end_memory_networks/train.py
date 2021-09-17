import paddle
from paddle import nn
from model import MenN2N
from config import config
from eval import eval
from data import load_vocab, read_data
from importlib import import_module
import os, math
import numpy as np
import random


def train_single_step(model: MenN2N, lr, data, config):
    """
    训练一个epoch
    :param model: 训练的模型
    :param lr: 本epoch的learning rate
    :param data: 训练数据
    :param config: 配置信息
    :return: 平均loss
    """
    model.train()
    N = int(math.ceil(len(data) / config.batch_size))  # 总共训练N个Batch

    clip = paddle.nn.ClipGradByGlobalNorm(clip_norm=config.max_grad_norm)
    optimizer = paddle.optimizer.SGD(learning_rate=lr,
                                     parameters=model.parameters(),
                                     grad_clip=clip)
    lossfn = nn.CrossEntropyLoss(reduction='sum')

    total_loss = 0

    if config.show:
        ProgressBar = getattr(import_module('utils'), 'ProgressBar')
        bar = ProgressBar('Train', max=N)

    for batch in range(N):
        if config.show:
            bar.next()

        optimizer.clear_grad()
        context = np.ndarray([config.batch_size, config.mem_size],
                             dtype=np.int64)
        target = np.ndarray([config.batch_size], dtype=np.int64)
        for i in range(config.batch_size):
            # 在原论文对应的实现中，这里采用的就是这种随机取样的方法
            # 这里的随机也许会导致模型不稳定
            # 我尝试过采用非随机的顺序取样，但得到的模型效果比随机取样的要差
            m = random.randrange(config.mem_size, len(data))
            target[i] = data[m]
            context[i, :] = data[m - config.mem_size: m]

        batch_data = paddle.to_tensor(context)
        batch_label = paddle.to_tensor(target)

        preict = model(batch_data)
        loss = lossfn(preict, batch_label)
        loss.backward()
        optimizer.step()
        total_loss += loss

    if config.show:
        bar.finish()

    return total_loss / N / config.batch_size


def train(model: MenN2N, train_data, valid_data, config):
    """
    完成训练
    """
    lr = config.init_lr

    train_losses = []
    train_perplexities = []

    valid_losses = []
    valid_perplexities = []

    for epoch in range(1, config.nepoch + 1):
        train_loss = train_single_step(model, lr, train_data, config)
        valid_loss = eval(model, valid_data, config, "Validation")

        info = {
            'epoch': epoch,
            'learning_rate': lr
        }

        # 当valid上的loss不再下降时，就像learning rate除以1.5
        if len(valid_losses) > 0 and valid_loss > valid_losses[-1] * 0.9999:
            lr /= 1.5

        train_losses.append(train_loss)
        train_perplexities.append(math.exp(train_loss))

        valid_losses.append(valid_loss)
        valid_perplexities.append(math.exp(valid_loss))

        info["train_perplexity"] = train_perplexities[-1]
        info["validate_perplexity"] = valid_perplexities[-1]

        print(info)

        if epoch % 5 == 0:
            save_dir = os.path.join(config.checkpoint_dir, "model_%d" % epoch)
            paddle.save(model.state_dict(), save_dir)
            lr_path = os.path.join(config.checkpoint_dir, "lr_%d" % epoch)
            with open(lr_path, "w") as f:
                f.write(f"{lr}")

        # 为了完成目标精度
        if info["validate_perplexity"] < 147.0:
            save_dir = os.path.join(config.checkpoint_dir, "model_good")
            paddle.save(model.state_dict(), save_dir)
            break

        if lr < 1e-5:
            break

    save_dir = os.path.join(config.checkpoint_dir, "model")
    paddle.save(model.state_dict(), save_dir)


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

    np.random.seed(config.srand)
    random.seed(config.srand)
    paddle.seed(config.srand)

    model = MenN2N(config)
    if config.recover_train:
        model_path = os.path.join(config.checkpoint_dir, config.model_name)
        state_dict = paddle.load(model_path)
        model.set_dict(state_dict)
    train(model, train_data, valid_data, config)
