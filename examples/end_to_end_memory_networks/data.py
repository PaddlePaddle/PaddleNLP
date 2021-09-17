import os


def read_data(fname, word2idx):
    """
    data被处理成了一个一维向量，每一个值都是一个单词对应的编码，
    2个句子之间采用特殊字符<eos>分隔。
    :param fname: 要读取的文件名
    :param word2idx: 词典
    :return: 处理好的词向量
    """
    if os.path.isfile(fname):
        with open(fname) as f:
            lines = f.readlines()
    else:
        raise (Exception("[!] Data %s not found" % fname))

    words = []
    for line in lines:
        words.extend(line.split())

    print("Read %s words from %s" % (len(words), fname))

    data = list()
    for line in lines:
        for word in line.split():
            index = word2idx[word]
            data.append(index)
        data.append(word2idx['<eos>'])
    return data


def load_vocab(fname):
    """
    加载词典
    :param fname: 词典存储名
    :return: 词典
    """
    word2idx = {}
    with open(fname, "r") as f:
        for line in f:
            pair = line.split()
            word2idx[pair[0]] = int(pair[1])
    return word2idx
