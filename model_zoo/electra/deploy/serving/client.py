import argparse
import time
import numpy as np
import os

import paddle
from paddle_serving_client import Client

from paddlenlp.transformers import ElectraTokenizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--client_config_file",
                        type=str,
                        required=True,
                        help="client prototxt config file")
    parser.add_argument("--server_ip_port",
                        type=str,
                        default="127.0.0.1:8383",
                        help="server_ip:port")
    parser.add_argument("--predict_sentences",
                        type=str,
                        nargs="*",
                        help="one or more sentence to predict")
    parser.add_argument(
        "--predict_file",
        type=str,
        nargs="*",
        help="one or more file which contain sentence to predict")
    parser.add_argument("--batch_size", type=int, default=1, help="batch size")
    parser.add_argument("--max_seq_length",
                        type=int,
                        default=128,
                        help="max length of each sequence")
    parser.add_argument(
        "--model_name",
        type=str,
        default="electra-small",
        help="shortcut name selected in the list: " +
        ", ".join(list(ElectraTokenizer.pretrained_init_configuration.keys())))
    return parser.parse_args()


def read_sentences(paths=[]):
    sentences = []
    for sen_path in paths:
        assert os.path.isfile(sen_path), "The {} isn't a valid file.".format(
            sen_path)
        sen = read_file(sen_path)
        if sen is None:
            logger.info("error in loading file:{}".format(sen_path))
            continue
        sentences.extend(sen)
    return sentences


def read_file(path):
    lines = []
    with open(path, encoding="utf-8") as f:
        while True:
            line = f.readline()
            if line:
                if (len(line) > 0 and not line.isspace()):
                    lines.append(line.strip())
            else:
                break
    return lines


def get_predicted_input(predicted_data, tokenizer, max_seq_length, batch_size):
    if predicted_data == [] or not isinstance(predicted_data, list):
        raise TypeError("The predicted_data is inconsistent with expectations.")

    sen_ids_batch = []
    sen_words_batch = []
    sen_ids = []
    sen_words = []
    batch_num = 0
    pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    for sen in predicted_data:
        sen_id = tokenizer(sen, max_seq_len=max_seq_length)['input_ids']
        sen_ids.append(sen_id)
        sen_words.append(tokenizer.cls_token + " " + sen + " " +
                         tokenizer.sep_token)
        batch_num += 1
        if batch_num == batch_size:
            tmp_list = []
            max_length = max([len(i) for i in sen_ids])
            for i in sen_ids:
                if len(i) < max_length:
                    tmp_list.append(i + (max_length - len(i)) * [pad_token_id])
                else:
                    tmp_list.append(i)
            sen_ids_batch.append(tmp_list)
            sen_words_batch.append(sen_words)
            sen_ids = []
            sen_words = []
            batch_num = 0

    if len(sen_ids) > 0:
        tmp_list = []
        max_length = max([len(i) for i in sen_ids])
        for i in sen_ids:
            if len(i) < max_length:
                tmp_list.append(i + (max_length - len(i)) * [pad_token_id])
            else:
                tmp_list.append(i)
        sen_ids_batch.append(tmp_list)
        sen_words_batch.append(sen_words)

    return sen_ids_batch, sen_words_batch


def predict(args, sentences=[], paths=[]):
    """
    Args:
        sentences (list[str]): each string is a sentence. If have sentences then no need paths
        paths (list[str]): The paths of file which contain sentences. If have paths then no need sentences
    Returns:
        res (list(numpy.ndarray)): The result of sentence, indicate whether each word is replaced, same shape with sentences.
    """

    # initialize client
    client = Client()
    client.load_client_config(args.client_config_file)
    #"serving_client/serving_client_conf.prototxt")
    client.connect([args.server_ip_port])

    # initialize data
    if sentences != [] and isinstance(sentences, list) and (paths == []
                                                            or paths is None):
        predicted_data = sentences
    elif (sentences == [] or sentences is None) and isinstance(
            paths, list) and paths != []:
        predicted_data = read_sentences(paths)
    else:
        raise TypeError("The input data is inconsistent with expectations.")

    tokenizer = ElectraTokenizer.from_pretrained(args.model_name)
    predicted_input, predicted_sens = get_predicted_input(
        predicted_data, tokenizer, args.max_seq_length, args.batch_size)

    start_time = time.time()
    count = 0
    for i, sen in enumerate(predicted_input):
        sen = np.array(sen).astype("int64")

        fetch_map = client.predict(feed={"input_ids": sen},
                                   fetch=["save_infer_model/scale_0.tmp_0"],
                                   batch=True)
        output_data = np.array(fetch_map["save_infer_model/scale_0.tmp_0"])
        output_res = np.argmax(output_data, axis=1)

        print("===== batch {} =====".format(i))
        for j in range(len(predicted_sens[i])):
            print("Input sentence is : {}".format(predicted_sens[i][j]))
            #print("Output logis is : {}".format(output_data[j]))
            print("Output data is : {}".format(output_res[j]))

        count += len(predicted_sens[i])
    print("inference total %s sentences done, total time : %s s" %
          (count, time.time() - start_time))


if __name__ == '__main__':
    #paddle.enable_static()
    args = parse_args()
    sentences = args.predict_sentences
    paths = args.predict_file
    predict(args, sentences, paths)
