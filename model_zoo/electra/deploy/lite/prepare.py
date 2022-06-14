import argparse
import time
import numpy as np
import os
import io
import shutil
import fileinput

from paddlenlp.transformers import ElectraTokenizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lite_lib_path",
                        type=str,
                        required=True,
                        default=None,
                        help="directory of paddle lite api library")
    parser.add_argument("--lite_model_file",
                        type=str,
                        required=True,
                        default=None,
                        help="paddle lite model file(.nb)")
    parser.add_argument("--predict_sentences",
                        type=str,
                        nargs="*",
                        help="one or more sentence to predict")
    parser.add_argument(
        "--predict_file",
        type=str,
        nargs="*",
        help="one or more file which contain sentence to predict")
    parser.add_argument(
        "--prepared_file_prefix",
        type=str,
        default="predict_input",
        help="prepared file prefix after processing predict sentences")
    parser.add_argument("--batch_size",
                        type=int,
                        default=100000,
                        help="batch size")
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


def prepare_predict(args, sentences=[], paths=[]):
    """
    Args:
        sentences (list[str]): each string is a sentence. If sentences not paths
        paths (list[str]): The paths of file which contain sentences. If paths not sentences
    """

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

    predicted_input_np = np.array(predicted_input)
    predict_num = predicted_input_np.shape[1]
    predict_length = predicted_input_np.shape[2]
    predict_input_bin = args.prepared_file_prefix + ".bin"
    predict_input_txt = args.prepared_file_prefix + ".txt"
    predicted_input_np[0].astype(np.int64).tofile(predict_input_bin)
    with io.open(predict_input_txt, "w", encoding="UTF-8") as f:
        for sen_batch in predicted_sens:
            for sen in sen_batch:
                if len(sen.strip()) > 0:
                    f.write(sen.strip() + '\n')

    for line in fileinput.input("./deploy/lite/config.txt", inplace=True):
        if "predict_num" in line:
            newline = "predict_num " + str(predict_num)
            print("%s" % newline)
        elif "predict_length" in line:
            newline = "predict_length " + str(predict_length)
            print("%s" % newline)
        else:
            print("%s" % line.strip())

    root_dir = args.lite_lib_path + "/demo/cxx/electra/"
    debug_dir = args.lite_lib_path + "/demo/cxx/electra/debug/"
    if not os.path.exists(debug_dir):
        os.makedirs(debug_dir)
    shutil.copy(args.lite_model_file, debug_dir)
    shutil.copy("./deploy/lite/sst2_label.txt", debug_dir)
    shutil.copy("./deploy/lite/config.txt", debug_dir)
    shutil.copy(predict_input_bin, debug_dir)
    shutil.copy(predict_input_txt, debug_dir)
    libpaddle_light_api = os.path.join(args.lite_lib_path,
                                       "cxx/lib/libpaddle_light_api_shared.so")
    shutil.copy(libpaddle_light_api, debug_dir)

    shutil.copy("./deploy/lite/config.txt", root_dir)
    shutil.copy("./deploy/lite/sentiment_classfication.cpp", root_dir)
    shutil.copy("./deploy/lite/Makefile", root_dir)


if __name__ == "__main__":
    args = parse_args()
    sentences = args.predict_sentences
    paths = args.predict_file
    start_time = time.time()
    #sentences = ["The quick brown fox see over the lazy dog.", "The quick brown fox jump over tree lazy dog."]
    #paths = ["../../debug/test.txt", "../../debug/test.txt.1"]
    prepare_predict(args, sentences, paths)
    print("prepare lite predict done, total time : %s s" %
          (time.time() - start_time))
