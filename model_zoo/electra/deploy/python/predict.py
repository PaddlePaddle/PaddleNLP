import argparse
import time
import numpy as np
import os

from paddle import inference
from paddlenlp.transformers import ElectraTokenizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_file",
                        type=str,
                        required=True,
                        help="model filename")
    parser.add_argument("--params_file",
                        type=str,
                        required=True,
                        help="parameter filename")
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
    parser.add_argument("--use_gpu",
                        action="store_true",
                        help="whether to use gpu")
    parser.add_argument("--use_trt",
                        action="store_true",
                        help="whether to use TensorRT")
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
        sentences (list[str]): each string is a sentence. If sentences not paths
        paths (list[str]): The paths of file which contain sentences. If paths not sentences
    Returns:
        res (list(numpy.ndarray)): The result of sentence, indicate whether each word is replaced, same shape with sentences.
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

    # config
    config = inference.Config(args.model_file, args.params_file)
    config.switch_use_feed_fetch_ops(False)
    config.enable_memory_optim()
    if args.use_gpu:
        config.enable_use_gpu(1000, 0)
    if args.use_trt:
        config.enable_tensorrt_engine(
            workspace_size=1 << 30,
            max_batch_size=args.batch_size,
            min_subgraph_size=5,
            precision_mode=inference.PrecisionType.Float32,
            use_static=False,
            use_calib_mode=False)

    # predictor
    predictor = inference.create_predictor(config)

    start_time = time.time()
    output_data = []
    count = 0
    for i, sen in enumerate(predicted_input):
        sen = np.array(sen).astype("int64")
        # get input name
        input_names = predictor.get_input_names()
        # get input pointer and copy data
        input_tensor = predictor.get_input_handle(input_names[0])
        input_tensor.copy_from_cpu(sen)

        # run predictor
        predictor.run()

        # get output name
        output_names = predictor.get_output_names()
        # get output pointer and copy data(nd.array)
        output_tensor = predictor.get_output_handle(output_names[0])
        predict_data = output_tensor.copy_to_cpu()
        output_res = np.argmax(predict_data, axis=1).tolist()
        output_data.append(output_res)

        print("===== batch {} =====".format(i))
        for j in range(len(predicted_sens[i])):
            print("Input sentence is : {}".format(predicted_sens[i][j]))
            #print("Output logis is : {}".format(output_data[j]))
            print("Output data is : {}".format(output_res[j]))
        count += len(predicted_sens[i])
    print("inference total %s sentences done, total time : %s s" %
          (count, time.time() - start_time))


if __name__ == "__main__":
    args = parse_args()
    sentences = args.predict_sentences
    paths = args.predict_file
    #sentences = ["The quick brown fox see over the lazy dog.", "The quick brown fox jump over tree lazy dog."]
    #paths = ["../../debug/test.txt", "../../debug/test.txt.1"]
    predict(args, sentences, paths)
