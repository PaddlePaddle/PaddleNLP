#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
This file is used to train the model.
"""
import os
import sys
import math
import time
import random
import argparse

import numpy as np
import paddle
import paddle.fluid as fluid

import reader
from network import lex_net
from bilm import init_pretraining_params


def parse_args():
    """
    Parsing the input parameters.
    """
    parser = argparse.ArgumentParser("Training for lexical analyzer.")
    parser.add_argument(
        "--traindata_dir",
        type=str,
        default="data/train_data",
        help="The folder where the training data is located.")
    parser.add_argument(
        "--testdata_dir",
        type=str,
        default="data/test_data",
        help="The folder where the training data is located.")
    parser.add_argument(
        "--model_save_dir",
        type=str,
        default="./models",
        help="The model will be saved in this path.")
    parser.add_argument(
        "--save_model_per_batchs",
        type=int,
        default=1000,
        help="Save the model once per xxxx batch of training")
    parser.add_argument(
        "--eval_window",
        type=int,
        default=20,
        help="Training will be suspended when the evaluation indicators on the validation set" \
             " no longer increase. The eval_window specifies the scope of the evaluation.")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="The number of sequences contained in a mini-batch, or the maximum" \
             "number of tokens (include paddings) contained in a mini-batch.")
    parser.add_argument(
        "--corpus_type_list",
        type=str,
        default=["human", "feed", "query", "title", "news"],
        nargs='+',
        help="The pattern list of different types of corpus used in training.")
    parser.add_argument(
        "--corpus_proportion_list",
        type=float,
        default=[0.2, 0.2, 0.2, 0.2, 0.2],
        nargs='+',
        help="The proportion list of different types of corpus used in training."
    )
    parser.add_argument(
        "--use_gpu",
        type=int,
        default=False,
        help="Whether or not to use GPU. 0-->CPU 1-->GPU")
    parser.add_argument(
        "--traindata_shuffle_buffer",
        type=int,
        default=200000,
        help="The buffer size used in shuffle the training data.")
    parser.add_argument(
        "--word_emb_dim",
        type=int,
        default=128,
        help="The dimension in which a word is embedded.")
    parser.add_argument(
        "--grnn_hidden_dim",
        type=int,
        default=256,
        help="The number of hidden nodes in the GRNN layer.")
    parser.add_argument(
        "--bigru_num",
        type=int,
        default=2,
        help="The number of bi_gru layers in the network.")
    parser.add_argument(
        "--base_learning_rate",
        type=float,
        default=1e-3,
        help="The basic learning rate that affects the entire network.")
    parser.add_argument(
        "--emb_learning_rate",
        type=float,
        default=5,
        help="The real learning rate of the embedding layer will be" \
        " (emb_learning_rate * base_learning_rate)."
    )
    parser.add_argument(
        "--crf_learning_rate",
        type=float,
        default=0.2,
        help="The real learning rate of the embedding layer will be" \
             " (crf_learning_rate * base_learning_rate)."
    )
    parser.add_argument(
        "--word_dict_path",
        type=str,
        default="../data/vocabulary_min5k.txt",
        help="The path of the word dictionary.")
    parser.add_argument(
        "--label_dict_path",
        type=str,
        default="data/tag.dic",
        help="The path of the label dictionary.")
    parser.add_argument(
        "--word_rep_dict_path",
        type=str,
        default="conf/q2b.dic",
        help="The path of the word replacement Dictionary.")
    parser.add_argument(
        "--num_iterations",
        type=int,
        default=40000,
        help="The maximum number of iterations. If set to 0 (default), do not limit the number."
    )

    #add elmo args
    parser.add_argument(
        "--elmo_l2_coef",
        type=float,
        default=0.001,
        help="Weight decay. (default: %(default)f)")
    parser.add_argument(
        "--elmo_dict_dir",
        default='data/vocabulary_min5k.txt',
        help="If set, load elmo dict.")
    parser.add_argument(
        '--pretrain_elmo_model_path',
        default="data/baike_elmo_checkpoint",
        help="If set, load elmo checkpoint.")
    args = parser.parse_args()
    if len(args.corpus_proportion_list) != len(args.corpus_type_list):
        sys.stderr.write(
            "The length of corpus_proportion_list should be equal to the length of corpus_type_list.\n"
        )
        exit(-1)
    return args


def print_arguments(args):
    print('-----------  Configuration Arguments -----------')
    for arg, value in sorted(vars(args).items()):
        print('%s: %s' % (arg, value))
    print('------------------------------------------------')


def to_lodtensor(data, place):
    """
    Convert data in list into lodtensor.
    """
    seq_lens = [len(seq) for seq in data]
    cur_len = 0
    lod = [cur_len]
    for l in seq_lens:
        cur_len += l
        lod.append(cur_len)
    flattened_data = np.concatenate(data, axis=0).astype("int64")
    flattened_data = flattened_data.reshape([len(flattened_data), 1])
    res = fluid.LoDTensor()
    res.set(flattened_data, place)
    res.set_lod([lod])
    return res


def test(exe, chunk_evaluator, save_dirname, test_data, place):
    """
    Test the network in training.
    """
    inference_scope = fluid.core.Scope()
    with fluid.scope_guard(inference_scope):
        [inference_program, feed_target_names,
         fetch_targets] = fluid.io.load_inference_model(save_dirname, exe)
        chunk_evaluator.reset()
        for data in test_data():
            word = to_lodtensor(list(map(lambda x: x[0], data)), place)
            target = to_lodtensor(list(map(lambda x: x[1], data)), place)
            result_list = exe.run(inference_program,
                                  feed={"word": word,
                                        "target": target},
                                  fetch_list=fetch_targets)
            number_infer = np.array(result_list[0])
            number_label = np.array(result_list[1])
            number_correct = np.array(result_list[2])
            chunk_evaluator.update(
                int(number_infer[0]),
                int(number_label[0]), int(number_correct[0]))
    return chunk_evaluator.eval()


def train(args):
    """
    Train the network.
    """
    if not os.path.exists(args.model_save_dir):
        os.mkdir(args.model_save_dir)

    word2id_dict = reader.load_reverse_dict(args.word_dict_path)
    label2id_dict = reader.load_reverse_dict(args.label_dict_path)
    word_rep_dict = reader.load_dict(args.word_rep_dict_path)
    word_dict_len = max(map(int, word2id_dict.values())) + 1
    label_dict_len = max(map(int, label2id_dict.values())) + 1

    avg_cost, crf_decode, word, target = lex_net(args, word_dict_len,
                                                 label_dict_len)
    adam_optimizer = fluid.optimizer.Adam(learning_rate=args.base_learning_rate)
    adam_optimizer.minimize(avg_cost)

    (precision, recall, f1_score, num_infer_chunks, num_label_chunks,
     num_correct_chunks) = fluid.layers.chunk_eval(
         input=crf_decode,
         label=target,
         chunk_scheme="IOB",
         num_chunk_types=int(math.ceil((label_dict_len - 1) / 2.0)))
    chunk_evaluator = fluid.metrics.ChunkEvaluator()
    chunk_evaluator.reset()

    train_reader_list = []
    corpus_num = len(args.corpus_type_list)
    for i in range(corpus_num):
        train_reader = paddle.batch(
            paddle.reader.shuffle(
                reader.file_reader(args.traindata_dir, word2id_dict,
                                   label2id_dict, word_rep_dict,
                                   args.corpus_type_list[i]),
                buf_size=args.traindata_shuffle_buffer),
            batch_size=int(args.batch_size * args.corpus_proportion_list[i]))
        train_reader_list.append(train_reader)
    test_reader = paddle.batch(
        reader.file_reader(args.testdata_dir, word2id_dict, label2id_dict,
                           word_rep_dict),
        batch_size=args.batch_size)
    train_reader_itr_list = []
    for train_reader in train_reader_list:
        cur_reader_itr = train_reader()
        train_reader_itr_list.append(cur_reader_itr)

    place = fluid.CUDAPlace(0) if args.use_gpu else fluid.CPUPlace()
    feeder = fluid.DataFeeder(feed_list=[word, target], place=place)
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())

    # load pretrained ELMo model
    init_pretraining_params(exe, args.pretrain_elmo_model_path,
                            fluid.default_main_program())

    batch_id = 0
    start_time = time.time()
    eval_list = []
    iter = 0
    while True:
        full_batch = []
        cur_batch = []
        for i in range(corpus_num):
            reader_itr = train_reader_itr_list[i]
            try:
                cur_batch = next(reader_itr)
            except StopIteration:
                print(args.corpus_type_list[i] +
                      " corpus finish a pass of training")
                new_reader = train_reader_list[i]
                train_reader_itr_list[i] = new_reader()
                cur_batch = next(train_reader_itr_list[i])
            full_batch += cur_batch
        random.shuffle(full_batch)

        cost_var, nums_infer, nums_label, nums_correct = exe.run(
            fluid.default_main_program(),
            fetch_list=[
                avg_cost, num_infer_chunks, num_label_chunks, num_correct_chunks
            ],
            feed=feeder.feed(full_batch))
        print("batch_id:" + str(batch_id) + ", avg_cost:" + str(cost_var[0]))
        chunk_evaluator.update(nums_infer, nums_label, nums_correct)
        batch_id += 1

        if (batch_id % args.save_model_per_batchs == 1):
            save_exe = fluid.Executor(place)
            save_dirname = os.path.join(args.model_save_dir,
                                        "params_batch_%d" % batch_id)

            temp_save_model = os.path.join(args.model_save_dir,
                                           "temp_model_for_test")
            fluid.io.save_inference_model(
                temp_save_model, ['word', 'target'],
                [num_infer_chunks, num_label_chunks, num_correct_chunks],
                save_exe)

            precision, recall, f1_score = chunk_evaluator.eval()
            print("[train] batch_id:" + str(batch_id) + ", precision:" + str(
                precision) + ", recall:" + str(recall) + ", f1:" + str(
                    f1_score))
            chunk_evaluator.reset()
            p, r, f1 = test(exe, chunk_evaluator, temp_save_model, test_reader,
                            place)
            chunk_evaluator.reset()
            print("[test] batch_id:" + str(batch_id) + ", precision:" + str(p) +
                  ", recall:" + str(r) + ", f1:" + str(f1))
            end_time = time.time()
            print("cur_batch_id:" + str(batch_id) + ", last " + str(
                args.save_model_per_batchs) + " batchs, time_cost:" + str(
                    end_time - start_time))
            start_time = time.time()

            if len(eval_list) < 2 * args.eval_window:
                eval_list.append(f1)
            else:
                eval_list.pop(0)
                eval_list.append(f1)
                last_avg_f1 = sum(eval_list[
                    0:args.eval_window]) / args.eval_window
                cur_avg_f1 = sum(eval_list[args.eval_window:2 *
                                           args.eval_window]) / args.eval_window
                if cur_avg_f1 <= last_avg_f1:
                    return
                else:
                    print("keep training!")
        iter += 1
        if (iter == args.num_iterations):
            return


if __name__ == "__main__":
    import paddle
    paddle.enable_static()
    args = parse_args()
    print_arguments(args)
    train(args)
