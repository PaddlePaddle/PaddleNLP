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
"""Fine-tuning on SQuAD."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
import sys
if six.PY2:
    reload(sys)
    sys.setdefaultencoding('utf8')

import io
import argparse
import collections
import multiprocessing
import os
import time
import numpy as np
import json
import paddle
import paddle.fluid as fluid

from reader.squad import DataProcessor, write_predictions
from model.xlnet import XLNetConfig, XLNetModel
from utils.args import ArgumentGroup, print_arguments
from optimization import optimization
from utils.init import init_pretraining_params, init_checkpoint
from modeling import log_softmax

if six.PY2:
    import cPickle as pickle
else:
    import pickle

# yapf: disable
parser = argparse.ArgumentParser(__doc__)
model_g = ArgumentGroup(parser, "model", "model configuration and paths.")
model_g.add_arg("model_config_path",        str,  None,           "Path to the json file for xlnet model config.")
model_g.add_arg("dropout",                  float,  0.1,          "Dropout rate.")
model_g.add_arg("dropatt",                  float,  0.1,          "Attention dropout rate.")
model_g.add_arg("clamp_len",                int,    -1,           "Clamp length.")
model_g.add_arg("summary_type",             str, "last",           "Method used to summarize a sequence into a vector.",
                choices=['last'])
model_g.add_arg("spiece_model_file",        str,  None,           "Sentence Piece model path.")
model_g.add_arg("init_checkpoint",          str,  None,           "Init checkpoint to resume training from.")
model_g.add_arg("init_pretraining_params",  str,  None,
                "Init pre-training params which preforms fine-tuning from. If the "
                 "arg 'init_checkpoint' has been set, this argument wouldn't be valid.")
model_g.add_arg("checkpoints",              str,  "checkpoints",  "Path to save checkpoints.")

# Parameter initialization
init_g = ArgumentGroup(parser, "init", "parameter initialization options.")
init_g.add_arg("init",       str, "normal",    "Initialization method.", choices=["normal", "uniform"])
init_g.add_arg("init_std",   str, 0.02,    "Initialization std when init is normal.")
init_g.add_arg("init_range", str, 0.1,   "Initialization std when init is uniform.")

train_g = ArgumentGroup(parser, "training", "training options.")
train_g.add_arg("epoch",               int,    3,      "Number of epoches for fine-tuning.")
train_g.add_arg("learning_rate",       float,  5e-5,   "Learning rate used to train with warmup.")
train_g.add_arg("lr_scheduler",        str,    "linear_warmup_decay",
                "scheduler of learning rate.", choices=['linear_warmup_decay', 'noam_decay'])
train_g.add_arg("weight_decay",        float,  0.01,   "Weight decay rate for L2 regularizer.")
train_g.add_arg("adam_epsilon",        float,  1e-6,   "Adam epsilon.")
train_g.add_arg("lr_layer_decay_rate", float,  0.75, "Top layer: lr[L] = args.learning_rate. "
                                                     "Lower layers: lr[l-1] = lr[l] * lr_layer_decay_rate.")
train_g.add_arg("train_batch_size",  int,    12,     "Total examples' number in batch for training.")
train_g.add_arg("train_steps",       int,    1000,   "The total steps for training.")
train_g.add_arg("warmup_steps",      int,    1000,   "The steps for warmup.")
train_g.add_arg("save_steps",        int,    1000,   "The steps interval to save checkpoints.")

predict_g = ArgumentGroup(parser, "prediction", "prediction options.")
predict_g.add_arg("predict_batch_size",                int,   12,    "Total examples' number in batch for training.")
predict_g.add_arg("start_n_top",    int,  5, "Beam size for span start.")
predict_g.add_arg("end_n_top",      int,  5, "Beam size for span end.")

log_g = ArgumentGroup(parser, "logging", "logging related.")
log_g.add_arg("skip_steps",          int,    10,    "The steps interval to print loss.")
log_g.add_arg("verbose",             bool,   False, "Whether to output verbose log.")

data_g = ArgumentGroup(parser, "data", "Data paths, vocab paths and data processing options")
data_g.add_arg("train_file",                str,   None,  "SQuAD json for training. E.g., train-v1.1.json.")
data_g.add_arg("predict_file",              str,   None,  "SQuAD json for predictions. E.g. dev-v1.1.json or test-v1.1.json.")
data_g.add_arg("max_seq_length",               int,   512,   "Number of words of the longest seqence.")
data_g.add_arg("max_query_length",          int,   64,    "Max query length.")
data_g.add_arg("max_answer_length",         int,   64,    "Max answer length.")
data_g.add_arg("uncased",             bool,  True,
               "Whether to lower case the input text. Should be True for uncased models and False for cased models.")
data_g.add_arg("doc_stride",                int,   128,
               "When splitting up a long document into chunks, how much stride to take between chunks.")
data_g.add_arg("n_best_size",               int,   5,
               "The total number of n-best predictions to generate in the nbest_predictions.json output file.")
data_g.add_arg("random_seed",               int,   0,      "Random seed.")

run_type_g = ArgumentGroup(parser, "run_type", "running type options.")
run_type_g.add_arg("use_cuda",                     bool,   True,  "If set, use GPU for training.")
run_type_g.add_arg("use_fast_executor",            bool,   False, "If set, use fast parallel executor (in experiment).")
run_type_g.add_arg("num_iteration_per_drop_scope", int,    1,     "Ihe iteration intervals to clean up temporary variables.")
run_type_g.add_arg("do_train",                     bool,   True,  "Whether to perform training.")
run_type_g.add_arg("do_predict",                   bool,   True,  "Whether to perform prediction.")

args = parser.parse_args()
# yapf: enable.

def get_qa_outputs(xlnet_config, features, is_training=False):

    # (qlen, batch size)
    input_ids = features['input_ids']
    cls_index = features['cls_index']
    segment_ids = features['segment_ids']
    input_mask = features['input_mask']
    p_mask = features['p_mask']
    inp = fluid.layers.transpose(input_ids, perm=[1, 0, 2])
    inp_mask = fluid.layers.transpose(input_mask, perm=[1, 0])
    cls_index = fluid.layers.reshape(cls_index, shape=[-1, 1])

    seq_len = inp.shape[0]

    xlnet = XLNetModel(
        input_ids=inp,
        seg_ids=segment_ids,
        input_mask=inp_mask,
        xlnet_config=xlnet_config,
        args=args)

    output = xlnet.get_sequence_output()
    initializer = xlnet.get_initializer()

    return_dict = {}

    # logit of the start position
    start_logits = fluid.layers.fc(
        input=output,
        num_flatten_dims=2,
        size=1,
        param_attr=fluid.ParamAttr(name='start_logits_fc_weight', initializer=initializer),
        bias_attr='start_logits_fc_bias')
    start_logits = fluid.layers.transpose(fluid.layers.squeeze(start_logits, [-1]), [1, 0])
    start_logits_masked = start_logits * (1 - p_mask) - 1e30 * p_mask
    start_log_probs = log_softmax(start_logits_masked)

    # logit of the end position
    if is_training:
        start_positions = features['start_positions']
        start_index = fluid.layers.one_hot(start_positions, depth=args.max_seq_length)
        # lbh,bl->bh
        trans_out = fluid.layers.transpose(output, perm=[1, 2, 0])
        start_index = fluid.layers.unsqueeze(start_index, axes=[2])
        start_features = fluid.layers.matmul(x=trans_out, y=start_index)

        start_features = fluid.layers.unsqueeze(start_features, axes=[0])
        start_features = fluid.layers.squeeze(start_features, axes=[3])
        start_features = fluid.layers.expand(start_features, [seq_len, 1, 1])

        end_logits = fluid.layers.fc(
          input=fluid.layers.concat([output, start_features], axis=-1),
          num_flatten_dims=2,
          size=xlnet_config['d_model'],
          param_attr=fluid.ParamAttr(name="end_logits_fc_0_weight",initializer=initializer),
          bias_attr="end_logits_fc_0_bias",
          act='tanh')
        end_logits = fluid.layers.layer_norm(end_logits,
                         epsilon=1e-12,
                         param_attr=fluid.ParamAttr(
                           name='end_logits_layer_norm_scale',
                           initializer=fluid.initializer.Constant(1.)),
                         bias_attr=fluid.ParamAttr(
                           name='end_logits_layer_norm_bias',
                           initializer=fluid.initializer.Constant(0.)),
                         begin_norm_axis=len(end_logits.shape)-1)

        end_logits = fluid.layers.fc(
            input=end_logits,
            num_flatten_dims=2,
            size=1,
            param_attr=fluid.ParamAttr(name='end_logits_fc_1_weight', initializer=initializer),
            bias_attr='end_logits_fc_1_bias')
        end_logits = fluid.layers.transpose(fluid.layers.squeeze(end_logits, [-1]), [1, 0])
        end_logits_masked = end_logits * (1 - p_mask) - 1e30 * p_mask
        end_log_probs = log_softmax(end_logits_masked)
    else:
        start_top_log_probs, start_top_index = fluid.layers.topk(start_log_probs, k=args.start_n_top)
        start_top_index = fluid.layers.unsqueeze(start_top_index, [-1])
        start_index = fluid.layers.one_hot(start_top_index, seq_len)
        # lbh,bkl->bkh
        trans_out = fluid.layers.transpose(output, perm=[1, 2, 0])
        trans_start_index = fluid.layers.transpose(start_index, [0, 2, 1])
        start_features = fluid.layers.matmul(x=trans_out, y=trans_start_index)
        start_features = fluid.layers.transpose(start_features, [0, 2, 1])

        end_input = fluid.layers.expand(fluid.layers.unsqueeze(output, [2]), [1, 1, args.start_n_top, 1])
        start_features = fluid.layers.expand(fluid.layers.unsqueeze(start_features, [0]), [seq_len, 1, 1, 1])
        end_input = fluid.layers.concat([end_input, start_features], axis=-1)


        end_logits = fluid.layers.fc(end_input, size=xlnet_config['d_model'],
                                     num_flatten_dims=3,
                                     param_attr=fluid.ParamAttr(name="end_logits_fc_0_weight", initializer=initializer),
                                     bias_attr="end_logits_fc_0_bias",
                                     act='tanh')
        end_logits = fluid.layers.layer_norm(end_logits,
                         epsilon=1e-12,
                         param_attr=fluid.ParamAttr(
                           name='end_logits_layer_norm_scale',
                           initializer=fluid.initializer.Constant(1.)),
                         bias_attr=fluid.ParamAttr(
                           name='end_logits_layer_norm_bias',
                           initializer=fluid.initializer.Constant(0.)),
                         begin_norm_axis=len(end_logits.shape)-1)
        end_logits = fluid.layers.fc(
            input=end_logits,
            num_flatten_dims=3,
            size=1,
            param_attr=fluid.ParamAttr(name='end_logits_fc_1_weight', initializer=initializer),
            bias_attr='end_logits_fc_1_bias')

        end_logits = fluid.layers.reshape(end_logits, [seq_len, -1, args.start_n_top])
        end_logits = fluid.layers.transpose(end_logits, [1, 2, 0])
        p_mask = fluid.layers.stack([p_mask]*args.start_n_top, axis=1)
        end_logits_masked = end_logits * (1 - p_mask) - 1e30 * p_mask
        end_log_probs = log_softmax(end_logits_masked)
        end_top_log_probs, end_top_index = fluid.layers.topk(end_log_probs, k=args.end_n_top)
        end_top_log_probs = fluid.layers.reshape(
          end_top_log_probs,
          [-1, args.start_n_top * args.end_n_top])
        end_top_index = fluid.layers.reshape(
          end_top_index,
          [-1, args.start_n_top * args.end_n_top])

    if is_training:
        return_dict["start_log_probs"] = start_log_probs
        return_dict["end_log_probs"] = end_log_probs
    else:
        return_dict["start_top_log_probs"] = start_top_log_probs
        return_dict["start_top_index"] = start_top_index
        return_dict["end_top_log_probs"] = end_top_log_probs
        return_dict["end_top_index"] = end_top_index

    cls_index = fluid.layers.one_hot(cls_index, seq_len)
    cls_index = fluid.layers.unsqueeze(cls_index, axes=[2])
    cls_feature = fluid.layers.matmul(x=trans_out, y=cls_index)

    start_p = fluid.layers.softmax(start_logits_masked)
    start_p = fluid.layers.unsqueeze(start_p, axes=[2])
    start_feature = fluid.layers.matmul(x=trans_out, y=start_p)

    ans_feature = fluid.layers.concat([start_feature, cls_feature], axis=1)
    ans_feature = fluid.layers.fc(
      input=ans_feature,
                    size=xlnet_config['d_model'],
                    act='tanh',
                    param_attr=fluid.ParamAttr(initializer=initializer, name="answer_class_fc_0_weight"),
                    bias_attr="answer_class_fc_0_bias")
    ans_feature = fluid.layers.dropout(ans_feature, args.dropout)
    cls_logits = fluid.layers.fc(
        ans_feature,
        size=1,
        param_attr=fluid.ParamAttr(name='answer_class_fc_1_weight', initializer=initializer),
        bias_attr=False)
    cls_logits = fluid.layers.squeeze(cls_logits, axes=[-1])

    return_dict["cls_logits"] = cls_logits

    return return_dict

def create_model(xlnet_config, is_training=False):
    if is_training:
        input_fields = {
            'names': ['input_ids', 'segment_ids', 'input_mask', 'cls_index', 'p_mask',
                       'start_positions', 'end_positions', 'is_impossible'],
            'shapes': [[None, args.max_seq_length, 1], [None, args.max_seq_length],
                       [None, args.max_seq_length], [None, 1],
                       [None, args.max_seq_length], [None, 1], [None, 1], [None, 1]],
            'dtypes': [
                'int64', 'int64', 'float32', 'int64',
                'float32', 'int64', 'int64', 'float32'],
            'lod_levels': [0, 0, 0, 0, 0, 0, 0, 0]
            }
    else:
        input_fields = {
            'names': ['input_ids', 'segment_ids', 'input_mask', 'cls_index', 'p_mask', 'unique_ids'],
            'shapes': [[None, args.max_seq_length, 1], [None, args.max_seq_length],
                       [None, args.max_seq_length], [None, 1], [None, args.max_seq_length], [None, 1]],
            'dtypes': [
                'int64', 'int64', 'float32', 'int64', 'float32', 'int64'],
            'lod_levels': [0, 0, 0, 0, 0, 0],
        }

    inputs = [fluid.layers.data(name=input_fields['names'][i],
                      shape=input_fields['shapes'][i],
                      dtype=input_fields['dtypes'][i],
                      lod_level=input_fields['lod_levels'][i]) for i in range(len(input_fields['names']))]

    data_loader = fluid.io.DataLoader.from_generator(feed_list=inputs, capacity=50, iterable=False)
    if is_training:
        (input_ids, segment_ids, input_mask, cls_index, p_mask, start_positions,
         end_positions, is_impossible) = inputs
    else:
        (input_ids, segment_ids, input_mask, cls_index, p_mask, unique_ids) = inputs

    features = {'input_ids': input_ids, 'segment_ids': segment_ids, 'input_mask': input_mask, 'cls_index': cls_index, 'p_mask':p_mask}
    if is_training:
        features['start_positions'] = start_positions
        features['end_positions'] = end_positions
        features['is_impossible'] = is_impossible
    else:
        features['unique_ids'] = unique_ids

    outputs = get_qa_outputs(xlnet_config, features, is_training=is_training)

    if not is_training:
        predictions = {
          "unique_ids": features["unique_ids"],
          "start_top_index": outputs["start_top_index"],
          "start_top_log_probs": outputs["start_top_log_probs"],
          "end_top_index": outputs["end_top_index"],
          "end_top_log_probs": outputs["end_top_log_probs"],
          "cls_logits": outputs["cls_logits"]
        }
        return data_loader, predictions

    seq_len = input_ids.shape[1]

    def compute_loss(log_probs, positions):
        one_hot_positions = fluid.layers.one_hot(positions, depth=seq_len)

        loss = -1 * fluid.layers.reduce_sum(one_hot_positions * log_probs, dim=-1)
        loss = fluid.layers.reduce_mean(loss)
        return loss

    start_loss = compute_loss(
        outputs["start_log_probs"], features["start_positions"])
    end_loss = compute_loss(
        outputs["end_log_probs"], features["end_positions"])

    total_loss = (start_loss + end_loss) * 0.5

    cls_logits = outputs["cls_logits"]
    is_impossible = fluid.layers.reshape(features["is_impossible"], [-1])
    regression_loss = fluid.layers.sigmoid_cross_entropy_with_logits(
        label=is_impossible, x=cls_logits)
    regression_loss = fluid.layers.reduce_mean(regression_loss)

    total_loss += regression_loss * 0.5

    return data_loader, total_loss

RawResult = collections.namedtuple("RawResult",
    ["unique_id", "start_top_log_probs", "start_top_index",
    "end_top_log_probs", "end_top_index", "cls_logits"])

def predict(test_exe, test_program, test_data_loader, fetch_list, processor, name):
    if not os.path.exists(args.checkpoints):
        os.makedirs(args.checkpoints)
    output_prediction_file = os.path.join(args.checkpoints, name + "predictions.json")
    output_nbest_file = os.path.join(args.checkpoints, name + "nbest_predictions.json")
    output_null_log_odds_file = os.path.join(args.checkpoints, name + "null_odds.json")

    test_data_loader.start()
    all_results = []
    time_begin = time.time()
    while True:
        try:
            outputs = test_exe.run(
                fetch_list=fetch_list,
                program=test_program)
            np_unique_ids, np_start_top_log_probs, np_start_top_index, np_end_top_log_probs, np_end_top_index,  np_cls_logits, \
                      = outputs[0:6]

            for idx in range(np_unique_ids.shape[0]):
                if len(all_results) % 1000 == 0:
                    print("Processing example: %d" % len(all_results))
                unique_id = int(np_unique_ids[idx])
                start_top_log_probs = [float(x) for x in np_start_top_log_probs[idx].flat]
                start_top_index = [int(x) for x in np_start_top_index[idx].flat]
                end_top_log_probs = [float(x) for x in np_end_top_log_probs[idx].flat]
                end_top_index = [int(x) for x in np_end_top_index[idx].flat]
                cls_logits = float(np_cls_logits[idx].flat[0])

                all_results.append(
                    RawResult(
                        unique_id=unique_id,
                        start_top_log_probs=start_top_log_probs,
                        start_top_index=start_top_index,
                        end_top_log_probs=end_top_log_probs,
                        end_top_index=end_top_index,
                        cls_logits=cls_logits))
        except fluid.core.EOFException:
            test_data_loader.reset()
            break
    time_end = time.time()

    with io.open(args.predict_file, "r", encoding="utf8") as f:
        orig_data = json.load(f)["data"]

    features = processor.get_features(
        processor.predict_examples, is_training=False)
    ret = write_predictions(processor.predict_examples, features, all_results,
                      args.n_best_size, args.max_answer_length,
                      output_prediction_file,
                      output_nbest_file, output_null_log_odds_file,
                      orig_data,  args)
    # Log current result
    print("=" * 80)
    log_str = "Result | "
    for key, val in ret.items():
        log_str += "{} {} | ".format(key, val)
    print(log_str)
    print("=" * 80)


def train(args):
    if not (args.do_train or args.do_predict):
        raise ValueError("For args `do_train` and `do_predict`, at "
                         "least one of them must be True.")

    xlnet_config = XLNetConfig(args.model_config_path)
    xlnet_config.print_config()

    if args.use_cuda:
        place = fluid.CUDAPlace(0)
        dev_count = fluid.core.get_cuda_device_count()
    else:
        place = fluid.CPUPlace()
        dev_count = int(os.environ.get('CPU_NUM', multiprocessing.cpu_count()))
    exe = fluid.Executor(place)

    processor = DataProcessor(
        spiece_model_file=args.spiece_model_file,
        uncased=args.uncased,
        max_seq_length=args.max_seq_length,
        doc_stride=args.doc_stride,
        max_query_length=args.max_query_length)


    startup_prog = fluid.Program()
    if args.random_seed is not None:
        startup_prog.random_seed = args.random_seed

    if args.do_train:
        train_data_generator = processor.data_generator(
            data_path=args.train_file,
            batch_size=args.train_batch_size,
            phase='train',
            shuffle=True,
            dev_count=dev_count,
            epoch=args.epoch)

        num_train_examples = processor.get_num_examples(phase='train')
        print("Device count: %d" % dev_count)
        print("Max num of epoches: %d" % args.epoch)
        print("Num of train examples: %d" % num_train_examples)
        print("Num of train steps: %d" % args.train_steps)
        print("Num of warmup steps: %d" % args.warmup_steps)

        train_program = fluid.Program()
        with fluid.program_guard(train_program, startup_prog):
            with fluid.unique_name.guard():
                train_data_loader, loss  = create_model(
                    xlnet_config=xlnet_config,
                    is_training=True)

                scheduled_lr = optimization(
                    loss=loss,
                    warmup_steps=args.warmup_steps,
                    num_train_steps=args.train_steps,
                    learning_rate=args.learning_rate,
                    train_program=train_program,
                    startup_prog=startup_prog,
                    weight_decay=args.weight_decay,
                    lr_layer_decay_rate=args.lr_layer_decay_rate,
                    scheduler=args.lr_scheduler)

    if args.do_predict:
        test_prog = fluid.Program()
        with fluid.program_guard(test_prog, startup_prog):
            with fluid.unique_name.guard():
                test_data_loader,  predictions = create_model(
                    xlnet_config=xlnet_config,
                    is_training=False)

        test_prog = test_prog.clone(for_test=True)

    exe.run(startup_prog)

    if args.do_train:
        if args.init_checkpoint and args.init_pretraining_params:
            print(
                "WARNING: args 'init_checkpoint' and 'init_pretraining_params' "
                "both are set! Only arg 'init_checkpoint' is made valid.")
        if args.init_checkpoint:
            init_checkpoint(
                exe,
                args.init_checkpoint,
                main_program=startup_prog)
        elif args.init_pretraining_params:
            init_pretraining_params(
                exe,
                args.init_pretraining_params,
                main_program=startup_prog)

    elif args.do_predict:
        if not args.init_checkpoint:
            raise ValueError("args 'init_checkpoint' should be set if"
                             "only doing prediction!")
        init_checkpoint(
            exe,
            args.init_checkpoint,
            main_program=startup_prog)

    if args.do_train:
        exec_strategy = fluid.ExecutionStrategy()
        exec_strategy.use_experimental_executor = args.use_fast_executor
        exec_strategy.num_threads = dev_count
        exec_strategy.num_iteration_per_drop_scope = args.num_iteration_per_drop_scope

        build_strategy = fluid.BuildStrategy()
        # These two flags must be set in this model for correctness
        build_strategy.fuse_all_optimizer_ops = True
        build_strategy.enable_inplace = False
        train_exe = fluid.ParallelExecutor(
            use_cuda=args.use_cuda,
            loss_name=loss.name,
            exec_strategy=exec_strategy,
            build_strategy=build_strategy,
            main_program=train_program)

        train_data_loader.set_batch_generator(train_data_generator, place)

        train_data_loader.start()
        steps = 0
        total_cost = []
        time_begin = time.time()
        print("Begin to train model  ...")
        print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
        while steps < args.train_steps:
            try:
                steps += 1
                if steps % args.skip_steps == 0:
                    fetch_list = [loss.name, scheduled_lr.name]
                else:
                    fetch_list = []

                outputs = train_exe.run(fetch_list=fetch_list)

                if steps % args.skip_steps == 0:
                    np_loss, np_lr = outputs
                    total_cost.extend(np_loss)

                    if args.verbose:
                        verbose = "train data_loader queue size: %d, " % train_data_loader.queue.size(
                        )
                        verbose += "learning rate: %f " % np_lr[0]
                        print(verbose)

                    time_end = time.time()
                    used_time = time_end - time_begin
                    current_example, epoch = processor.get_train_progress()

                    print("epoch: %d, progress: %d/%d, step: %d, loss: %f, "
                          "speed: %f steps/s" %
                          (epoch, current_example, num_train_examples, steps,
                           np.mean(total_cost),
                           args.skip_steps / used_time))
                    total_cost = []
                    time_begin = time.time()

                if steps % args.save_steps == 0 or steps == args.train_steps:
                    save_path = os.path.join(args.checkpoints,
                                             "step_" + str(steps))
                    fluid.io.save_persistables(exe, save_path, train_program)
            except fluid.core.EOFException:
                save_path = os.path.join(args.checkpoints,
                                         "step_" + str(steps) + "_final")
                fluid.io.save_persistables(exe, save_path, train_program)
                train_data_loader.reset()
                break
        print("Finish model training ...")
        print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))

    if args.do_predict:
        print("Begin to do prediction  ...")
        print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
        test_data_loader.set_batch_generator(
            processor.data_generator(
                data_path=args.predict_file,
                batch_size=args.predict_batch_size,
                phase='predict',
                shuffle=False,
                dev_count=1,
                epoch=1), place)

        predict(exe, test_prog, test_data_loader, [predictions['unique_ids'].name, predictions['start_top_log_probs'].name,
            predictions['start_top_index'].name, predictions['end_top_log_probs'].name, predictions['end_top_index'].name,
            predictions['cls_logits'].name
        ], processor, name='')

        print("Finish prediction ...")
        print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))


if __name__ == '__main__':
    import paddle
    paddle.enable_static()
    print_arguments(args)
    train(args)
