# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import json
import logging
import os
import pickle
import sys
import random
from functools import partial
from pathlib import Path

import numpy as np
import paddle
import paddle.optimizer
import paddle.static
from scipy.stats import truncnorm
from datasets import load_dataset
from args import load_custom_ops, parse_args
from paddlenlp.metrics.squad import compute_prediction, squad_evaluate
from paddlenlp.transformers import BertTokenizer, LinearDecayWithWarmup

root_folder = str(Path(__file__).parent.parent.parent.absolute())
root_folder = "../../../paddlenlp"
sys.path.insert(0, root_folder)
from trainer import IPUTrainer
from IPU.model_wrapper import (
    IpuBertForQuestionAnswering,
    IpuBertQAAccAndLoss)
from IPU.utils import (
    AutoModel,
    AutoConfig,
    AutoPipeline
)

class Squadtrainer(IPUTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def log(self):
        logging.info({
            "global_step": self.global_step,
            "loss": np.mean(self.loss[0]),
            "accuracy": np.mean(self.loss[1:]),
            "train_cost": self.train_cost,
            "total_cost": self.total_cost,
            "throughput": self.tput,
            "learning_rate": self.lr_scheduler(),
        })


def set_seed(seed):
    """
    Use the same data seed(for data shuffle) for all procs to guarantee data
    consistency after sharding.
    """
    random.seed(seed)
    np.random.seed(seed)
    paddle.seed(seed)


def create_data_holder(args):
    bs = args.micro_batch_size
    indices = paddle.static.data(
        name="indices", shape=[bs * args.seq_len], dtype="int32")
    segments = paddle.static.data(
        name="segments", shape=[bs * args.seq_len], dtype="int32")
    positions = paddle.static.data(
        name="positions", shape=[bs * args.seq_len], dtype="int32")
    input_mask = paddle.static.data(
        name="input_mask", shape=[bs, 1, 1, args.seq_len], dtype="float32")
    if not args.is_training:
        return [indices, segments, positions, input_mask]
    else:
        start_labels = paddle.static.data(
            name="start_labels", shape=[bs], dtype="int32")
        end_labels = paddle.static.data(
            name="end_labels", shape=[bs], dtype="int32")
        return [
            indices, segments, positions, input_mask, start_labels, end_labels
        ]


def reset_program_state_dict(state_dict, mean=0, scale=0.02):
    """
    Initialize the parameter from the bert config, and set the parameter by
    reseting the state dict."
    """
    new_state_dict = dict()
    for n, p in state_dict.items():
        if  n.endswith('_moment1_0') or n.endswith('_moment2_0') \
            or n.endswith('_beta2_pow_acc_0') or n.endswith('_beta1_pow_acc_0'):
            continue
        if 'learning_rate' in n:
            continue

        dtype_str = "float32"
        if p._dtype == paddle.float64:
            dtype_str = "float64"

        if "layer_norm" in n and n.endswith('.w_0'):
            new_state_dict[n] = np.ones(p.shape()).astype(dtype_str)
            continue

        if n.endswith('.b_0'):
            new_state_dict[n] = np.zeros(p.shape()).astype(dtype_str)
        else:
            new_state_dict[n] = truncnorm.rvs(-2,
                                              2,
                                              loc=mean,
                                              scale=scale,
                                              size=p.shape()).astype(dtype_str)
    return new_state_dict


def prepare_train_features(examples, tokenizer, args):
    # Some of the questions have lots of whitespace on the left, which is not useful and will make the
    # truncation of the context fail (the tokenized question will take a lots of space). So we remove that
    # left whitespace
    contexts = examples['context']
    questions = examples['question']

    # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
    # in one example possible giving several features when a context is long, each of those features having a
    # context that overlaps a bit the context of the previous feature.
    tokenized_examples = tokenizer(
        questions,
        contexts,
        stride=128,
        max_seq_len=args.seq_len,
        pad_to_max_seq_len=True,
        return_position_ids=True,
        return_token_type_ids=True,
        return_attention_mask=True,
        return_length=True)

    # Since one example might give us several features if it has a long context, we need a map from a feature to
    # its corresponding example. This key gives us just that.
    sample_mapping = tokenized_examples.pop("overflow_to_sample")
    # The offset mappings will give us a map from token to character position in the original context. This will
    # help us compute the start_positions and end_positions.
    offset_mapping = tokenized_examples.pop("offset_mapping")

    # Let's label those examples!
    tokenized_examples["start_positions"] = []
    tokenized_examples["end_positions"] = []
    tokenized_examples["input_mask"] = []

    for i, offsets in enumerate(offset_mapping):
        # We will label impossible answers with the index of the CLS token.
        input_ids = tokenized_examples["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)

        sequence_ids = tokenized_examples['token_type_ids'][i]

        # attention_mask to input_mask
        input_mask = (
            np.asarray(tokenized_examples["attention_mask"][i]) - 1) * 1e3
        input_mask = np.expand_dims(input_mask, axis=(0, 1))
        if args.ipu_enable_fp16:
            input_mask = input_mask.astype(np.float16)
        else:
            input_mask = input_mask.astype(np.float32)
        tokenized_examples["input_mask"].append(input_mask)

        # One example can give several spans, this is the index of the example containing this span of text.
        sample_index = sample_mapping[i]
        answers = examples['answers'][sample_index]
        # If no answers are given, set the cls_index as answer.
        if len(answers["answer_start"]) == 0:
            tokenized_examples["start_positions"].append(cls_index)
            tokenized_examples["end_positions"].append(cls_index)
        else:
            # Start/end character index of the answer in the text.
            start_char = answers["answer_start"][0]
            end_char = start_char + len(answers["text"][0])

            # Start token index of the current span in the text.
            token_start_index = 0
            while sequence_ids[token_start_index] != 1:
                token_start_index += 1

            # End token index of the current span in the text.
            token_end_index = len(input_ids) - 1
            while sequence_ids[token_end_index] != 1:
                token_end_index -= 1
            token_end_index -= 1

            # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
            if not (offsets[token_start_index][0] <= start_char and
                    offsets[token_end_index][1] >= end_char):
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                # Note: we could go after the last offset if the answer is the last word (edge case).
                while token_start_index < len(offsets) and offsets[
                        token_start_index][0] <= start_char:
                    token_start_index += 1
                tokenized_examples["start_positions"].append(token_start_index -
                                                             1)
                while offsets[token_end_index][1] >= end_char:
                    token_end_index -= 1
                tokenized_examples["end_positions"].append(token_end_index + 1)

    return tokenized_examples


def prepare_validation_features(examples, tokenizer, args):
    # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
    # in one example possible giving several features when a context is long, each of those features having a
    # context that overlaps a bit the context of the previous feature.
    #NOTE: Almost the same functionality as HuggingFace's prepare_train_features function. The main difference is
    # that HugggingFace uses ArrowTable as basic data structure, while we use list of dictionary instead.
    contexts = examples['context']
    questions = examples['question']
    tokenized_examples = tokenizer(
        questions,
        contexts,
        stride=128,
        max_seq_len=args.seq_len,
        pad_to_max_seq_len=True,
        return_position_ids=True,
        return_token_type_ids=True,
        return_attention_mask=True,
        return_length=True)

    # Since one example might give us several features if it has a long context, we need a map from a feature to
    # its corresponding example. This key gives us just that.
    sample_mapping = tokenized_examples.pop("overflow_to_sample")

    # For evaluation, we will need to convert our predictions to substrings of the context, so we keep the
    # corresponding example_id and we will store the offset mappings.
    tokenized_examples["example_id"] = []
    tokenized_examples["input_mask"] = []

    for i in range(len(tokenized_examples["input_ids"])):
        # Grab the sequence corresponding to that example (to know what is the context and what is the question).
        input_ids = tokenized_examples["input_ids"][i]
        sequence_A_lengths = input_ids.index(tokenizer.sep_token_id) + 2
        sequence_B_lengths = len(input_ids) - sequence_A_lengths
        sequence_ids = [0] * sequence_A_lengths + [1] * sequence_B_lengths
        context_index = 1

        # One example can give several spans, this is the index of the example containing this span of text.
        sample_index = sample_mapping[i]
        tokenized_examples["example_id"].append(examples["id"][sample_index])

        # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
        # position is part of the context or not.
        tokenized_examples["offset_mapping"][i] = [
            (o if sequence_ids[k] == context_index else None)
            for k, o in enumerate(tokenized_examples["offset_mapping"][i])
        ]

        # attention_mask to input_mask
        input_mask = (
            np.asarray(tokenized_examples["attention_mask"][i]) - 1) * 1e3
        input_mask = np.expand_dims(input_mask, axis=(0, 1))
        if args.ipu_enable_fp16:
            input_mask = input_mask.astype(np.float16)
        else:
            input_mask = input_mask.astype(np.float32)
        tokenized_examples["input_mask"].append(input_mask)

    return tokenized_examples


def load_squad_dataset(args):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    features_fn = prepare_train_features if args.is_training else prepare_validation_features
    if args.is_training:
        raw_dataset = load_dataset('squad', split='train')
    else:
        raw_dataset = load_dataset('squad', split='validation')
    column_names = raw_dataset.column_names
    dataset = raw_dataset.map(partial(
        features_fn, tokenizer=tokenizer, args=args),
                              batched=True,
                              remove_columns=column_names,
                              num_proc=4)

    bs = args.micro_batch_size * args.grad_acc_factor * args.batches_per_step * args.num_replica
    args.batch_size = bs
    return raw_dataset, dataset


def main(args):
    paddle.enable_static()
    place = paddle.set_device('ipu')
    set_seed(args.seed)
    main_program = paddle.static.default_main_program()
    startup_program = paddle.static.default_startup_program()

    # custom_ops
    custom_ops = load_custom_ops()

    logging.info("building model")
    if args.is_training:
        [indices, segments, positions, input_mask, start_labels,
         end_labels] = create_data_holder(args)
    else:
        [indices, segments, positions, input_mask] = create_data_holder(args)

    # Encoder Layers
    config = AutoConfig(args)
    bert_model = AutoModel(args, config, custom_ops)
    AutoPipeline(args, config, bert_model)
    encoders, _ = bert_model(indices, segments, positions, input_mask)

    # squad_scope = DeviceScope(args.num_ipus - 1, args.num_ipus - 1, "squad")
    with config.squad_scope:
        qa_cls = IpuBertForQuestionAnswering(args.hidden_size, args.seq_len)
        start_logits, end_logits = qa_cls(encoders)

        if args.is_training:
            acc_loss = IpuBertQAAccAndLoss(custom_ops)
            acc0, acc1, loss = acc_loss(start_logits, end_logits, start_labels,
                                        end_labels)

    # load squad dataset
    raw_dataset, dataset = load_squad_dataset(args)

    total_samples = len(dataset)
    max_steps = total_samples // args.batch_size * args.epochs
    logging.info("total samples: %d, total batch_size: %d, max steps: %d" %
                 (total_samples, args.batch_size, max_steps))

    if args.is_training:
        lr_scheduler = LinearDecayWithWarmup(args.learning_rate, max_steps,
                                             args.warmup_steps)
        optimizer = paddle.optimizer.Adam(
            learning_rate=lr_scheduler,
            weight_decay=args.weight_decay,
            beta1=args.beta1,
            beta2=args.beta2,
            epsilon=args.adam_epsilon)
        optimizer.minimize(loss)

    # Static executor
    exe = paddle.static.Executor(place)
    exe.run(startup_program)

    # Set initial weights
    state_dict = main_program.state_dict()
    reset_state_dict = reset_program_state_dict(state_dict)
    paddle.static.set_program_state(main_program, reset_state_dict)

    amp_list = paddle.static.amp.CustomOpLists()
    amp_list.unsupported_list = {}
    to_fp16_var_names = paddle.static.amp.cast_model_to_fp16(
        main_program, amp_list, use_fp16_guard=False)
    paddle.static.amp.cast_parameters_to_fp16(
        paddle.CPUPlace(),
        main_program,
        to_fp16_var_names=to_fp16_var_names)

    if args.enable_load_params:
        logging.info(f'loading weights from: {args.load_params_path}')
        if not args.load_params_path.endswith('pdparams'):
            raise Exception('need pdparams file')
        with open(args.load_params_path, 'rb') as file:
            params = pickle.load(file)
        # Delete mlm and nsp weights
        if args.is_training and 'linear_72.w_0' in params:
            params.pop("linear_72.w_0")
            params.pop("linear_72.b_0")
        paddle.static.set_program_state(main_program, params)

    if args.tf_checkpoint:
        from load_tf_ckpt import load_initializers_from_tf
        logging.info(f'loading weights from: {args.tf_checkpoint}')
        initializers, _ = load_initializers_from_tf(args.tf_checkpoint, args)
        paddle.static.set_program_state(main_program, initializers)

    if args.is_training:
        feed_list = [
            "indices", "segments", "positions", "input_mask", "start_labels",
            "end_labels"
        ]
        fetch_list = [loss.name, acc0.name, acc1.name]
    else:
        feed_list = ["indices", "segments", "positions", "input_mask"]
        fetch_list = [start_logits.name, end_logits.name]

    if args.is_training:
        # Initialize Trainer
        trainer = Squadtrainer(
            args = args,
            dataset = dataset,
            exe = exe,
            tensor_list = [feed_list, fetch_list],
            program = [main_program, startup_program],
            optimizers = [optimizer, lr_scheduler]
        )
        trainer.train()
        trainer.save_model()

    if not args.is_training:
        # Initialize Trainer
        trainer = IPUTrainer(
            args = args,
            dataset = dataset,
            exe = exe,
            tensor_list = [feed_list, fetch_list],
            program = [main_program, startup_program]
        )

        trainer.eval()

        # evaluate results
        all_predictions, all_nbest_json, scores_diff_json = compute_prediction(
            raw_dataset, dataset,
            (trainer.all_start_logits, trainer.all_end_logits))
        squad_evaluate(
            examples=[raw_data for raw_data in raw_dataset],
            preds=all_predictions,
            na_probs=scores_diff_json)
        # write results to file
        with open('squad_prediction.json', "w", encoding='utf-8') as writer:
            writer.write(
                json.dumps(
                    all_predictions, ensure_ascii=False, indent=4) + "\n")


if __name__ == "__main__":
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt='%Y-%m-%d %H:%M:%S %a')

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    if args.wandb:
        import wandb
        wandb.init(
            project="paddle-squad",
            settings=wandb.Settings(console='off'),
            name='paddle-squad')
        wandb_config = vars(args)
        wandb_config["global_batch_size"] = args.batch_size
        wandb.config.update(args)

    logging.info(args)
    main(args)
    logging.info("program finished")
