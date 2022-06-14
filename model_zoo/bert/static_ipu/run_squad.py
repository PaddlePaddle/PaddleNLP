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
import time
from functools import partial

import numpy as np
import paddle
import paddle.optimizer
import paddle.static
from datasets import load_dataset
from paddle.io import BatchSampler, DataLoader
from paddlenlp.data import Dict, Stack
from paddlenlp.metrics.squad import compute_prediction, squad_evaluate
from paddlenlp.transformers import BertTokenizer, LinearDecayWithWarmup

from modeling import (BertModel, DeviceScope, IpuBertConfig,
                      IpuBertForQuestionAnswering, IpuBertQAAccAndLoss)
from run_pretrain import (create_ipu_strategy, reset_program_state_dict,
                          set_seed)
from utils import load_custom_ops, parse_args


def create_data_holder(args):
    bs = args.micro_batch_size
    indices = paddle.static.data(name="indices",
                                 shape=[bs * args.seq_len],
                                 dtype="int32")
    segments = paddle.static.data(name="segments",
                                  shape=[bs * args.seq_len],
                                  dtype="int32")
    positions = paddle.static.data(name="positions",
                                   shape=[bs * args.seq_len],
                                   dtype="int32")
    input_mask = paddle.static.data(name="input_mask",
                                    shape=[bs, 1, 1, args.seq_len],
                                    dtype="float32")
    if not args.is_training:
        return [indices, segments, positions, input_mask]
    else:
        start_labels = paddle.static.data(name="start_labels",
                                          shape=[bs],
                                          dtype="int32")
        end_labels = paddle.static.data(name="end_labels",
                                        shape=[bs],
                                        dtype="int32")
        return [
            indices, segments, positions, input_mask, start_labels, end_labels
        ]


def prepare_train_features(examples, tokenizer, args):
    # Some of the questions have lots of whitespace on the left, which is not useful and will make the
    # truncation of the context fail (the tokenized question will take a lots of space). So we remove that
    # left whitespace
    contexts = examples['context']
    questions = examples['question']

    # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
    # in one example possible giving several features when a context is long, each of those features having a
    # context that overlaps a bit the context of the previous feature.
    tokenized_examples = tokenizer(questions,
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
        input_mask = (np.asarray(tokenized_examples["attention_mask"][i]) -
                      1) * 1e3
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
            if not (offsets[token_start_index][0] <= start_char
                    and offsets[token_end_index][1] >= end_char):
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
    tokenized_examples = tokenizer(questions,
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
        input_mask = (np.asarray(tokenized_examples["attention_mask"][i]) -
                      1) * 1e3
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
    dataset = raw_dataset.map(partial(features_fn,
                                      tokenizer=tokenizer,
                                      args=args),
                              batched=True,
                              remove_columns=column_names,
                              num_proc=4)

    bs = args.micro_batch_size * args.grad_acc_factor * args.batches_per_step * args.num_replica
    args.batch_size = bs
    if args.is_training:
        train_batch_sampler = BatchSampler(dataset,
                                           batch_size=bs,
                                           shuffle=args.shuffle,
                                           drop_last=True)
    else:
        train_batch_sampler = BatchSampler(dataset,
                                           batch_size=bs,
                                           shuffle=args.shuffle,
                                           drop_last=False)

    if args.is_training:
        collate_fn = lambda samples, fn=Dict({
            "input_ids": Stack(),
            "token_type_ids": Stack(),
            "position_ids": Stack(),
            "input_mask": Stack(),
            "start_positions": Stack(),
            "end_positions": Stack()
        }): fn(samples)
    else:
        collate_fn = lambda samples, fn=Dict({
            "input_ids": Stack(),
            "token_type_ids": Stack(),
            "position_ids": Stack(),
            "input_mask": Stack()
        }): fn(samples)

    data_loader = DataLoader(dataset=dataset,
                             batch_sampler=train_batch_sampler,
                             collate_fn=collate_fn,
                             return_list=True)
    return raw_dataset, data_loader


def main(args):
    paddle.enable_static()
    place = paddle.set_device('ipu')
    set_seed(args.seed)
    main_program = paddle.static.default_main_program()
    startup_program = paddle.static.default_startup_program()

    # The sharding of encoder layers
    if args.num_hidden_layers == 12:
        attn_ipu_index = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
        ff_ipu_index = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
    else:
        raise Exception("Only support num_hidden_layers = 12")

    bert_config = {
        k: getattr(args, k)
        for k in IpuBertConfig._fields if hasattr(args, k)
    }
    bert_config['embeddings_scope'] = DeviceScope(0, 0, "Embedding")
    bert_config['attn_scopes'] = [
        DeviceScope(attn_ipu_index[i], attn_ipu_index[i])
        for i in range(args.num_hidden_layers)
    ]
    bert_config['ff_scopes'] = [
        DeviceScope(ff_ipu_index[i], ff_ipu_index[i])
        for i in range(args.num_hidden_layers)
    ]
    bert_config['layers_per_ipu'] = [6, 6]

    config = IpuBertConfig(**bert_config)

    # custom_ops
    custom_ops = load_custom_ops()

    logging.info("building model")

    if args.is_training:
        [indices, segments, positions, input_mask, start_labels,
         end_labels] = create_data_holder(args)
    else:
        [indices, segments, positions, input_mask] = create_data_holder(args)

    # Encoder Layers
    bert_model = BertModel(config, custom_ops)
    encoders, _ = bert_model(indices, segments, positions, input_mask)

    squad_scope = DeviceScope(args.num_ipus - 1, args.num_ipus - 1, "squad")
    with squad_scope:
        qa_cls = IpuBertForQuestionAnswering(args.hidden_size, args.seq_len)
        start_logits, end_logits = qa_cls(encoders)

        if args.is_training:
            acc_loss = IpuBertQAAccAndLoss(custom_ops)
            acc0, acc1, loss = acc_loss(start_logits, end_logits, start_labels,
                                        end_labels)

    # load squad dataset
    raw_dataset, data_loader = load_squad_dataset(args)

    total_samples = len(data_loader.dataset)
    max_steps = total_samples // args.batch_size * args.epochs
    logging.info("total samples: %d, total batch_size: %d, max steps: %d" %
                 (total_samples, args.batch_size, max_steps))

    if args.is_training:
        lr_scheduler = LinearDecayWithWarmup(args.learning_rate, max_steps,
                                             args.warmup_steps)
        optimizer = paddle.optimizer.Adam(learning_rate=lr_scheduler,
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

    # Create ipu_strategy
    ipu_strategy = create_ipu_strategy(args)

    if args.is_training:
        feed_list = [
            "indices", "segments", "positions", "input_mask", "start_labels",
            "end_labels"
        ]
        fetch_list = [loss.name, acc0.name, acc1.name]
    else:
        feed_list = ["indices", "segments", "positions", "input_mask"]
        fetch_list = [start_logits.name, end_logits.name]

    ipu_compiler = paddle.static.IpuCompiledProgram(main_program,
                                                    ipu_strategy=ipu_strategy)
    logging.info(f'start compiling, please wait some minutes')
    cur_time = time.time()
    main_program = ipu_compiler.compile(feed_list, fetch_list)
    time_cost = time.time() - cur_time
    logging.info(f'finish compiling! time cost: {time_cost}')

    if args.is_training:
        global_step = 0
        batch_start = time.time()
        for epoch in range(args.epochs):
            for batch in data_loader:
                global_step += 1

                feed = {
                    "indices": batch[0],
                    "segments": batch[1],
                    "positions": batch[2],
                    "input_mask": batch[3],
                    "start_labels": batch[4],
                    "end_labels": batch[5],
                }
                lr_scheduler.step()

                train_start = time.time()
                outputs = exe.run(main_program,
                                  feed=feed,
                                  fetch_list=fetch_list,
                                  use_program_cache=True)
                train_cost = time.time() - train_start
                total_cost = time.time() - batch_start

                tput = args.batch_size / total_cost
                if args.wandb:
                    wandb.log({
                        "epoch": epoch,
                        "global_step": global_step,
                        "loss": np.mean(outputs[0]),
                        "accuracy": np.mean(outputs[1:]),
                        "train_cost": train_cost,
                        "total_cost": total_cost,
                        "throughput": tput,
                        "learning_rate": lr_scheduler(),
                    })

                if global_step % args.logging_steps == 0:
                    logging.info({
                        "epoch": epoch,
                        "global_step": global_step,
                        "loss": np.mean(outputs[0]),
                        "accuracy": np.mean(outputs[1:]),
                        "train_cost": train_cost,
                        "total_cost": total_cost,
                        "throughput": tput,
                        "learning_rate": lr_scheduler(),
                    })

                batch_start = time.time()

        # save final state
        ipu_compiler._backend.weights_to_host()
        paddle.static.save(main_program.org_program,
                           os.path.join(args.output_dir, 'Final_model'))

    if not args.is_training:
        all_start_logits = []
        all_end_logits = []
        for step, batch in enumerate(data_loader):
            if step % args.logging_steps == 0:
                logging.info(f'running step: {step}')

            real_len = np.array(batch[0]).shape[0]
            # padding zeros if needed
            if real_len < args.batch_size:
                batch = [np.asarray(x) for x in batch]
                pad0 = np.zeros([args.batch_size - real_len,
                                 args.seq_len]).astype(batch[0].dtype)
                batch[0] = np.vstack((batch[0], pad0))
                batch[1] = np.vstack((batch[1], pad0))
                batch[2] = np.vstack((batch[2], pad0))
                pad1 = np.zeros(
                    [args.batch_size - real_len, 1, 1, args.seq_len]) - 1e3
                pad1 = pad1.astype(batch[3].dtype)
                batch[3] = np.vstack((batch[3], pad1))

            feed = {
                "indices": batch[0],
                "segments": batch[1],
                "positions": batch[2],
                "input_mask": batch[3],
            }
            start_logits, end_logits = exe.run(main_program,
                                               feed=feed,
                                               fetch_list=fetch_list)

            start_logits = start_logits.reshape([-1, args.seq_len])
            end_logits = end_logits.reshape([-1, args.seq_len])
            for idx in range(real_len):
                all_start_logits.append(start_logits[idx])
                all_end_logits.append(end_logits[idx])

        # evaluate results
        all_predictions, all_nbest_json, scores_diff_json = compute_prediction(
            raw_dataset, data_loader.dataset,
            (all_start_logits, all_end_logits))
        squad_evaluate(examples=[raw_data for raw_data in raw_dataset],
                       preds=all_predictions,
                       na_probs=scores_diff_json)
        # write results to file
        with open('squad_prediction.json', "w", encoding='utf-8') as writer:
            writer.write(
                json.dumps(all_predictions, ensure_ascii=False, indent=4) +
                "\n")


if __name__ == "__main__":
    args = parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s",
                        datefmt='%Y-%m-%d %H:%M:%S %a')

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    if args.wandb:
        import wandb
        wandb.init(project="paddle-squad",
                   settings=wandb.Settings(console='off'),
                   name='paddle-squad')
        wandb_config = vars(args)
        wandb_config["global_batch_size"] = args.batch_size
        wandb.config.update(args)

    logging.info(args)
    main(args)
    logging.info("program finished")
