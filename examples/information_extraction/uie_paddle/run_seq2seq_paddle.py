#!/usr/bin/env python
# -*- coding:utf-8 -*-
import logging
import argparse
from tqdm import tqdm
import math
import os
import json
from collections import defaultdict
import tabulate

import paddle
from paddle.io import DistributedBatchSampler, BatchSampler, DataLoader
from paddle.optimizer import AdamW
from paddle.amp import GradScaler, auto_cast
from paddlenlp.datasets import load_dataset

import paddle
from paddlenlp.transformers import T5ForConditionalGeneration, T5Tokenizer

from uie.extraction import constants
from uie.extraction.predict_parser import decoding_format_dict
from uie.extraction.record_schema import RecordSchema
from uie.extraction.scorer import *
from uie.extraction.noiser.spot_asoc_noiser import SpotAsocNoiser
from uie.sel2record.record import MapConfig
from uie.sel2record.sel2record import SEL2Record
from uie.seq2seq_paddle.data_collator.multi_meta_data_collator import (
    DataCollatorForMultiTaskSeq2Seq,
    DynamicMultiTaskSSIGenerator, )
from uie.seq2seq_paddle.t5_bert_tokenizer import T5BertTokenizer
from uie.seq2seq_paddle.utils import (
    get_scheduler,
    get_writer,
    set_seed,
    set_logger, )
from uie.seq2seq_paddle.data_collator.meta_data_collator import (
    DataCollatorForSeq2Seq,
    DynamicSSIGenerator, )

task_dict = {
    'entity': EntityScorer,
    'relation': RelationScorer,
    'event': EventScorer,
}

logger = logging.getLogger(__name__)


def read_json_file(file_name):
    return [json.loads(line) for line in open(file_name)]


def save_checkpoint(tokenizer, model, output_dir):
    if isinstance(model, paddle.DataParallel):
        model = model._layers
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


def better_print(results):
    better_print_results = defaultdict(dict)
    for k, v in results.items():
        att = k.split('-')
        prefix = '-'.join(att[:-1])
        metric = att[-1]
        better_print_results[prefix][metric] = v
    headers = ['tp', 'gold', 'pred', 'P', 'R', 'F1']
    table = [
        [task] + [better_print_results[task][metric] for metric in headers]
        for task in better_print_results
    ]
    return tabulate.tabulate(table, headers=['task'] + headers)


def read_func(tokenizer,
              data_file,
              max_source_length,
              max_target_length,
              is_train=False):
    def tokenize(x, max_length):
        return tokenizer(
            x,
            return_token_type_ids=False,
            return_attention_mask=True,
            max_seq_len=max_length, )

    with open(data_file, 'r', encoding='utf-8') as f:
        for line in f:
            instance = json.loads(line)
            inputs = tokenize(instance['text'], max_source_length)

            inputs['spots'] = instance['spot']
            inputs['asocs'] = instance['asoc']
            inputs['spot_asoc'] = instance['spot_asoc']

            if is_train:
                inputs['sample_prompt'] = [True] * len(inputs['input_ids'])
            else:
                inputs['sample_prompt'] = [False] * len(inputs['input_ids'])

            yield inputs


def get_train_dataloader(model, tokenizer, train_filename, args):

    schema = RecordSchema.read_from_file(args.record_schema)

    dataset = load_dataset(
        read_func,
        tokenizer=tokenizer,
        data_file=train_filename,
        max_source_length=args.max_source_length,
        max_target_length=args.max_target_length,
        is_train=True,
        lazy=False)

    batch_sampler = DistributedBatchSampler(
        dataset=dataset,
        batch_size=args.per_device_train_batch_size,
        shuffle=True, )

    if args.spot_noise > 0 or args.asoc_noise > 0:
        spot_asoc_nosier = SpotAsocNoiser(
            spot_noise_ratio=args.spot_noise,
            asoc_noise_ratio=args.asoc_noise,
            null_span=constants.null_span, )
    else:
        spot_asoc_nosier = None

    label_pad_token_id = -100 if args.ignore_pad_token_for_loss else tokenizer.pad_token_id

    if args.multi_task:
        seq2seq_data_collator = DataCollatorForMultiTaskSeq2Seq
        dynamic_ssi_generator = DynamicMultiTaskSSIGenerator
    else:
        seq2seq_data_collator = DataCollatorForSeq2Seq
        dynamic_ssi_generator = DynamicSSIGenerator

    collate_fn = seq2seq_data_collator(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8 if args.use_amp else None,
        max_source_length=args.max_source_length,
        max_prefix_length=args.max_prefix_length,
        max_target_length=args.max_target_length,
        negative_sampler=dynamic_ssi_generator(
            tokenizer=tokenizer,
            schema=schema,
            positive_rate=args.meta_positive_rate,
            negative=args.meta_negative,
            ordered_prompt=args.ordered_prompt, ),
        spot_asoc_nosier=spot_asoc_nosier,
        decoding_format=args.decoding_format, )

    data_loader = DataLoader(
        dataset=dataset,
        batch_sampler=batch_sampler,
        collate_fn=collate_fn,
        num_workers=args.dataloader_num_workers,
        return_list=True)

    return data_loader


def get_eval_dataloader(model, tokenizer, eval_filename, args):

    schema = RecordSchema.read_from_file(args.record_schema)

    dataset = load_dataset(
        read_func,
        tokenizer=tokenizer,
        data_file=eval_filename,
        max_source_length=args.max_source_length,
        max_target_length=args.max_target_length,
        is_train=False,
        lazy=False)

    batch_sampler = BatchSampler(
        dataset=dataset,
        batch_size=args.per_device_eval_batch_size,
        shuffle=False)

    label_pad_token_id = -100 if args.ignore_pad_token_for_loss else tokenizer.pad_token_id

    if args.multi_task:
        seq2seq_data_collator = DataCollatorForMultiTaskSeq2Seq
        dynamic_ssi_generator = DynamicMultiTaskSSIGenerator
    else:
        seq2seq_data_collator = DataCollatorForSeq2Seq
        dynamic_ssi_generator = DynamicSSIGenerator

    collate_fn = seq2seq_data_collator(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8 if args.use_amp else None,
        max_source_length=args.max_source_length,
        max_prefix_length=args.max_prefix_length,
        max_target_length=args.max_target_length,
        negative_sampler=dynamic_ssi_generator(
            tokenizer=tokenizer,
            schema=schema,
            positive_rate=1,
            negative=-1,
            ordered_prompt=True, ),
        spot_asoc_nosier=None,
        decoding_format=args.decoding_format, )

    data_loader = DataLoader(
        dataset=dataset,
        batch_sampler=batch_sampler,
        collate_fn=collate_fn,
        num_workers=args.dataloader_num_workers,
        return_list=True)

    return data_loader


def get_eval_instances(filename):
    return [json.loads(line.strip()) for line in open(filename)]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--multi_task",
        action="store_true",
        help="Path to pre-trained model or shortcut name of model.", )
    parser.add_argument(
        "--model_name_or_path",
        default="t5-large",
        type=str,
        help="Path to pre-trained model or shortcut name of model.", )
    parser.add_argument(
        "--train_file",
        default=None,
        type=str,
        help="The input training data jsonlines file.", )
    parser.add_argument(
        "--validation_file",
        default=None,
        type=str,
        help="An optional input evaluation data file to evaluate the metrics.",
    )
    parser.add_argument(
        "--test_file",
        default=None,
        type=str,
        help="An optional input evaluation data file to evaluate the metrics.",
    )
    parser.add_argument(
        "--record_schema",
        default=None,
        type=str,
        help="Schema for information extraction.", )
    parser.add_argument(
        "--record_schema_dir",
        default=None,
        type=str,
        help="Schema Dir for information extraction.", )
    parser.add_argument(
        "--output_dir",
        required=True,
        type=str,
        help="The output directory where the model predictions and checkpoints"
        " will be written. Default as `outputs`", )
    parser.add_argument(
        "--overwrite_output_dir",
        action='store_true',
        help="Overwrite output directory", )
    parser.add_argument(
        "--logging_dir",
        required=True,
        type=str,
        help="The output logging directory", )
    parser.add_argument(
        "--max_source_length",
        default=256,
        type=int,
        help="The maximum total input sequence length after tokenization."
        " Sequences longer than this will be truncated, sequences shorter will be padded."
    )
    parser.add_argument(
        "--max_target_length",
        default=128,
        type=int,
        help="The maximum total target sequence length to be generated.", )
    parser.add_argument(
        "--max_prefix_length",
        default=None,
        type=int,
        help="The maximum prefix length.")
    parser.add_argument(
        "--per_device_train_batch_size",
        default=16,
        type=int,
        help="Batch size per GPU/CPU for training.", )
    parser.add_argument(
        "--per_device_eval_batch_size",
        default=16,
        type=int,
        help="Batch size per GPU/CPU for evaluating.", )
    parser.add_argument(
        "--metric_for_best_model",
        type=str,
        required=True,
        help="The main metric to choose the best epoch.", )
    parser.add_argument(
        "--gradient_accumulation_steps",
        default=1,
        type=int,
        help="gradient_accumulation_steps.", )
    parser.add_argument(
        "--learning_rate",
        default=1e-4,
        type=float,
        help="The initial learning rate for Adam.", )
    parser.add_argument(
        "--weight_decay",
        default=0.0,
        type=float,
        help="Weight decay if we apply some.")
    parser.add_argument(
        "--adam_beta1",
        default=0.9,
        type=float,
        help="Beta1 for AdamW optimizer")
    parser.add_argument(
        "--adam_beta2",
        default=0.999,
        type=float,
        help="Beta2 for AdamW optimizer")
    parser.add_argument(
        "--adam_epsilon",
        default=1e-6,
        type=float,
        help="Epsilon for Adam optimizer.")
    parser.add_argument(
        "--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--num_train_epochs",
        default=4,
        type=int,
        help="Total number of training epochs to perform.", )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument(
        "--warmup_ratio",
        default=0.06,
        type=float,
        help="Proportion of training steps to perform linear learning rate warmup for.",
    )
    parser.add_argument(
        "--ignore_pad_token_for_loss",
        type=bool,
        default=True,
        help="Whether to ignore the tokens corresponding to padded labels in the loss computation or not.",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=-1,
        help="warmup_steps.", )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=100,
        help="Log every X updates steps.", )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="random seed for initialization", )
    parser.add_argument(
        "--writer_type",
        choices=["visualdl", "tensorboard"],
        default="visualdl",
        help="writer_type.", )
    parser.add_argument(
        "--lr_scheduler_type",
        choices=["linear", "cosine", "poly"],
        default="linear",
        type=str,
        help="lr_scheduler_type.", )
    parser.add_argument(
        "--use_amp",
        "--fp16",
        action="store_true",
        help="Enable mixed precision training.")
    parser.add_argument(
        "--scale_loss",
        type=float,
        default=2**15,
        help="The value of scale_loss for fp16.", )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help="dataloader_num_workers.", )
    parser.add_argument(
        "--spot_noise",
        type=float,
        default=0,
        help="The noise rate of null spot.")
    parser.add_argument(
        "--asoc_noise",
        type=float,
        default=0,
        help="The noise rate of null asoc.")
    parser.add_argument(
        "--meta_positive_rate",
        type=float,
        default=1,
        help="The keep rate of positive spot.")
    parser.add_argument(
        "--meta_negative",
        type=int,
        default=-1,
        help="Negative Schema Number in Training.")
    parser.add_argument(
        "--ordered_prompt",
        action='store_true',
        help="Whether to sort the spot prompt and asoc prompt or not.")
    parser.add_argument(
        "--save_steps",
        default=-1,
        type=int,
        help="Save checkpoint each steps.")
    parser.add_argument(
        "--decoding_format",
        default='tree',
        help="Decoding Format, valid in %s" % decoding_format_dict.keys())
    parser.add_argument(
        '--config',
        dest='map_config',
        help='Offset match strategy configure.',
        default='config/offset_map/closest_offset_en.yaml')
    parser.add_argument('--decoding', default='spotasoc')
    parser.add_argument(
        '--match_mode',
        default='normal',
        choices=['set', 'normal', 'multimatch'])
    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--do_predict', action='store_true')
    parser.add_argument(
        "--device",
        type=str,
        default="gpu",
        choices=["cpu", "gpu", "xpu"],
        help="Device for selecting for the training.")

    args = parser.parse_args()

    # Sanity check
    if not (args.do_train or args.do_predict):
        raise ValueError(
            "At least one of the \"--do_train\" or \"--do_predict\" should be true."
        )

    return args


@paddle.no_grad()
def evaluate(model, tokenizer, data_loader, generate_max_length, eval_instances,
             sel2record):

    model = model._layers if isinstance(model, paddle.DataParallel) else model

    model.eval()

    to_remove_token_list = list()
    if tokenizer.eos_token:
        to_remove_token_list += [tokenizer.eos_token]
    if tokenizer.pad_token:
        to_remove_token_list += [tokenizer.pad_token]

    def postprocess_text(x_str):
        # Clean `bos` `eos` `pad` for cleaned text
        for to_remove_token in to_remove_token_list:
            x_str = x_str.replace(to_remove_token, '')

        return x_str.strip()

    all_preds = []
    for batch in tqdm(data_loader):
        source_ids, source_mask = batch['input_ids'], batch['attention_mask']

        # We can no longer use model.generate because it fails in distributed envs
        outputs, scores = model.generate(
            input_ids=source_ids,
            attention_mask=source_mask,
            max_length=generate_max_length,
            use_faster=True, )

        outputs = tokenizer.batch_decode(
            outputs,
            clean_up_tokenization_spaces=False,
            skip_special_tokens=False, )

        preds = [postprocess_text(o) for o in outputs]
        all_preds.extend(preds)

    assert len(all_preds) == len(eval_instances)

    all_records = []
    for p, instance in zip(all_preds, eval_instances):
        r = sel2record.sel2record(
            pred=p, text=instance['text'], tokens=instance['tokens'])
        all_records += [r]

    results = dict()
    for task, scorer in task_dict.items():

        gold_list = [x[task] for x in eval_instances]
        pred_list = [x[task] for x in all_records]

        gold_instance_list = scorer.load_gold_list(gold_list)
        pred_instance_list = scorer.load_pred_list(pred_list)
        sub_results = scorer.eval_instance_list(
            gold_instance_list=gold_instance_list,
            pred_instance_list=pred_instance_list,
            verbose=False,
            match_mode=args.match_mode, )
        results.update(sub_results)

    prediction = {
        'preds_record.txt': all_records,
        'preds_seq2seq.txt': all_preds,
        'results.txt': results
    }

    return results, prediction


def test(args, model, tokenizer):

    generate_max_length = args.max_target_length

    paddle.set_device(args.device)

    test_instances = get_eval_instances(args.test_file)

    sel2record = SEL2Record(
        schema_dict=SEL2Record.load_schema_dict(args.record_schema_dir),
        decoding_schema=args.decoding,
        map_config=MapConfig.load_from_yaml(args.map_config), )

    test_dataloader = get_eval_dataloader(
        model=model,
        tokenizer=tokenizer,
        eval_filename=args.test_file,
        args=args, )

    eval_results, eval_prediction = evaluate(
        model,
        tokenizer,
        test_dataloader,
        generate_max_length,
        test_instances,
        sel2record, )

    better_print_results = defaultdict(dict)

    for k, v in eval_results.items():
        att = k.split('-')
        prefix = "-".join(att[:-1])
        metric = att[-1]
        better_print_results[prefix][metric] = v
        logger.info(f"  {k} = {v}")

    write_prediction(eval_prediction, 'test')


def write_prediction(eval_prediction, prefix='eval'):
    for pred_file, pred_result in eval_prediction.items():
        output_file = os.path.join(args.output_dir, prefix + '-' + pred_file)
        logger.info(f"writing to {output_file}")
        with open(output_file, 'w') as output:
            if isinstance(pred_result, list):
                for pred in pred_result:
                    if isinstance(pred_result, str):
                        output.write(pred + '\n')
                    else:
                        output.write(
                            json.dumps(
                                pred, ensure_ascii=False) + '\n')
            elif isinstance(pred_result, dict):
                for pred in pred_result:
                    output.write(f"{prefix}-{pred} = {pred_result[pred]}" +
                                 '\n')


def train(args, model, tokenizer):

    set_seed(args)

    generate_max_length = args.max_target_length

    writer = get_writer(args)

    # Distributed Setting
    if paddle.distributed.get_world_size() > 1:
        paddle.distributed.init_parallel_env()

    if paddle.distributed.get_world_size() > 1:
        model = paddle.DataParallel(model)

    # get dataloader
    train_dataloader = get_train_dataloader(
        model=model,
        tokenizer=tokenizer,
        train_filename=args.train_file,
        args=args, )
    dev_dataloader = get_eval_dataloader(
        model=model,
        tokenizer=tokenizer,
        eval_filename=args.validation_file,
        args=args, )
    # load dev texts and tokens
    val_instances = get_eval_instances(args.validation_file)

    map_config = MapConfig.load_from_yaml(args.map_config)
    schema_dict = SEL2Record.load_schema_dict(args.record_schema_dir)
    sel2record = SEL2Record(
        schema_dict=schema_dict,
        decoding_schema=args.decoding,
        map_config=map_config, )

    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_steps > 0:
        args.num_train_epochs = math.ceil(args.max_steps /
                                          num_update_steps_per_epoch)
    else:
        args.max_steps = args.num_train_epochs * num_update_steps_per_epoch

    # get lr_scheduler
    lr_scheduler = get_scheduler(
        learning_rate=args.learning_rate,
        scheduler_type=args.lr_scheduler_type,
        num_warmup_steps=args.warmup_steps
        if args.warmup_steps > 0 else args.warmup_ratio,
        num_training_steps=args.max_steps, )

    total_batch_size = args.per_device_train_batch_size * \
        args.gradient_accumulation_steps

    decay_params = [
        p.name for n, p in model.named_parameters()
        if not any(nd in n for nd in ["bias", "norm"])
    ]

    optimizer = AdamW(
        learning_rate=lr_scheduler,
        beta1=args.adam_beta1,
        beta2=args.adam_beta2,
        epsilon=args.adam_epsilon,
        parameters=model.parameters(),
        weight_decay=args.weight_decay,
        apply_decay_param_fun=lambda x: x in decay_params, )

    if args.use_amp:
        scaler = GradScaler(init_loss_scaling=args.scale_loss)

    logger.info("********** Running training **********")
    logger.info(f"  Num examples = {len(train_dataloader.dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(
        f"  Device train batch size = {args.per_device_train_batch_size}")
    logger.info(
        f"  Device eval  batch size = {args.per_device_eval_batch_size}")
    logger.info(
        f"  Total  train batch size (w. accumulation) = {total_batch_size}")
    logger.info(
        f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_steps}")

    progress_bar = tqdm(range(args.max_steps))

    global_steps = 0
    tr_loss, logging_loss = 0.0, 0.0

    best_score = 0.0

    def logging_lr_loss():
        cur_lr = lr_scheduler.get_lr()
        cur_loss = (tr_loss - logging_loss) / args.logging_steps
        writer.add_scalar("lr", cur_lr, global_steps)
        writer.add_scalar("loss", cur_loss, global_steps)
        logger.info("global_steps {} - lr: {:.10f}  loss: {:.10f}".format(
            global_steps, cur_lr, cur_loss))

    for epoch in range(args.num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            model.train()

            with auto_cast(
                    args.use_amp, custom_white_list=["layer_norm", "softmax"]):
                outputs = model(**batch)
                loss = outputs[0] / args.gradient_accumulation_steps
                tr_loss += loss.item()

            if args.use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (step % args.gradient_accumulation_steps == 0 or
                    step == len(train_dataloader) - 1):
                if args.use_amp:
                    scaler.minimize(optimizer, loss)
                else:
                    optimizer.step()

                lr_scheduler.step()
                optimizer.clear_grad()
                global_steps += 1
                progress_bar.update(1)

                if args.logging_steps > 0 and global_steps % args.logging_steps == 0:

                    if paddle.distributed.get_rank() == 0:
                        logging_lr_loss()
                        logging_loss = tr_loss

            if global_steps > 0 and args.save_steps > 0 and global_steps % args.save_steps == 0:
                save_checkpoint(tokenizer, model,
                                os.path.join(args.output_dir,
                                             f"checkpoint-{global_steps}"))

        if paddle.distributed.get_rank() == 0:

            logger.info(f"********** Running evaluating **********")
            logger.info(f"************* Epoch {epoch} ************")

            eval_results, eval_prediction = evaluate(
                model=model,
                tokenizer=tokenizer,
                data_loader=dev_dataloader,
                generate_max_length=generate_max_length,
                eval_instances=val_instances,
                sel2record=sel2record, )

            for k, v in eval_results.items():
                writer.add_scalar(f"eval/{k}", v, global_steps)
                # logger.info(f"  {k} = {v}")

            for line in better_print(eval_results).split('\n'):
                logger.info(line)

            if args.metric_for_best_model not in eval_results:
                raise ValueError(f"Main metric {args.metric_for_best_model} "
                                 f"is not in {eval_results.keys()}.")

            logger.info("********** Evaluating Done **********")

            if eval_results[args.metric_for_best_model] > best_score:

                logger.info("********** Saving Model **********")
                best_score = eval_results[args.metric_for_best_model]
                save_checkpoint(tokenizer, model, args.output_dir)
                write_prediction(eval_prediction, 'valid')


def main(args):
    logger.info("**********  Configuration Arguments **********")
    for arg, value in sorted(vars(args).items()):
        logger.info(f"{arg}: {value}")
    logger.info("**********************************************")

    if os.path.exists(args.output_dir
                      ) and args.do_train and not args.overwrite_output_dir:
        raise ValueError(
            f"Output directory ({args.output_dir}) already exists and is not empty. "
            "Use --overwrite_output_dir to overcome.")
    else:
        os.makedirs(args.output_dir, exist_ok=True)

    # Set device
    paddle.set_device(args.device)

    # get model and tokenizer
    if 'char' in args.model_name_or_path:
        tokenizer_type = T5BertTokenizer
    else:
        tokenizer_type = T5Tokenizer

    tokenizer = tokenizer_type.from_pretrained(args.model_name_or_path, )
    model = T5ForConditionalGeneration.from_pretrained(args.model_name_or_path)

    if args.do_train:
        train(args, model, tokenizer)
    if args.do_predict:
        test(args, model, tokenizer)


if __name__ == "__main__":
    args = parse_args()
    os.makedirs("caches", exist_ok=True)
    os.makedirs(args.logging_dir, exist_ok=True)
    set_logger(args)
    logger.info(args)
    main(args)
