#!/usr/bin/env python3
# -*- coding:utf-8 -*-
from typing import List
import json
import random
import os
import logging
import tabulate
import numpy as np
from dataclasses import dataclass
import paddle
from paddlenlp.transformers import (CosineDecayWithWarmup,
                                    LinearDecayWithWarmup, PolyDecayWithWarmup)

from paddlenlp.datasets import load_dataset
from paddle.io import (
    DistributedBatchSampler,
    BatchSampler,
    DataLoader, )
from uie.evaluation import constants
from uie.evaluation.sel2record import SEL2Record, RecordSchema, MapConfig
from uie.seq2struct.t5_bert_tokenizer import T5BertTokenizer
from uie.seq2struct.data_collator import (
    DataCollatorForSeq2Seq,
    DynamicSSIGenerator,
    DataCollatorForMultiTaskSeq2Seq,
    DynamicMultiTaskSSIGenerator,
    SpotAsocNoiser, )

logger = logging.getLogger("__main__")


def get_writer(args):
    if args.writer_type == "visualdl":
        from visualdl import LogWriter

        writer = LogWriter(logdir=args.logging_dir)
    elif args.writer_type == "tensorboard":
        from tensorboardX import SummaryWriter

        writer = SummaryWriter(logdir=args.logging_dir)
    else:
        raise ValueError("writer_type must be in ['visualdl', 'tensorboard']")
    return writer


scheduler_type2cls = {
    "linear": LinearDecayWithWarmup,
    "cosine": CosineDecayWithWarmup,
    "poly": PolyDecayWithWarmup,
}


def get_scheduler(
        learning_rate,
        scheduler_type,
        num_warmup_steps=None,
        num_training_steps=None,
        **scheduler_kwargs, ):

    if scheduler_type not in scheduler_type2cls.keys():
        data = " ".join(scheduler_type2cls.keys())
        raise ValueError(f"scheduler_type must be choson from {data}")

    if num_warmup_steps is None:
        raise ValueError(
            f"requires `num_warmup_steps`, please provide that argument.")

    if num_training_steps is None:
        raise ValueError(
            f"requires `num_training_steps`, please provide that argument.")

    return scheduler_type2cls[scheduler_type](learning_rate=learning_rate,
                                              total_steps=num_training_steps,
                                              warmup=num_warmup_steps,
                                              **scheduler_kwargs)


def set_seed(args):
    random.seed(args.seed + paddle.distributed.get_rank())
    np.random.seed(args.seed + paddle.distributed.get_rank())
    paddle.seed(args.seed + paddle.distributed.get_rank())


def save_checkpoint(tokenizer, model, output_dir):
    logger.info(f"saving checkpoint to {output_dir}")
    if isinstance(model, paddle.DataParallel):
        model = model._layers
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


def set_logger(args):
    logger.setLevel(logging.DEBUG if 'DEBUG' in os.environ else logging.INFO)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=[
            logging.FileHandler(
                filename=f"{args.output_dir}.log",
                mode="w",
                encoding="utf-8", )
        ], )
    # create console handler and set level to debug
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level=logging.DEBUG)
    # add formatter to console_handler
    console_handler.setFormatter(fmt=logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    # add console_handler to logger
    logger.addHandler(console_handler)


def read_json_file(file_name):
    return [json.loads(line) for line in open(file_name, encoding='utf8')]


def better_print_multi(results):
    table = [(task, results[task]) for task in results]
    return tabulate.tabulate(table, headers=['Task', 'Metric'])


def read_func(tokenizer,
              data_file,
              max_source_length,
              max_target_length,
              is_train=False,
              negative_keep=1.0):
    def tokenize(x, max_length):
        return tokenizer(
            x,
            return_token_type_ids=False,
            return_attention_mask=True,
            max_seq_len=max_length, )

    negative_drop_count = 0
    with open(data_file, 'r', encoding='utf-8') as f:
        for line in f:
            instance = json.loads(line)

            if is_train and len(instance['spot_asoc']) == 0:
                if random.random() > negative_keep:
                    negative_drop_count += 1
                    continue

            inputs = tokenize(instance['text'], max_source_length)
            inputs.update({
                'spots': instance['spot'],
                'asocs': instance['asoc'],
                'spot_asoc': instance['spot_asoc'],
                'sample_prompt': [is_train] *
                len(inputs['input_ids'])  # Sample SSI during Training
            })
            yield inputs
    if negative_drop_count > 0:
        logger.info(
            f'Drop negative {negative_drop_count} instance during loading {data_file}.'
        )


def get_train_dataloader(model, tokenizer, train_filename, args):
    logger.info(f'Load data from {train_filename} ...')
    schema = RecordSchema.read_from_file(args.record_schema)

    dataset = load_dataset(
        read_func,
        tokenizer=tokenizer,
        data_file=train_filename,
        max_source_length=args.max_source_length,
        max_target_length=args.max_target_length,
        is_train=True,
        lazy=False,
        negative_keep=args.negative_keep)

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
        spot_asoc_nosier=spot_asoc_nosier, )

    data_loader = DataLoader(
        dataset=dataset,
        batch_sampler=batch_sampler,
        collate_fn=collate_fn,
        num_workers=args.dataloader_num_workers,
        return_list=True)

    return data_loader


def get_eval_dataloader(model, tokenizer, eval_filename, record_schema, args):

    logger.info(f'Load data from {eval_filename} ...')

    schema = RecordSchema.read_from_file(record_schema)

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
        spot_asoc_nosier=None, )

    data_loader = DataLoader(
        dataset=dataset,
        batch_sampler=batch_sampler,
        collate_fn=collate_fn,
        num_workers=args.dataloader_num_workers,
        return_list=True)

    return data_loader


def load_eval_tasks(model, tokenizer, args):

    # Evaluate performance of different tasks
    eval_tasks = dict()
    task_configs = list(TaskConfig.load_list_from_yaml(args.multi_task_config))

    for task_config in task_configs:

        task_dataloader = get_eval_dataloader(
            model=model,
            tokenizer=tokenizer,
            eval_filename=os.path.join(task_config.data_path, 'val.json'),
            record_schema=os.path.join(task_config.data_path, 'record.schema'),
            args=args)

        sel2record = SEL2Record(
            schema_dict=SEL2Record.load_schema_dict(task_config.data_path),
            map_config=MapConfig.load_by_name(task_config.sel2record),
            tokenizer=tokenizer
            if isinstance(tokenizer, T5BertTokenizer) else None, )

        eval_tasks[task_config.dataset_name] = Task(
            config=task_config,
            dataloader=task_dataloader,
            sel2record=sel2record,
            val_instances=read_json_file(
                os.path.join(task_config.data_path, 'val.json')),
            metrics=task_config.metrics, )

    return eval_tasks


class TaskConfig:
    def __init__(self, task_dict) -> None:
        self.dataset_name = task_dict.get('name', '')
        self.task_name = task_dict.get('task', '')
        self.data_path = task_dict.get('path', '')
        self.sel2record = task_dict.get('sel2record', '')
        self.metrics = task_dict.get('metrics', [])
        self.eval_match_mode = task_dict.get('eval_match_mode', 'normal')
        self.schema = RecordSchema.read_from_file(
            f"{self.data_path}/record.schema")

    def __repr__(self) -> str:
        return f"dataset: {self.dataset_name}\n" \
               f"task   : {self.task_name}\n" \
               f"path   : {self.data_path}\n" \
               f"schema : {self.schema}\n" \
               f"metrics: {self.metrics}\n" \
               f"eval_match_mode : {self.eval_match_mode}"

    @staticmethod
    def load_list_from_yaml(task_config):
        import yaml
        configs = yaml.load(
            open(
                task_config, encoding='utf8'), Loader=yaml.FullLoader)
        task_configs = filter(lambda x: x.startswith('T'), configs)
        for task_config in task_configs:
            yield TaskConfig(configs[task_config])


@dataclass
class Task:
    config: TaskConfig
    dataloader: DataLoader
    sel2record: SEL2Record
    val_instances: List[dict]
    metrics: List[str]
