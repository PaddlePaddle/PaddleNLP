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
    DataLoader,
)
from uie.evaluation import constants
from uie.evaluation.sel2record import (
    SEL2Record,
    RecordSchema,
    MapConfig,
    merge_schema,
)
from uie.seq2struct.t5_bert_tokenizer import T5BertTokenizer
from uie.seq2struct.data_collator import (
    DataCollatorForSeq2Seq,
    DynamicSSIGenerator,
    DataCollatorForMultiTaskSeq2Seq,
    SpotAsocNoiser,
)

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
    **scheduler_kwargs,
):
    """ Set learning rate scheduler """

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
    """ Set default seed """
    random.seed(args.seed + paddle.distributed.get_rank())
    np.random.seed(args.seed + paddle.distributed.get_rank())
    paddle.seed(args.seed + paddle.distributed.get_rank())


def save_checkpoint(tokenizer, model, output_dir):
    """ Save tokenizer and checkpoint model to output_dir """
    logger.info(f"saving checkpoint to {output_dir}")
    if isinstance(model, paddle.DataParallel):
        model = model._layers
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


def set_logger(args):
    """ Set logger """
    logger.setLevel(logging.DEBUG if 'DEBUG' in os.environ else logging.INFO)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=[
            logging.FileHandler(
                filename=f"{args.output_dir}.log",
                mode="w",
                encoding="utf-8",
            )
        ],
    )
    # create console handler and set level to debug
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level=logging.DEBUG)
    # add formatter to console_handler
    console_handler.setFormatter(fmt=logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    # add console_handler to logger
    logger.addHandler(console_handler)


def read_json_file(file_name):
    """ Read jsonline file as generator """
    with open(file_name, encoding='utf8') as fin:
        for line in fin:
            yield json.loads(line)


def better_print_multi(results):
    """ Better print multi task results
    results: Dictionary of task and metric {"task:metric": "value", ...}
    """
    table = [(task, results[task]) for task in results]
    return tabulate.tabulate(table, headers=['Task', 'Metric'])


def read_func(tokenizer,
              data_file: str,
              max_source_length: int,
              is_train: bool = False,
              negative_keep: float = 1.0):
    """ Read instance from data_file

    Args:
        tokenizer (PretrainedTokenizer): Tokenizer
        data_file (str): Data filename
        max_source_length (int): Max source length
        is_train (bool): instance from this file whether for training
        negative_keep (float): the ratio of keeping negative instances
    """

    negative_drop_num = 0
    with open(data_file, 'r', encoding='utf-8') as fin:
        for line in fin:
            instance = json.loads(line)

            # Drop negative sample in random during training stage
            if is_train and len(instance['spot_asoc']) == 0:
                # if negative_keep >= 1, keep all negative instances
                # else drop negative instance when random() > negative_keep
                if random.random() > negative_keep:
                    negative_drop_num += 1
                    continue

            inputs = tokenizer(
                instance['text'],
                return_token_type_ids=False,
                return_attention_mask=True,
                max_seq_len=max_source_length,
            )

            # `sample_ssi` can be True in the training stage
            # `sample_ssi` can only be False in the evaluation stage
            # 在训练时，ssi可以动态变化 (sample_ssi=True)
            # 但是在推理和验证时，ssi必须固定保证推理结果的一致 (sample_ssi=False)
            inputs.update({
                'spots': instance['spot'],
                'asocs': instance['asoc'],
                'spot_asoc': instance['spot_asoc'],
                'sample_ssi': is_train
            })
            yield inputs

    if negative_drop_num > 0:
        logger.info(
            f'Drop negative {negative_drop_num} instance during loading {data_file}.'
        )


def read_training_instance_based_config(tokenizer,
                                        config_file: str,
                                        max_source_length: int,
                                        negative_keep: float = 1.0):
    """Read training instances based on config_file

    Args:
        tokenizer (PretrainedTokenizer): Tokenizer
        config_file (str): Config filename
        max_source_length (int): Max source length
        negative_keep: the ratio of keeping negative instances

    Yields:
        dict: instance for training
    """
    task_configs = list(TaskConfig.load_list_from_yaml(config_file))

    for task_config in task_configs:
        negative_drop_num = 0

        train_file = os.path.join(task_config.data_path, "train.json")
        schema_file = os.path.join(task_config.data_path, "record.schema")
        record_schema = RecordSchema.read_from_file(schema_file)
        with open(train_file, 'r', encoding='utf-8') as fin:
            count = 0
            for line in fin:
                instance = json.loads(line)

                # Drop negative sample in random during training stage
                if len(instance['spot_asoc']) == 0:
                    # if negative_keep >= 1, keep all negative instances
                    # else drop negative instance when random() > negative_keep
                    if random.random() > negative_keep:
                        negative_drop_num += 1
                        continue

                inputs = tokenizer(instance['text'],
                                   return_token_type_ids=False,
                                   return_attention_mask=True,
                                   max_seq_len=max_source_length)

                # `sample_ssi` is True in the training stage
                inputs.update({
                    'spots': record_schema.type_list,
                    'asocs': record_schema.role_list,
                    'spot_asoc': instance['spot_asoc'],
                    'sample_ssi': True
                })
                yield inputs
                count += 1
            logger.info(f"Load {count} instances from {train_file}")

        if negative_drop_num > 0:
            logger.info(
                f'Drop negative {negative_drop_num} instance during loading {train_file}.'
            )


def get_train_dataloader(model, tokenizer, args):
    logger.info(f'Load data according to {args.multi_task_config} ...')

    dataset = load_dataset(read_training_instance_based_config,
                           tokenizer=tokenizer,
                           config_file=args.multi_task_config,
                           max_source_length=args.max_source_length,
                           lazy=False,
                           negative_keep=args.negative_keep)

    # Merge schema in all datasets for pre-tokenize
    schema_list = list()
    for task_config in TaskConfig.load_list_from_yaml(args.multi_task_config):
        schema_file = os.path.join(task_config.data_path, "record.schema")
        schema_list += [RecordSchema.read_from_file(schema_file)]
    schema = merge_schema(schema_list)

    batch_sampler = DistributedBatchSampler(
        dataset=dataset,
        batch_size=args.per_device_train_batch_size,
        shuffle=True,
    )

    if args.spot_noise > 0 or args.asoc_noise > 0:
        spot_asoc_nosier = SpotAsocNoiser(
            spot_noise_ratio=args.spot_noise,
            asoc_noise_ratio=args.asoc_noise,
            null_span=constants.null_span,
        )
    else:
        spot_asoc_nosier = None

    label_pad_token_id = -100 if args.ignore_pad_token_for_loss else tokenizer.pad_token_id

    collate_fn = DataCollatorForMultiTaskSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        max_source_length=args.max_source_length,
        max_prefix_length=args.max_prefix_length,
        max_target_length=args.max_target_length,
        ssi_generator=DynamicSSIGenerator(
            tokenizer=tokenizer,
            schema=schema,
            positive_rate=args.meta_positive_rate,
            negative=args.meta_negative,
            ordered_prompt=args.ordered_prompt,
        ),
        spot_asoc_nosier=spot_asoc_nosier,
    )

    data_loader = DataLoader(dataset=dataset,
                             batch_sampler=batch_sampler,
                             collate_fn=collate_fn,
                             num_workers=args.dataloader_num_workers,
                             return_list=True)

    return data_loader


def get_eval_dataloader(model, tokenizer, eval_filename, record_schema, args):
    """ Get evaluation dataloader
    """

    logger.info(f'Load data from {eval_filename} ...')

    schema = RecordSchema.read_from_file(record_schema)

    dataset = load_dataset(read_func,
                           tokenizer=tokenizer,
                           data_file=eval_filename,
                           max_source_length=args.max_source_length,
                           is_train=False,
                           lazy=False)

    batch_sampler = BatchSampler(dataset=dataset,
                                 batch_size=args.per_device_eval_batch_size,
                                 shuffle=False)

    label_pad_token_id = -100 if args.ignore_pad_token_for_loss else tokenizer.pad_token_id

    collate_fn = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        max_source_length=args.max_source_length,
        max_prefix_length=args.max_prefix_length,
        max_target_length=args.max_target_length,
        ssi_generator=DynamicSSIGenerator(
            tokenizer=tokenizer,
            schema=schema,
            positive_rate=1,
            negative=-1,
            ordered_prompt=True,
        ),
        spot_asoc_nosier=None,
    )

    data_loader = DataLoader(dataset=dataset,
                             batch_sampler=batch_sampler,
                             collate_fn=collate_fn,
                             num_workers=args.dataloader_num_workers,
                             return_list=True)

    return data_loader


def load_eval_tasks(model, tokenizer, args):
    """ Load evaluation tasks

    Args:
        model (PretrainedModel): Pretrain Model
        tokenizer (PretrainedTokenizer): Tokenizer
        args (Namespace): arguments for loading eval tasks

    Returns:
        list(Task): list of evaluation tasks
    """
    eval_tasks = dict()
    task_configs = list(TaskConfig.load_list_from_yaml(args.multi_task_config))

    for task_config in task_configs:

        val_filename = os.path.join(task_config.data_path, 'val.json')
        record_schema = os.path.join(task_config.data_path, 'record.schema')

        task_dataloader = get_eval_dataloader(model=model,
                                              tokenizer=tokenizer,
                                              eval_filename=val_filename,
                                              record_schema=record_schema,
                                              args=args)

        sel2record = SEL2Record(
            schema_dict=SEL2Record.load_schema_dict(task_config.data_path),
            map_config=MapConfig.load_by_name(task_config.sel2record),
            tokenizer=tokenizer
            if isinstance(tokenizer, T5BertTokenizer) else None,
        )

        eval_tasks[task_config.dataset_name] = Task(
            config=task_config,
            dataloader=task_dataloader,
            sel2record=sel2record,
            val_instances=list(read_json_file(val_filename)),
            metrics=task_config.metrics,
        )

    return eval_tasks


def write_prediction(eval_prediction, output_dir, prefix='eval'):
    """Write prediction to output_dir

    Args:
        eval_prediction (dict):
            - `record` (list(dict)), each element is extraction reocrd
            - `sel` (list(str)): each element is sel expression
            - `metric` (dict)
        output_dir (str): Output directory path
        prefix (str, optional): prediction file prefix. Defaults to 'eval'.

    Write prediction to files:
        - `preds_record.txt`, each line is extracted record
        - `preds_seq2seq.txt`, each line is generated sel
        - `results.txt`, detailed metrics of prediction
    """
    output_filename = os.path.join(output_dir, f"{prefix}-preds_record.txt")
    with open(output_filename, 'w', encoding='utf8') as output:
        for pred in eval_prediction.get('record', []):
            output.write(json.dumps(pred, ensure_ascii=False) + '\n')

    output_filename = os.path.join(output_dir, f"{prefix}-preds_seq2seq.txt")
    with open(output_filename, 'w', encoding='utf8') as output:
        for pred in eval_prediction.get('sel', []):
            output.write(pred + '\n')

    output_filename = os.path.join(output_dir, f"{prefix}-results.txt")
    with open(output_filename, 'w', encoding='utf8') as output:
        for key, value in eval_prediction.get('metric', {}).items():
            output.write(f"{prefix}-{key} = {value}\n")


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
        task_config_list = [
            f"dataset: {self.dataset_name}", f"task   : {self.task_name}",
            f"path   : {self.data_path}", f"schema : {self.schema}",
            f"metrics: {self.metrics}",
            f"eval_match_mode : {self.eval_match_mode}"
        ]
        return '\n'.join(task_config_list)

    @staticmethod
    def load_list_from_yaml(task_config):
        import yaml
        configs = yaml.load(open(task_config, encoding='utf8'),
                            Loader=yaml.FullLoader)
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
