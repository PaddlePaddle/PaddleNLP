#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import logging
import argparse
from tqdm import tqdm
import math
import os
import json

import paddle

from paddle.optimizer import AdamW
from paddle.amp import GradScaler, auto_cast
from paddlenlp.transformers import T5ForConditionalGeneration

from uie.evaluation import constants
from uie.evaluation.sel2record import evaluate_extraction_results
from uie.seq2struct.t5_bert_tokenizer import T5BertTokenizer
from uie.seq2struct.utils import (
    get_scheduler,
    get_writer,
    set_seed,
    save_checkpoint,
    set_logger,
    better_print_multi,
    get_train_dataloader,
    load_eval_tasks, )

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--multi_task",
        action="store_true",
        help="Path to pre-trained model or shortcut name of model.", )
    parser.add_argument(
        "--multi_task_config",
        required=True,
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
        "--record_schema",
        default=None,
        type=str,
        help="Schema for information extraction.", )
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
        '--negative_keep',
        type=float,
        default=1.0,
        help="The keep rate of negative instance for fast training.")
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
        '--offset_map',
        dest='map_config',
        help='Offset match strategy.',
        choices=constants.offset_map_strategy.keys())
    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--do_eval', action='store_true')
    parser.add_argument('--do_predict', action='store_true')
    parser.add_argument(
        "--device",
        type=str,
        default="gpu",
        choices=["cpu", "gpu"],
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
             sel2record, eval_match_mode):

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
    for batch in tqdm(data_loader, desc='evaluate'):
        source_ids, source_mask = batch['input_ids'], batch['attention_mask']

        # We can no longer use model.generate because it fails in distributed envs
        outputs, scores = model.generate(
            input_ids=source_ids,
            attention_mask=source_mask,
            max_length=generate_max_length,
            use_faster=True, )

        # Convert Token id to Token String
        outputs = tokenizer.batch_decode(
            outputs,
            clean_up_tokenization_spaces=False,
            skip_special_tokens=False)

        preds = [postprocess_text(o) for o in outputs]
        all_preds.extend(preds)

    assert len(all_preds) == len(eval_instances)

    # Parsing SEL to Record
    all_records = []
    for p, instance in zip(all_preds, eval_instances):
        r = sel2record.sel2record(
            pred=p, text=instance['text'], tokens=instance['tokens'])
        all_records += [r]

    results = evaluate_extraction_results(
        eval_instances, all_records, eval_match_mode=eval_match_mode)

    prediction = {
        'preds_record.txt': all_records,
        'preds_seq2seq.txt': all_preds,
        'results.txt': results
    }

    return results, prediction


def eval_all_tasks(eval_tasks, model, tokenizer, generate_max_length):
    eval_overall_results = dict()
    eval_overall_predictions = dict()
    for task_name, eval_task in eval_tasks.items():
        logger.info(f"Evaluate {task_name} ...")
        eval_results, eval_prediction = evaluate(
            model=model,
            tokenizer=tokenizer,
            data_loader=eval_task.dataloader,
            generate_max_length=generate_max_length,
            eval_instances=eval_task.val_instances,
            sel2record=eval_task.sel2record,
            eval_match_mode=eval_task.config.eval_match_mode, )
        for metric_name in eval_task.metrics:
            eval_overall_results[f"{task_name}:{metric_name}"] = eval_results[
                metric_name]
        eval_overall_predictions[task_name] = eval_prediction

    metric_sum = sum(eval_overall_results.values())
    metric_num = len(eval_overall_results.values())
    eval_overall_results['all-task-ave'] = metric_sum / float(metric_num)

    for line in better_print_multi(eval_overall_results).split('\n'):
        logger.info(line)

    return eval_overall_results, eval_overall_predictions


def test(args, model, tokenizer):
    generate_max_length = args.max_target_length
    eval_tasks = load_eval_tasks(model=model, tokenizer=tokenizer, args=args)

    eval_overall_results, eval_overall_predictions = eval_all_tasks(
        eval_tasks=eval_tasks,
        model=model,
        tokenizer=tokenizer,
        generate_max_length=generate_max_length, )

    better_print_multi(eval_overall_results)

    for task_name in eval_overall_predictions:
        write_prediction(
            eval_prediction=eval_overall_predictions[task_name],
            prefix='test-' + task_name, )


def write_prediction(eval_prediction, prefix='eval'):

    for pred_file, pred_result in eval_prediction.items():
        output_filename = os.path.join(args.output_dir, f"{prefix}-{pred_file}")

        with open(output_filename, 'w', encoding='utf8') as output:
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
        model = paddle.DataParallel(model)

    # get dataloader
    train_dataloader = get_train_dataloader(
        model=model,
        tokenizer=tokenizer,
        train_filename=args.train_file,
        args=args, )
    eval_tasks = load_eval_tasks(
        model=model, tokenizer=tokenizer, args=args) if args.do_eval else None

    def math_ceil(x, y):
        return math.ceil(x / float(y))

    num_update_steps_per_epoch = math_ceil(
        len(train_dataloader), args.gradient_accumulation_steps)
    if args.max_steps > 0:
        args.num_train_epochs = math_ceil(args.max_steps,
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

        save_checkpoint(tokenizer, model, args.output_dir)
        if args.do_eval and paddle.distributed.get_rank() == 0:

            logger.info(f"********** Running evaluating **********")
            logger.info(f"************* Epoch {epoch} ************")

            eval_overall_results, eval_overall_predictions = eval_all_tasks(
                eval_tasks=eval_tasks,
                model=model,
                tokenizer=tokenizer,
                generate_max_length=generate_max_length, )

            for task_name in eval_overall_predictions:
                write_prediction(
                    eval_overall_predictions[task_name],
                    'valid-' + task_name, )

            if args.metric_for_best_model not in eval_overall_results:
                raise ValueError(f"Main metric {args.metric_for_best_model} "
                                 f"is not in {eval_overall_results.keys()}.")

            logger.info("********** Evaluating Done **********")
            current_score = eval_overall_results[args.metric_for_best_model]
            if current_score > best_score:
                logger.info("********** Saving Model **********")
                best_score = current_score
                save_checkpoint(tokenizer, model,
                                os.path.join(args.output_dir, f"best"))


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

    tokenizer = T5BertTokenizer.from_pretrained(args.model_name_or_path, )
    model = T5ForConditionalGeneration.from_pretrained(args.model_name_or_path)

    if args.do_train:
        train(args, model, tokenizer)

    if args.do_predict:
        test(args, model, tokenizer)

    logger.info(f"Output Dir: {args.output_dir}")


if __name__ == "__main__":
    args = parse_args()
    os.makedirs("caches", exist_ok=True)
    os.makedirs(args.logging_dir, exist_ok=True)
    set_logger(args)
    logger.info(args)
    main(args)
